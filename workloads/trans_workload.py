#!/usr/bin/env python3
"""
simai2conweave.py

把 SimAI 的集合通信 workload 转换为 ConWeave/NS-3 接受的 p2p 流量文件。

要点：
- 解析 header（使用正则从 "key: value" 片段中提取参数，至少读取 all_gpus）
- 解析层行（按指定列索引），只处理 embedding_layer, attention_layer, mlp_layer, embedding_norm
- 使用 ring 展开集合通信为点对点：生成顺序为 for r in 1..(N-1): for i in 0..N-1: src=i, dst=(i+r)%N
- **不对 size 做 / N**（即 size 保持为 simai 提供的 comm_size）
- 时间戳：current_time 从 initial_time 开始；对每个集合通信（一个层的 forward 或 backward）
  - 为该集合通信的所有产生的 p2p flows 指定相同 timestamp = current_time
  - 然后 current_time += duration_for_this_comm
  - duration_for_this_comm 优先使用用户传入的 per-comm 固定值（如 ALLREDUCE=0.005），否则 fallback = size / bandwidth_bytes_per_sec * time_scale
- 输出格式：
    第一行：<flow_total_count>
    后续行：src dst 3 size_bytes timestamp_seconds
"""

import re
import argparse
import json
from typing import List, Tuple, Dict

TARGET_LAYERS = {"embedding_layer", "attention_layer", "mlp_layer", "embedding_norm"}
COLLECTIVE_NONE = {"NONE", "none", "None"}


def parse_header(header_line: str) -> Dict[str, object]:
    """用正则解析 header 中形如 key: value 的片段"""
    params = {}
    matches = re.findall(r"(\S+):\s*(\S+)", header_line)
    for k, v in matches:
        # 尝试转 int，否则保留字符串
        try:
            params[k] = int(v)
        except ValueError:
            params[k] = v
    return params


def parse_layer_line(line: str) -> Dict[str, object]:
    """
    解析一行 layer 信息，返回 dict，字段（基于你提供的格式顺序）：
    tokens indices:
    0 name
    1 placeholder
    2 forward_compute_time
    3 forward_comm
    4 forward_comm_size
    5 backward_compute_time
    6 backward_comm
    7 backward_comm_size
    8 dp_compute_time
    9 dp_comm
    10 dp_comm_size
    11 process_time
    如果行短缺字段，尽量容错并把缺失值设为默认。
    """
    tokens = line.strip().split()
    if not tokens:
        return None

    # Ensure length
    def safe_get(idx, default=None):
        try:
            return tokens[idx]
        except Exception:
            return default

    name = safe_get(0, "")
    f_comm = safe_get(3, "NONE")
    f_size_s = safe_get(4, "0")
    b_comm = safe_get(6, "NONE")
    b_size_s = safe_get(7, "0")
    # try parse sizes to int
    try:
        f_size = int(f_size_s)
    except Exception:
        f_size = 0
    try:
        b_size = int(b_size_s)
    except Exception:
        b_size = 0
    return {
        "name": name,
        "forward_comm": f_comm,
        "forward_size": f_size,
        "backward_comm": b_comm,
        "backward_size": b_size,
        "raw_tokens": tokens
    }


def expand_ring_p2p(comm_size: int, world_size: int) -> List[Tuple[int, int, int]]:
    """
    Ring 展开：对于 r=1..N-1, 对每个 i=0..N-1 生成 (i, (i+r)%N, comm_size)
    注意：按你的要求，这里**不做分块**，每条 p2p 的 size = comm_size。
    """
    flows = []
    N = world_size
    if N <= 1:
        return flows
    for r in range(1, N):
        for i in range(N):
            src = i
            dst = (i + r) % N
            flows.append((src, dst, comm_size))
    return flows


def parse_durations_arg(s: str) -> Dict[str, float]:
    """
    解析类似 "ALLREDUCE=0.005,ALLGATHER=0.004" 的输入到 dict
    """
    if not s:
        return {}
    out = {}
    parts = s.split(",")
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            try:
                out[k.strip()] = float(v.strip())
            except:
                pass
    return out


def build_comm_entries(layer_list: List[Dict]) -> List[Dict]:
    """
    从 parsed layer_list 构建一个 comm_sequence：
    - 先把每个 layer 的 forward（正序）加入（条目是 dict 包含 name, direction, comm, size）
    - 然后把每个 layer 的 backward（逆序）加入
    """
    entries = []
    # forward, in original order
    for layer in layer_list:
        c = layer["forward_comm"]
        s = layer["forward_size"]
        if c and c not in COLLECTIVE_NONE and s > 0:
            entries.append({
                "name": layer["name"],
                "direction": "forward",
                "comm": c,
                "size": s
            })
    # backward, reverse layer order
    for layer in reversed(layer_list):
        c = layer["backward_comm"]
        s = layer["backward_size"]
        if c and c not in COLLECTIVE_NONE and s > 0:
            entries.append({
                "name": layer["name"],
                "direction": "backward",
                "comm": c,
                "size": s
            })
    return entries


def comm_duration(comm_type: str, size: int, durations_map: Dict[str, float],
                  bandwidth: float, time_scale: float, min_duration: float = 1e-6) -> float:
    """返回该集合通信的增量时长（秒）"""
    if comm_type in durations_map:
        return float(durations_map[comm_type]) * float(time_scale)

    # fallback estimate: size / bandwidth
    if bandwidth and bandwidth > 0:
        d = float(size) / float(bandwidth)
        return max(d * float(time_scale), min_duration)
    return max(min_duration, float(time_scale) * 1e-3)


def generate_flows_from_entries(entries: List[Dict], world_size: int,
                                initial_time: float,
                                durations_map: Dict[str, float],
                                bandwidth: float,
                                time_scale: float) -> List[Tuple[int, int, int, float]]:
    """
    entries: list of {"name","direction","comm","size"} in the order we want to "execute"
    For each entry:
      - expand into p2p flows (ring)
      - assign timestamp = current_time to all these p2p flows
      - current_time += comm_duration(...)
    Returns list of (src, dst, size_bytes, timestamp_seconds)
    """
    flows = []
    current_time = float(initial_time)
    for ent in entries:
        comm = ent["comm"]
        size = int(ent["size"])
        # expand according to comm type. For now treat different comm types the same in expansion (ring),
        # because your requirement: use ring and no chunking.
        # (If you want special handling for ALLGATHER/REDUCESCATTER you can add here.)
        # We'll just call expand_ring_p2p with size unchanged.
        p2p_list = expand_ring_p2p(size, world_size)
        # assign timestamp current_time to all p2p in this entry
        for (src, dst, s) in p2p_list:
            flows.append((src, dst, s, current_time))
        # increment time
        delta = comm_duration(comm, size, durations_map, bandwidth, time_scale)
        current_time += delta
    return flows


def write_conweave_file(flows: List[Tuple[int, int, int, float]], out_path: str):
    total = len(flows)
    with open(out_path, "w") as fw:
        fw.write(f"{total}\n")
        for src, dst, size, ts in flows:
            # timestamp with 6 decimal places
            fw.write(f"{src} {dst} 3 {size} {ts:.6f}\n")


def main():
    p = argparse.ArgumentParser(description="Convert SimAI collective workload to ConWeave p2p flow file.")
    p.add_argument("--simai_input", type=str, default=None, help="SimAI workload file (text)")
    p.add_argument("--out_flow", type=str, default=None, help="Output ConWeave flow file")
    p.add_argument("--bandwidth", type=float, default=12.5e9,
                   help="Fallback network bandwidth in bytes/sec for duration estimation (default 12.5e9 B/s ~ 100Gbps)")
    p.add_argument("--time-scale", type=float, default=1.0,
                   help="Global time scale multiplier applied to computed durations")
    p.add_argument("--initial-time", type=float, default=0.0,
                   help="Initial timestamp in seconds (default 0.0)")
    p.add_argument("--durations", type=str, default="ALLREDUCE=0.5,ALLGATHER=0.4",
                   help="Comma separated per-comm durations")
    p.add_argument("--num-batches", type=int, default=1,
                   help="How many identical batches to generate sequentially (default 1)")
    args = p.parse_args()

    parameters = {
        'simai_input': './workloads/None-gpt_7B-world_size8-tp4-pp1-ep1-gbs16-mbs1-seq2048-MOE-False-GEMM-False-flash_attn'
                       '-False.txt',
        'out_flow': './test.txt',
        'bandwidth': 12.5e9,
        'time-scale': 1.0,
        'initial-time': 0.0,
        'durations': 'ALLREDUCE=0.5,ALLGATHER=0.4',
        'num-batches': 1,
    }

    for key, value in vars(args).items():
        if value is not None:
            parameters[key] = value

    with open(parameters['simai_input'], "r") as fr:
        lines = [ln for ln in fr.readlines() if ln.strip()]

    if not lines:
        raise SystemExit("输入文件为空")

    header_params = parse_header(lines[0])
    world_size = int(header_params.get("all_gpus", header_params.get("all_gpus:", 1) or 1))

    # parse layers (从第三行开始，跳过第二行)
    layer_lines = lines[2:]
    parsed_layers = []
    for ln in layer_lines:
        pl = parse_layer_line(ln)
        if pl is None:
            continue
        # keep only target layers
        if pl["name"] in TARGET_LAYERS:
            parsed_layers.append(pl)

    # Build comm entries for one batch (forward seq then backward reversed)
    base_entries = build_comm_entries(parsed_layers)
    durations_map = parse_durations_arg(parameters['durations'])

    # For multiple batches, replicate entries sequentially
    all_entries = []
    for b in range(parameters['num_batches']):
        all_entries.extend(base_entries)

    # Generate flows with timestamps
    flows = generate_flows_from_entries(all_entries, world_size,
                                        initial_time=parameters['initial_time'],
                                        durations_map=durations_map,
                                        bandwidth=parameters['bandwidth'],
                                        time_scale=parameters['time_scale'])

    # write out
    write_conweave_file(flows, parameters['out_flow'])
    print(f"Done. Parsed {len(parsed_layers)} target layers. Generated {len(flows)} flows.")


if __name__ == "__main__":
    main()
