#!/usr/bin/env python3
"""
Visualize flow dependency graph for a specific GPU from gpu_flows_parsed.txt.

Usage:
    python visualize_flows.py                          # GPU 0, all channels
    python visualize_flows.py --gpu 0 --channel 0      # GPU 0, channel 0 only
    python visualize_flows.py --gpu 0 --channel 0 1 2  # GPU 0, channels 0,1,2
"""

import csv
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math

# ── Color Palette ──
COLOR_OUTGOING = "#8B0000"   # dark red  — flows SENT by this GPU (src=gpu)
COLOR_INCOMING = "#191970"   # midnight blue — flows RECEIVED by this GPU (dst=gpu)
COLOR_ROOT     = "#2F4F4F"   # dark slate gray — the virtual -1 node
EDGE_COLOR     = "#555555"
BG_COLOR       = "#0E0E12"

# Premium palette for Mode 3 (color by GPU rank)
GPU_COLORS = [
    "#38bdf8",  # GPU 0: sky blue
    "#fb7185",  # GPU 1: rose
    "#34d399",  # GPU 2: emerald
    "#fbbf24",  # GPU 3: amber
    "#c084fc",  # GPU 4: purple
    "#22d3ee",  # GPU 5: cyan
    "#f472b6",  # GPU 6: pink
    "#a7f3d0",  # GPU 7: light emerald
    "#60a5fa",  # GPU 8: blue
    "#f87171",  # GPU 9: red
    "#fb923c",  # GPU 10: orange
    "#4ade80",  # GPU 11: green
    "#818cf8",  # GPU 12: indigo
    "#e879f9",  # GPU 13: magenta
    "#2dd4bf",  # GPU 14: teal
    "#e2e8f0",  # GPU 15: slate
]


def load_flows(filepath, target_gpu=None, channels=None):
    """Load flows. If target_gpu is specified, filter to those involving target_gpu."""
    flows = {}
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = int(row["src"])
            dst = int(row["dst"])
            if target_gpu is not None and src != target_gpu and dst != target_gpu:
                continue
            ch = int(row["channel_id"])
            if channels is not None and ch not in channels:
                continue
            fid = int(row["flow_id"])
            flows[fid] = {
                "src": src,
                "dst": dst,
                "size": int(row["size"]),
                "channel_id": ch,
                "chunk_id": int(row["chunk_id"]),
                "parent": int(row["parent_flow_id"]),
                "child": int(row["child_flow_id"]),
            }
    return flows


def build_graph(flows, target_gpu=None):
    """Build a networkx DiGraph from the flow dependency data."""
    G = nx.DiGraph()

    # Virtual root node
    G.add_node("START", kind="root", label="START\n(no parent)")
    G.add_node("END",   kind="root", label="END\n(no child)")

    flow_ids_in_graph = set(flows.keys())

    for fid, info in flows.items():
        if target_gpu is not None:
            is_outgoing = (info["src"] == target_gpu)
            kind = "out" if is_outgoing else "in"
        else:
            kind = f"gpu_{info['src']}"

        G.add_node(
            fid,
            kind=kind,
            label=str(fid),
            src=info["src"],
            dst=info["dst"],
            channel=info["channel_id"],
            chunk=info["chunk_id"],
        )

        # Parent edge
        parent = info["parent"]
        if parent == -1:
            G.add_edge("START", fid)
        elif parent in flow_ids_in_graph:
            G.add_edge(parent, fid)

        # Child edge
        child = info["child"]
        if child == -1:
            G.add_edge(fid, "END")
        elif child in flow_ids_in_graph:
            G.add_edge(fid, child)

    return G


def custom_layout(G, target_gpu=None):
    """
    Custom layout:
    If target_gpu is specified:
      START on left, END on right.
      Rows are channels. Within a row, chunk_id is x-position.
      Two flows per chunk: out on top, in on bottom.
    If target_gpu is None (Mode 3: all GPUs, specified channel):
      START on left, END on right.
      Rows are (channel, src_gpu). Column is chunk_id.
    """
    pos = {}
    
    # Identify non-root nodes
    non_root_nodes = []
    for n, data in G.nodes(data=True):
        if data.get("kind") == "root":
            continue
        non_root_nodes.append((n, data))
        
    if not non_root_nodes:
        pos["START"] = (-1, 0)
        pos["END"] = (1, 0)
        return pos

    x_scale = 1.2
    
    if target_gpu is not None:
        # Mode 1 & 2: Filtered by target_gpu. Group by channel.
        channels = {}
        for n, data in non_root_nodes:
            ch = data.get("channel", 0)
            kind = data.get("kind", "out")
            channels.setdefault(ch, []).append((n, data.get("chunk", 0), kind))
            
        sorted_chs = sorted(channels.keys())
        n_channels = len(sorted_chs)
        
        # Determine the max chunk_id for x-scaling
        all_chunks = set()
        for nodes in channels.values():
            for _, ck, _ in nodes:
                all_chunks.add(ck)
        max_ck = max(all_chunks) if all_chunks else 0
        
        total_height = n_channels * 1.0
        pos["START"] = (-1.2 * x_scale, total_height / 2)
        pos["END"]   = ((max_ck + 1.2) * x_scale, total_height / 2)
        
        for row_idx, ch in enumerate(sorted_chs):
            nodes = sorted(channels[ch], key=lambda t: (t[1], 0 if t[2] == "out" else 1))
            y_base = total_height - row_idx * 1.0
            for i, (nid, ck, kind) in enumerate(nodes):
                # Two flows per chunk: out on top, in on bottom
                y_offset = 0.2 if kind == "out" else -0.2
                pos[nid] = (ck * x_scale, y_base + y_offset)
                
    else:
        # Mode 3: All GPUs, specified channel(s). Group by (channel, src).
        row_keys = set()
        for n, data in non_root_nodes:
            ch = data.get("channel", 0)
            src = data.get("src", 0)
            row_keys.add((ch, src))
            
        sorted_rows = sorted(list(row_keys))  # sorted by (channel, src)
        row_to_idx = {key: idx for idx, key in enumerate(sorted_rows)}
        
        all_chunks = set()
        for n, data in non_root_nodes:
            all_chunks.add(data.get("chunk", 0))
        max_ck = max(all_chunks) if all_chunks else 0
        
        n_rows = len(sorted_rows)
        row_height = 0.7
        total_height = n_rows * row_height
        
        pos["START"] = (-1.2 * x_scale, total_height / 2)
        pos["END"]   = ((max_ck + 1.2) * x_scale, total_height / 2)
        
        for n, data in non_root_nodes:
            ch = data.get("channel", 0)
            src = data.get("src", 0)
            ck = data.get("chunk", 0)
            row_idx = row_to_idx[(ch, src)]
            y = total_height - row_idx * row_height
            pos[n] = (ck * x_scale, y)
            
    return pos


def draw_graph(G, target_gpu, channels_label, output_path):
    """Render the dependency graph."""
    pos = custom_layout(G, target_gpu)

    # Node properties
    node_colors = []
    node_sizes = []
    labels = {}
    
    # Legend items
    legend_items = []
    
    # Larger node sizes
    size_root = 1000
    size_node = 500
    
    if target_gpu is not None:
        # Mode 1 & 2
        for n, data in G.nodes(data=True):
            kind = data.get("kind", "root")
            if kind == "root":
                node_colors.append(COLOR_ROOT)
                node_sizes.append(size_root)
                labels[n] = data.get("label", str(n))
            elif kind == "out":
                node_colors.append(COLOR_OUTGOING)
                node_sizes.append(size_node)
                labels[n] = str(n)
            else:
                node_colors.append(COLOR_INCOMING)
                node_sizes.append(size_node)
                labels[n] = str(n)
                
        legend_items = [
            mpatches.Patch(color=COLOR_OUTGOING, label=f"Outgoing (src=GPU{target_gpu})"),
            mpatches.Patch(color=COLOR_INCOMING, label=f"Incoming (dst=GPU{target_gpu})"),
            mpatches.Patch(color=COLOR_ROOT,     label="Virtual START / END"),
        ]
    else:
        # Mode 3: All GPUs
        gpus_in_graph = set()
        for n, data in G.nodes(data=True):
            kind = data.get("kind", "root")
            if kind != "root" and "src" in data:
                gpus_in_graph.add(data["src"])
        sorted_gpus = sorted(list(gpus_in_graph))
        
        gpu_to_color = {gpu: GPU_COLORS[gpu % len(GPU_COLORS)] for gpu in sorted_gpus}
        
        for n, data in G.nodes(data=True):
            kind = data.get("kind", "root")
            if kind == "root":
                node_colors.append(COLOR_ROOT)
                node_sizes.append(size_root)
                labels[n] = data.get("label", str(n))
            else:
                src = data.get("src", 0)
                node_colors.append(gpu_to_color[src])
                node_sizes.append(size_node)
                labels[n] = str(n)
                
        legend_items = [
            mpatches.Patch(color=gpu_to_color[gpu], label=f"GPU {gpu}")
            for gpu in sorted_gpus
        ]
        legend_items.append(mpatches.Patch(color=COLOR_ROOT, label="Virtual START / END"))

    # Scale figure to content
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    if xs and ys:
        fig_w = max(12, 0.8 * (max(xs) - min(xs)))
        fig_h = max(6, 1.2 * (max(ys) - min(ys)))
    else:
        fig_w, fig_h = 12, 6
    fig_w = min(fig_w, 40)
    fig_h = min(fig_h, 30)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=EDGE_COLOR,
        arrows=True,
        arrowsize=12,
        arrowstyle="-|>",
        width=1.0,
        alpha=0.6,
        connectionstyle="arc3,rad=0.05",
        min_source_margin=12,
        min_target_margin=12,
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="#FFFFFF33",
        linewidths=0.5,
    )

    # Draw labels
    nx.draw_networkx_labels(
        G, pos, labels, ax=ax,
        font_size=8,
        font_color="white",
        font_weight="bold",
    )

    # Legend
    legend = ax.legend(
        handles=legend_items, loc="upper left",
        fontsize=10, facecolor="#1a1a2e", edgecolor="#444",
        labelcolor="white",
    )

    if target_gpu is not None:
        title = f"Flow Dependency Graph — GPU {target_gpu}"
        if channels_label:
            title += f" — Channels: {channels_label}"
    else:
        title = "Flow Dependency Graph — All GPUs"
        if channels_label:
            title += f" — Channels: {channels_label}"
    title += f" — {G.number_of_nodes()-2} flows, {G.number_of_edges()} edges"
    ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=15)

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor=BG_COLOR, bbox_inches="tight")
    print(f"[OK] Saved to: {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize flow dependency graph")
    parser.add_argument("--input", default="gpu_flows_parsed.txt",
                        help="Path to gpu_flows_parsed.txt")
    parser.add_argument("--gpu", type=int, default=None,
                        help="Target GPU ID to visualize. If omitted, plots all GPUs for specified channel(s).")
    parser.add_argument("--channel", type=int, nargs="*", default=None,
                        help="Filter by channel ID(s). If omitted, plots all channels.")
    parser.add_argument("--output", default=None,
                        help="Output image path")
    args = parser.parse_args()

    # If neither is specified, default to GPU 0 (Mode 2)
    if args.gpu is None and args.channel is None:
        args.gpu = 0

    if args.gpu is not None:
        # Mode 1 or 2
        mode_desc = f"GPU {args.gpu}"
        if args.channel is not None:
            # Mode 1: 指定gpu, 指定channel
            channels_set = set(args.channel)
            channels_label = ",".join(map(str, sorted(channels_set)))
            mode_desc += f", Channel {channels_label}"
        else:
            # Mode 2: 仅指定gpu (绘制所有channel)
            channels_set = None
            channels_label = "all"
            mode_desc += ", All Channels"
    else:
        # Mode 3: 仅指定channel (绘制所有gpu)
        if args.channel is not None:
            channels_set = set(args.channel)
            channels_label = ",".join(map(str, sorted(channels_set)))
        else:
            channels_set = None
            channels_label = "all"
        mode_desc = f"All GPUs, Channel {channels_label}"

    if args.output is None:
        gpu_part = f"gpu{args.gpu}" if args.gpu is not None else "allgpus"
        ch_part = f"_ch{'_'.join(map(str, args.channel))}" if args.channel is not None else ""
        args.output = f"flow_dep_{gpu_part}{ch_part}.png"

    print(f"Mode: {mode_desc}")
    print(f"Loading flows from {args.input} ...")
    flows = load_flows(args.input, args.gpu, channels_set)
    print(f"  Found {len(flows)} flows")

    if not flows:
        print("No flows found matching criteria. Check your --gpu and --channel arguments.")
        return

    G = build_graph(flows, args.gpu)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    draw_graph(G, args.gpu, channels_label, args.output)


if __name__ == "__main__":
    main()
