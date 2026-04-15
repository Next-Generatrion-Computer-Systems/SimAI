# Vidur-AlibabaCloud 测试教程与测试报告

> 测试日期: 2026-04-14
> 测试环境: 8x NVIDIA H20-3e (143 GB each), Python 3.13.11

---

## 1. 环境安装步骤

### 1.1 Python 环境

```bash
# 使用 base conda 环境 (Python 3.10+)
conda activate base

# 或创建专用环境
conda env create -p ./env -f ./environment.yml
conda activate vidur
```

### 1.2 安装依赖

```bash
cd vidur-alibabacloud
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
pip install -r requirements-dev.txt -i https://mirrors.aliyun.com/pypi/simple/
```

### 1.3 数据准备

```bash
# 从上游 microsoft/vidur 获取 trace 文件
git clone https://github.com/microsoft/vidur.git /tmp/vidur
cp -r /tmp/vidur/data/processed_traces ./data/

# 从上游 microsoft/vidur 获取 profiling 数据 (Native Vidur 模式需要)
cp -r /tmp/vidur/data/profiling ./data/
```

准备完成后目录结构：

```
data/
├── processed_traces/    # trace 文件 (从 microsoft/vidur 拷贝)
│   ├── splitwise_conv.csv
│   ├── splitwise_code.csv
│   └── arxiv_summarization_stats_llama2_tokenizer_filtered_v2.csv
├── profiling/           # profiling 数据 (从 microsoft/vidur 拷贝, Native Vidur 模式需要)
│   ├── compute/
│   └── network/
├── hf_configs/          # 已包含在仓库中
└── aicb_workload/       # 已包含在仓库中
```

---

## 2. PD 分离单元测试

### 2.1 运行命令

```bash
cd vidur-alibabacloud
python -m pytest tests/test_pd_separation.py -v
```

### 2.2 测试结果

| 测试用例 | 状态 | 说明 |
|---------|------|------|
| `TestPDOff::test_config_init_defaults` | PASSED | PD 关闭时配置默认值验证 |
| `TestPDOff::test_mixed_mode_cluster` | PASSED | 混合模式集群创建验证 |
| `TestPDOn::test_pd_cluster_creation` | PASSED | PD 开启时集群创建验证 |
| `TestPDOn::test_per_phase_world_size` | PASSED | 各阶段 world size 计算验证 |
| `TestPDParamsFallback::test_none_fallback` | PASSED | 参数回退机制验证 |
| `TestPDParamsFallback::test_explicit_per_phase_params` | PASSED | 显式参数覆盖验证 |
| `TestIllegalPdNodeRatio::test_zero_ratio` | PASSED | 非法比例 0 拒绝验证 |
| `TestIllegalPdNodeRatio::test_negative_ratio` | PASSED | 非法负比例拒绝验证 |
| `TestIllegalPdNodeRatio::test_ratio_greater_than_one` | PASSED | 非法比例 >1 拒绝验证 |
| `TestNumPrefillReplicasPriority::test_explicit_prefill_replicas` | PASSED | 显式 prefill 副本数优先级验证 |

**结果: 10/10 通过, 耗时 0.16s**

---

## 3. 集成测试 — Llama-3-8B Native Vidur 模式

### 3.1 运行命令

```bash
cd vidur-alibabacloud

python -m vidur.main \
  --replica_config_pd_p2p_comm_bandwidth 800 \
  --replica_config_nvlink_bandwidth 1600 \
  --replica_config_rdma_bandwidth 800 \
  --replica_config_pd_p2p_comm_dtype float32 \
  --poisson_request_interval_generator_config_qps 100 \
  --synthetic_request_generator_config_num_requests 10 \
  --length_generator_config_type trace \
  --trace_request_length_generator_config_max_tokens 2048 \
  --trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv \
  --interval_generator_config_type poisson \
  --cluster_config_num_replicas 4 \
  --replica_config_pd_node_ratio 0.5 \
  --global_scheduler_config_type split_wise \
  --replica_scheduler_config_type split_wise \
  --replica_config_model_name meta-llama/Meta-Llama-3-8B \
  --replica_config_tensor_parallel_size 4 \
  --replica_config_num_pipeline_stages 1 \
  --random_forrest_execution_time_predictor_config_backend vidur
```

### 3.2 测试结果

| 项目 | 结果 |
|------|------|
| 状态 | **通过** |
| 模拟结束时间 | 4.059s |
| 处理请求数 | 10 |
| 集群配置 | 4 replicas, PD ratio=0.5 (2P+2D) |
| 输出文件 | request_metrics.csv, batch_metrics.csv, plots/ 等 |
| 注意事项 | PNG 生成跳过 (无 Chrome/Kaleido), CSV 数据正常保存 |

### 3.3 前置依赖

- `data/processed_traces/splitwise_conv.csv` — 从 microsoft/vidur 拷贝
- `data/profiling/` — 从 microsoft/vidur 拷贝 (Native Vidur 模式必需)

---

## 4. 集成测试 — DeepSeek-671B AICB 模式

### 4.1 运行命令

```bash
cd vidur-alibabacloud

python -m vidur.main \
  --replica_config_pd_p2p_comm_bandwidth 800 \
  --replica_config_nvlink_bandwidth 1600 \
  --replica_config_rdma_bandwidth 800 \
  --replica_config_pd_p2p_comm_dtype fp8 \
  --poisson_request_interval_generator_config_qps 100 \
  --synthetic_request_generator_config_num_requests 5 \
  --length_generator_config_type fixed \
  --fixed_request_length_generator_config_prefill_tokens 1024 \
  --fixed_request_length_generator_config_decode_tokens 10 \
  --cluster_config_num_replicas 4 \
  --replica_config_pd_node_ratio 0.5 \
  --global_scheduler_config_type split_wise \
  --replica_scheduler_config_type split_wise \
  --replica_config_model_name deepseek-671B \
  --replica_config_tensor_parallel_size 8 \
  --replica_config_num_pipeline_stages 1 \
  --random_forrest_execution_time_predictor_config_backend aicb \
  --replica_config_device h20
```

### 4.2 测试结果

| 项目 | 结果 |
|------|------|
| 状态 | **通过** |
| 模拟结束时间 | 0.038s |
| 处理请求数 | 5 |
| 集群配置 | 4 replicas, PD ratio=0.5 (2P+2D), H20 GPU |
| 已知警告 | "AICB data is empty, using default execution time" — 预期行为 |
| 注意事项 | EP 自动设为 world_size (=8)。DeepSeek-671B 在 H20 上需要 TP=8 + FP8 才能通过内存检查 |

### 4.3 GPU 内存分析

DeepSeek-671B 模型参数量极大, 在 H20 (141GB) 上需要较高的并行度:

| 配置 | Prefill 参数内存 | 可用内存 | 状态 |
|------|-----------------|---------|------|
| TP=2, EP=8, FP16 | 320.97 GB | 72.00 GB | OOM |
| TP=4, EP=8, FP16 | 162.86 GB | 126.90 GB | OOM |
| TP=8, EP=8, FP8 | ~81 GB | 126.90 GB | **通过** |

---

## 5. 集成测试 — run_scenarios.sh 四场景套件

### 5.1 场景概览

| 场景 | 模型 | PD 分离 | 集群配置 | 调度策略 |
|:---:|------|---------|---------|---------|
| 1 | Qwen3-Next-80B | 否 (ratio=1) | 32 replicas, tp=1, pp=1, EP=auto | lor + sarathi |
| 2 | Qwen3-Next-80B | 是 (P=2, D=6) | 8 replicas, tp=1, pp=1, EP=auto | split_wise |
| 3 | DeepSeek-671B | 是 (P=2, D=6) | 8 replicas, tp=8, pp=1, EP=auto | split_wise |
| 4 | Qwen3-MoE-235B | 是 (P=2, D=6) | 8 replicas, tp=4, pp=1, EP=auto | split_wise |

**公共参数**: H20 GPU, FP8, AICB 后端, Poisson QPS=100, 4 请求, prefill=100 tokens, decode=8 tokens

### 5.2 运行命令

```bash
# 通过脚本运行全部四场景 (需要 vidur conda 环境)
bash examples/vidur-ali-scenarios/run_scenarios.sh --all

# 或单独运行某个场景
bash examples/vidur-ali-scenarios/run_scenarios.sh --scenario 1
```

### 5.3 测试结果

| 场景 | 模型 | 模拟结束时间 | 状态 | 说明 |
|:---:|------|------------|------|------|
| 1 | Qwen3-Next-80B (无PD) | 0.016s | **通过** | AICB CSV 缺失使用默认执行时间 (预期行为) |
| 2 | Qwen3-Next-80B (PD) | 0.016s | **通过** | PD 分离 P=2 D=6 正常调度 |
| 3 | DeepSeek-671B (PD) | 0.017s | **通过** | MoE 671B 参数 + MLA KV cache |
| 4 | Qwen3-MoE-235B (PD) | 0.016s | **通过** | MoE 235B 参数 + MHA KV cache |

**结果: 4/4 场景全部通过**

### 5.4 已知提示

- AICB 后端会输出 `AICB command failed` / `无法找到任何AICB CSV文件` 错误 → 这是**预期行为**, 表示无实际 profiling 数据, 系统自动使用经验线性公式估算执行时间
- numpy `RuntimeWarning: invalid value encountered in subtract` → 统计指标计算中的边界情况, 不影响仿真结果
- `run_scenarios.sh` 脚本要求 `conda activate vidur` 环境; 也可直接用 `python -m vidur.main` 在 base 环境中运行

---

## 6. 测试结果汇总

| 测试类型 | 测试项 | 状态 | 说明 |
|---------|--------|------|------|
| Layer 1 | PD 分离单元测试 (10 cases) | **全部通过** | 无外部依赖 |
| Layer 2 | Llama-3-8B Native Vidur | **通过** | 需要 processed_traces + profiling 数据 |
| Layer 2 | DeepSeek-671B AICB (H20 FP8) | **通过** | 需要 H20 + TP=8 + FP8 配置 |
| Layer 2 | run_scenarios.sh 四场景套件 | **全部通过** | 4/4 场景通过 (AICB 后端 + H20 FP8) |
| Layer 3 | Llama-3-8B SimAI Simulation | **未测试** | 需要编译 SimAI ns3 |
| Layer 3 | Llama-3-8B SimAI Analytical | **未测试** | 需要编译 SimAI Analytical |

---

## 7. 已知限制与依赖

| 限制 | 说明 |
|------|------|
| profiling 数据 | Native Vidur 模式需要 `data/profiling/` (来自 microsoft/vidur) |
| AICB 数据 | DeepSeek-671B AICB 模式使用默认执行时间 (无实际 profiling 数据) |
| PNG 输出 | 需要 Chrome/Kaleido 才能生成 PNG 图表, 否则只输出 CSV |
| GPU 内存 | DeepSeek-671B 在 H20 上需要 TP>=8 + FP8 才能通过内存检查 |
| SimAI 构建 | SimAI Simulation/Analytical 模式需要额外编译步骤 |
| seq_len=1 | KV cache 计算中 seq_len 硬编码为 1 (已知限制, 见 TODO 注释) |

---

## 附录 A: DeepSeek-V3-671B Prefill AICB Profiling 失败分析

> 测试日期: 2026-04-15
> 测试目的: 定位并验证 AICB 对 DeepSeek-V3-671B prefill 做 GPU kernel profiling 时的失败根因

### A.1 问题描述

AICB 使用 `--aiob_enable` 对 DeepSeek-V3-671B prefill 做 GPU kernel profiling 时，
在 H20 (SM90) 上当 `tp>=4` 时运行崩溃，报错：

```
Assertion failed (/tmp/pip-req-build-gg88vm1n/csrc/sm90/prefill/sparse/fwd.cu:647): params.h_q % B_H == 0
```

DeepSeek-V3 有 128 个 attention heads (`num_attention_heads=128`)，
AICB 按 `h_q = num_attention_heads / tp` 计算每个 TP rank 的 head 数量。
当 tp>=4 时，h_q < 64，无法整除 FlashMLA 的 tile 常量 B_H=64。

### A.2 FlashMLA 官方依据

FlashMLA 的 SM90 prefill sparse kernel 在编译时定义 `B_H = 64`（对应 SM90 WGMMA 64×64 tile），
运行时要求 `params.h_q % B_H == 0`，否则触发 CUDA 断言失败。

- 源码位置: [`csrc/sm90/prefill/sparse/config.h`](https://github.com/deepseek-ai/FlashMLA/blob/main/csrc/sm90/prefill/sparse/config.h) — `B_H = 64` 编译时常量
- 断言位置: `csrc/sm90/prefill/sparse/fwd.cu:647` — `params.h_q % B_H == 0`
- FlashMLA 版本: `1.0.0+1408756` (pip install)

### A.3 完整调用栈

```
Vidur (vidur.main)
  └── RandomForrestExecutionTimePredictor
        └── ExecutionTimeSeries._generate_aicb_csv()
              └── subprocess: python -m workload_generator.Vidur_workload_generator
                    └── AiobDeepSeek (prefill phase)
                          └── flash_mla.flash_mla_sparse_fwd()
                                └── SM90 CUDA kernel (fwd.cu:647)
                                      └── ASSERTION FAILED: params.h_q % B_H == 0
```

### A.4 完整验证矩阵

**Prefill 测试 (DeepSeek-V3-671B, seq=1024)**

| tp | h_q=128/tp | bs=1 | bs=2 | bs=4 | bs=8 | 结论 |
|----|-----------|------|------|------|------|------|
| 1 | 128 | ✅ | ✅ | ✅ | ✅ | h_q=128, 128%64=0, 全部通过 |
| 2 | 64 | ✅ | ✅ | ✅ | ✅ | h_q=64, 64%64=0, 边界值通过 |
| 4 | 32 | ❌ | ❌ | ❌ | — | h_q=32, 32%64≠0, 全部失败 |
| 8 | 16 | ❌ | ❌ | ❌ | — | h_q=16, 16%64≠0, 全部失败 |

**关键结论**: 失败完全由 tp 决定（h_q 对齐），与 bs 无关。tp=1/2 下 bs=2/4/8 全部通过，
tp=4/8 下即使 bs=1 也会失败。之前 "bs>1 fails" 的描述不准确。

**Decode 测试 (不受影响验证)**

| 测试 | tp | bs | 预期 | 实际 | 说明 |
|------|-----|-----|------|------|------|
| tp=8 decode | 8 | 2 | PASS | **PASS** | decode 路径使用不同 kernel，不受 B_H 限制 |

**测试命令示例:**

```bash
# tp=1 prefill (PASS)
cd aicb && conda run -n vidur python -m workload_generator.Vidur_workload_generator \
  DeepSeek-671B ./scripts/inference_configs/deepseek_default.json \
  --seq_length 1024 --micro_batch 1 --world_size 1 \
  --tensor_model_parallel_size 1 --expert_model_parallel_size 1 \
  --aiob_enable --phase prefill

# tp=4 prefill (FAIL — h_q=32, 32%64≠0)
cd aicb && conda run -n vidur python -m workload_generator.Vidur_workload_generator \
  DeepSeek-671B ./scripts/inference_configs/deepseek_default.json \
  --seq_length 1024 --micro_batch 1 --world_size 4 \
  --tensor_model_parallel_size 4 --expert_model_parallel_size 4 \
  --aiob_enable --phase prefill

# tp=8 decode (PASS — decode uses different kernel)
cd aicb && conda run -n vidur python -m workload_generator.Vidur_workload_generator \
  DeepSeek-671B ./scripts/inference_configs/deepseek_default.json \
  --seq_length 1024 --micro_batch 2 --world_size 8 \
  --tensor_model_parallel_size 8 --expert_model_parallel_size 8 \
  --aiob_enable --phase decode
```

### A.5 Vidur 降级路径验证

使用 DeepSeek-671B tp=8 PD 分离配置运行完整 Vidur 模拟 (场景 3 类似配置):

| 项目 | 结果 |
|------|------|
| 退出码 | 0 (成功) |
| 请求数 | 3/3 完成 |
| PD 分离 | 正常 (prefill replica 0/1, decode replica 2/3) |
| 执行时间数据 | 完整 (request_e2e_time, execution_time 等均有值) |
| 结果文件 | request_metrics.csv, config.json, chrome_trace.json |

结论: Vidur 在 AICB prefill profiling 受限的情况下仍能正常完成模拟，降级路径可靠。

### A.6 适用范围

此问题**仅影响**以下特定组合:

- **GPU**: H20 (SM90 架构)
- **模型**: DeepSeek-V3-671B (num_attention_heads=128)
- **阶段**: prefill (使用 `flash_mla_sparse_fwd` kernel)
- **并行度**: tp >= 4 (h_q = 128/tp < 64)

**不受影响**:
- 其他 GPU 架构 (非 SM90)
- 其他模型 (Qwen3-MoE-235B, Qwen3-Next-80B 等)
- Decode 阶段 (使用不同的 attention kernel)
- tp <= 2 的配置 (h_q >= 64, 满足对齐要求)

### A.7 测试环境

| 项目 | 版本 |
|------|------|
| GPU | NVIDIA H20-3e × 8 (143771 MiB each) |
| GPU Driver | 570.133.20 |
| CUDA | 12.9 |
| Python | 3.10.19 (vidur conda env) |
| PyTorch | 2.8.0 |
| FlashMLA | 1.0.0+1408756 |
| FlashInfer | 0.2.5 |
| AICB | commit `23eec3c48ca2d2d93dd888a4c7b22ab4421e782f` |
| conda env | vidur |
