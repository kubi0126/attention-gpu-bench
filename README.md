# GPU 上的 GQA Attention 推理性能实测（Prefill / Decode 双阶段）

> 一个独立的最小基准：在同一组 shape 网格下，把 MHA / GQA / MQA 的 attention 分别放到
> **prefill** 与 **decode** 两个阶段，测时延、有效算力、峰值显存与有效带宽。
> 动机来自《智能计算系统》课程实验留下的一个问题——
> **"为什么 NumPy / CPU 计时不能直接代表实际 GPU Attention 的性能"**。
> 本仓库不引用任何没在本机跑过的数字；下文第二节的命令可复现全部结果。

## 一、CPU / NumPy 口径与本仓库的差异

| | CPU / NumPy 口径（对照，本仓库未复现） | 本仓库（PyTorch / GPU） |
| --- | --- | --- |
| 计算实现 | 手写 `stable_softmax` + `matmul`，显式 `np.repeat` 展开 KV | `F.scaled_dot_product_attention(enable_gqa=True)`，fused kernel |
| 计时方式 | `time.perf_counter()`（同步执行） | `torch.cuda.Event` + 预热；prefill 取中位数，decode 另加 CUDA Graph 口径 |
| 可测指标 | 墙钟时间、K/V 元素数 | 时延、有效 TFLOPS、峰值显存、KV 占用、有效带宽 |
| 结论范围 | 只能看"存储量按 1/H_kv 线性下降" | 能区分 prefill（算力瓶颈）与 decode（访存瓶颈），以及 kernel 实现的代价 |

## 二、实验环境（实测，非引用参数）

```text
GPU      NVIDIA GeForce RTX 4060 Laptop GPU   8.0 GiB
架构     compute capability 8.9 (Ada)         SM 数 24      L2 32 MiB
软件     Python 3.12.6 / torch 2.12.0+cu126 / cuDNN 9.10.02
系统     Windows 11
依赖     pip install -r requirements.txt（需 torch >= 2.5，Python 3.12）
```

建议先跑 `python env_probe.py`：它会打印本机 GPU 参数、`enable_gqa` 支持情况与
各 SDPA 后端在当前 shape 下是否可用——**换一台机器，这三项的结论就可能全变**，
本仓库所有数字都以上面这台机器为准。

硬件上限一律实测（`bw_probe.py`），不引用官方标称——笔记本的实际频率随功耗墙波动很大：

```text
只读带宽   209.1 GB/s（1 GiB bf16 张量归约，取各尺寸最好值）
拷贝带宽   187.3 GB/s（512 MiB copy，读+写按 2 倍字节计）
bf16 GEMM  27.5 TFLOPS（4096³，短占空比）
           22.5 TFLOPS（8192³，持续负载下降 ~18%）
```

## 三、SDPA 后端可用性探测（`env_probe.py`）

这是本机最值得记录的一条约束——**同一份代码，后端差别极大，且并非全部可用**：

| 后端 | MHA（H_q==H_kv） | GQA/MQA（H_q>H_kv，`enable_gqa=True`） |
| --- | --- | --- |
| `CUDNN_ATTENTION` | ✅ 可用 | ✅ 可用（原生支持分组 head） |
| `EFFICIENT_ATTENTION` | ✅ 可用 | ❌ 不可用：要求 Q/K/V head 数相同 |
| `FLASH_ATTENTION` | ❌ **不可用**：该 Windows 轮子 "Torch was not compiled with flash attention" | ❌ 同左 |
| `MATH` | ✅ 可用 | ✅ 可用（会实体化 `[B,H,S,S]` 分数矩阵） |

两条结论：

1. `enable_gqa=True` 在 PyTorch 里**不是所有后端都支持**。走 `EFFICIENT`/`FLASH` 时只能手动把 KV 复制成 MHA 形状，而这会把"省下的 KV"又以临时张量的形式吃回去——用 `repeat_interleave` 展开做 GQA 的代价，和 CPU 上 `np.repeat` 是同一类问题，在 GPU 上同样存在。
2. 本机拿不到 FlashAttention 后端，所以**本项目的 fused-kernel 基线是 cuDNN attention，不是 FA2**。任何"这就是 FlashAttention 性能"的说法在本机都不成立，报告里如实标注。

数值一致性：`cudnn + enable_gqa` 与 `math + enable_gqa` 在 `H_q=8, H_kv=2, S=1024, bf16` 下最大绝对差 `0.0078125`（bf16 量级正常）。

## 四、指标口径（避免数字被误读）

- **时延**：预热 5 次后正式计时。prefill 取**中位数**并给 `[min-max]` 区间；
  但 `B=8, S=4096` 一档 median 被系统噪声拉得不稳（见 5.3），该档统一改用 `min` 并标注。
  decode 单步则同时报 **CUDA Graph 重放**（纯 kernel）与 **eager 连续下发**（含 CPU 下发开销）两个口径。
- **有效 TFLOPS**：`2·B·H_q·S²·D`（QKᵀ 与 PV 两次 batched matmul，causal 已按半三角折算），
  再除以实测中位时延。注意分子用的是 **H_q 而非 H_kv**：GQA 的计算量不随 H_kv 减少，减少的只是访存量，
  所以这个指标能直接暴露"GQA 在 prefill 阶段本来就不该省算力"。
- **peak/MiB**：`max_memory_allocated` 减去输入张量分配后的基线，即**输出 + 中间激活/workspace**，不含 Q/K/V 输入。
- **KV/MiB**：`2·B·H_kv·S·D·sizeof(dtype)`，单独列出，便于和 peak 对照。
- 每个配置前 `empty_cache()` + `reset_peak_memory_stats()`，避免前一个配置的碎片影响后一个。

## 五、Prefill 结果

运行命令：

```bash
python bench_prefill.py
```

主表（`B=8`，bf16，`H_q=8`，`D=128`；完整 64 行见 `results/prefill.json`）：

| S | H_kv | 类型 | cuDNN ms | math ms | 倍数 | cuDNN 峰值 | math 峰值 | KV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 8 | MHA | 0.176 | 4.353 | 24.7× | 16.0 MiB | 217 MiB | 16.0 MiB |
| 1024 | 8 | MHA | 0.623 | 21.922 | 35.2× | 32.0 MiB | 724 MiB | 32.0 MiB |
| 1024 | 4 | GQA | 0.623 | 20.419 | 32.8× | 32.0 MiB | 756 MiB | 16.0 MiB |
| 1024 | 2 | GQA | 0.620 | 21.051 | 34.0× | 32.0 MiB | 740 MiB | 8.0 MiB |
| 1024 | 1 | MQA | 0.621 | 21.988 | 35.4× | 32.0 MiB | 732 MiB | 4.0 MiB |
| 2048 | 8 | MHA | 2.325 | 83.161 | 35.8× | 64.0 MiB | 2.61 GiB | 64.0 MiB |
| 2048 | 1 | MQA | 2.334 | 82.214 | 35.2× | 64.0 MiB | 2.62 GiB | 8.0 MiB |
| 4096 | 8 | MHA | 10.08 (min) | SKIP | — | 128.0 MiB | — | 128.0 MiB |
| 4096 | 1 | MQA | 10.50 (min) | SKIP | — | 128.0 MiB | — | 16.0 MiB |

### 结论 1：prefill 阶段 H_kv 几乎不影响时延

`B=8, S=2048`：MHA `2.325 ms` vs MQA `2.334 ms`，差 **0.4%**；`S=1024` 四档全在 `0.620–0.623 ms`。

原因是计算量 `2·B·H_q·S²·D` 只由 `H_q` 决定，GQA 省的是 **KV 访存量**不是计算量；
而此时 cuDNN 的有效算力已到 `29.4–29.6 TFLOPS`，与本机实测 bf16 GEMM 峰值 `27.5 TFLOPS` 同量级
（略超是因为短 kernel 的 boost 时钟更高，不能当精确 MFU 报），
说明 prefill 已经把 tensor core 打满，省下来的访存换不到时间。

**这条说明"减少 KV 存储量"不能直接外推成"prefill 会变快"**：省下来的是访存量，而 prefill 的时间由计算量决定。

### 结论 2：同一份语义，kernel 实现差 24–36 倍、显存差 14–40 倍

cuDNN fused 路径 vs 实体化 `[B,H,S,S]` 分数矩阵的 MATH 路径：
`S=2048, B=8` 时 `2.325 ms / 64 MiB` 对 `83.161 ms / 2.61 GiB`。
差距随 S 增大而拉大（`S=512,B=1` 只有 7.3×，`S≥1024` 稳定在 22–38×），
因为 MATH 的中间张量按 **S²** 增长。这就是 FlashAttention"不把 S×S 矩阵写回显存"的动机，现在有本机数字。

### 结论 3：KV 存储量确实按 `1/H_kv` 线性下降

表里 KV 列 `128 → 16 MiB`，严格服从 `2·B·H_kv·S·D·bytes`。**存储量的算术没错，错的是把 CPU 计时外推到 GPU**——这正是那道思考题要的答案。

### 测量噪声（如实记录，不藏）

- `B=8, S=4096` 档：`min` 稳定在 `10.0–10.5 ms`，但 `median` 在 `11–27 ms` 之间跳，
  两次复测的 median 相差 2.3 倍。判定为 Windows 下长 kernel 被系统噪声打断，
  因此该档统一用 `min` 口径，并在表里标注。
- `S=512, B=1` 档：kernel 时间 `< 0.1 ms`，已进入下发开销区间，不用于任何结论。
- 显存预估闸门：MATH 后端在 `B=8, S=4096` 需要 ~12 GiB 中间张量，
  Windows 下超出显存不会报错而是静默分页到内存，会把计时污染几个数量级，
  所以这类配置直接标 `SKIP` 而不是记录一个假数字。预估公式对着实测峰值标定
  （`B=1,S=512` 公式给 27 MiB、实测 28 MiB）。


## 六、Decode 结果

运行命令：

```bash
python bench_decode.py --show-shape-cost --peak-gbps 209.1
```

### 结论 4：decode 阶段 H_kv 的减少会**线性**换成时间收益

`B=8`，cuDNN，CUDA Graph 口径（纯 kernel，已剥掉 CPU 下发开销）：

| ctx | H_kv | 类型 | KV 总量 | 单步 ms | 读 KV 有效带宽 | 相对 MHA |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 8 | MHA | 128 MiB | 0.5857 | 229 GB/s | 1.00× |
| 4096 | 4 | GQA | 64 MiB | 0.2836 | 237 GB/s | **2.07×** |
| 2048 | 8 | MHA | 64 MiB | 0.2806 | 239 GB/s | — |
| 4096 | 2 | GQA | 32 MiB | 0.0652 | 515 GB/s | 数据被 L2 兜住，不采信 |
| 4096 | 1 | MQA | 16 MiB | 0.1044 | — | 同上 |

两组干净的对照都指向同一条规律：**KV 减半，单步时延减半，有效带宽恒定在 ~230 GB/s**。
这就是 memory-bound 的判据。而 `2·B·H_kv·S·D·bytes / 时延` 算出来的 229 / 237 / 239 GB/s
几乎不随配置变化，说明这些 kernel 已经贴着显存带宽上限跑。

与结论 1 合起来就是完整的一句话：**同一个 GQA，在 prefill 换不到时间（算力瓶颈），
在 decode 能线性换到时间（访存瓶颈）**——只测单一阶段、或者只在 CPU 上测，都看不出这个区分。

### 结论 5：小 batch 的 decode 根本不是 GPU 瓶颈，是 CPU 下发瓶颈

同一份单步 kernel，`eager 连续下发` 与 `CUDA Graph 重放` 的差：

| 配置 | eager | CUDA Graph | 下发开销占比 |
| --- | --- | --- | --- |
| `B=1, ctx=512, MHA` | 0.2246 ms | 0.0174 ms | 92% |
| `B=1, ctx=1024, GQA-4` | 0.1204 ms | 0.0143 ms | 88% |
| `B=8, ctx=4096, MHA` | 1.0237 ms | 0.5857 ms | 43% |
| `B=8, ctx=2048, MHA` | 0.2926 ms | 0.2806 ms | 4% |

batch 和序列越长，kernel 越重，下发开销占比从 92% 一路掉到 4%。
这解释了为什么推理引擎在小 batch 场景要上 CUDA Graph 和连续批处理。

### 结论 6：shape 变化对 fused kernel 是灾难级的

`--show-shape-cost` 单独量了这一项：让缓存长度每步 +1，

| 后端 | 长度递增 | 长度固定 | 比值 |
| --- | --- | --- | --- |
| cuDNN | 491.68 ms/步 | 0.057 ms/步 | **8656×** |
| MATH | 0.74 ms/步 | 0.375 ms/步 | 2× |

cuDNN attention 每见一个新 shape 就要重建执行计划，而 MATH 后端没有这个成本。
这条是本机踩出来的意外收获，也正好是推理引擎要做序列分桶 / padding 到固定块的原因。

### 结论 7：KV Cache 本身的收益

`B=8, ctx=4096, MHA`：用 Cache `0.5857 ms` vs 每步重算整段 `11.561 ms` = **19.7×**；
MATH 后端在 `B=1, ctx=4096` 更极端，`43.88 ms` 对 `0.4127 ms` = 106×。

### 口径与局限（不藏）

- 表里的 `tok/s` 是 **attention 单算子**的吞吐，不含采样、LM head 和其余层，不能当模型端到端吞吐。
- 有效带宽出现 110–115% 是正常的：分母来自归约探针（209 GB/s），与 attention 的读模式不同，
  应理解为"贴着带宽上限"，不要当精确比值引用。
- KV 总量 < 64 MiB（2× L2）的配置一律标 `L2驻留`：重复回放时数据不出显存，算出来的 GB/s 是 L2 带宽。
  第一版阈值取的是 1× L2，结果 32 MiB 档算出 515 GB/s（峰值的 246%）露了馅，已收紧。
- 每步单独用 event 计时会被下发开销污染（同配置 median 0.24 / min 0.024 ms），
  所以统一改成 CUDA Graph 与 eager 连续下发两种口径分别报。

## 七、容量建模：公式 vs 真去分配到 OOM

运行命令：

```bash
python capacity.py --max-s 4194304
```

预算 6500 MiB（8188 MiB 总显存里给桌面和 allocator 留余量），二分搜索真实能跑通的上限：

| H_kv | 类型 | `ctx=4096` 最大 batch（实测 / 公式） | `B=8` 最大 ctx（实测 / 公式） | 误差 |
| --- | --- | --- | --- | --- |
| 8 | MHA | 406 / 406 | 207,935 / 208,000 | 0.0% |
| 4 | GQA | 812 / 812 | 415,871 / 416,000 | 0.0% |
| 2 | GQA | 1,623 / 1,625 | 831,743 / 832,000 | −0.1% |
| 1 | MQA | 3,242 / 3,250 | 1,663,487 / 1,664,000 | −0.2% |

### 结论 8：纯 KV Cache 的容量公式极准，但它只是上界

`H_kv` 每减半，可容纳的 batch 或上下文就翻倍，实测与 `2·B·H_kv·S·D·bytes` 的误差 ≤0.2%。

但这条要连着它的**局限**一起说：探针只分配了 KV 和一步 decode，**没有扣模型权重、激活值
和 kernel workspace**，所以真实服务里的可用容量必然比这张表小。换句话说，
"公式在纯缓存场景准"本身就是结论，而不是"公式到处都准"。

同理，`B=8` 下 20 万–166 万的 ctx 上限只是**显存容量**意义上的数字：
prefill 的计算量按 S² 增长，这些配置在时间上完全不可用（`B=8, S=4096` 的 prefill 已经要 10 ms）。
容量、时延两张表必须一起看，单看任何一张都会得出荒谬结论。

## 八、待办

- [x] Prefill：时延 / 有效算力 / 峰值显存
- [x] 带宽与算力峰值探针（`bw_probe.py`）
- [x] Decode：固定长度单步 + KV Cache 收益 + shape 抖动代价
- [x] 容量建模：公式 vs 实测（`capacity.py`，误差 ≤0.2%，并标明只是上界）
- [ ] 可选：`torch.compile` 对下发开销的影响（结论 5 的直接缓解手段）
- [ ] 可选：WSL2 + Triton 手写 FA kernel，补上本机缺失的 FlashAttention 对照


