"""Prefill 阶段 Attention 性能实测：GQA head 数对时延/算力/显存的影响。

动机来自「智能计算系统」课程实验留下的一个问题：为什么 CPU / NumPy 的计时
不能代表 GPU 上的真实 Attention 性能。本脚本在同一组 shape 网格上用 fused
kernel 实测时延、有效算力与峰值显存。

计时口径：torch.cuda.Event + 预热 + 多次取中位数（不用 time.perf_counter，
         因为 CUDA kernel 是异步下发的）。
显存口径：每个配置前 reset 峰值统计 + empty_cache，报 max_memory_allocated。

运行：E:/ai_agent/.venv/Scripts/python.exe bench_prefill.py
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

BACKENDS = {
    "cudnn": SDPBackend.CUDNN_ATTENTION,
    "math": SDPBackend.MATH,
    "efficient": SDPBackend.EFFICIENT_ATTENTION,
}

DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16}


def causal_attn_flops(b, h_q, s, d):
    """QK^T 与 PV 两次 batched matmul，每次 2*B*H*S*S*D FLOPs，causal 只算下半三角。"""
    return 2 * b * h_q * s * s * d


def est_extra_mib(b, h_q, h_kv, s, d, elem, backend):
    """估算一次前向除输入外还要额外占多少显存。

    MATH 后端会把 [B,H_q,S,S] 的分数矩阵实体化（softmax 前后各一份），
    fused 后端在片上分块处理，额外占用接近输出张量本身。
    这个估算用来提前跳过必然超额的配置：Windows 下显存用超不会干净报错，
    而是静默分页到系统内存，测出来的时延会慢几个数量级、完全不可信。
    """
    out = b * h_q * s * d * elem / 2**20
    if backend == "math":
        # MATH 后端会 upcast 到 fp32 做 softmax，且 scores / exp / 归一化结果同时存在，
        # 系数 3×4 字节是对着 B=1,S=512 实测峰值 28 MiB 标定的（公式给 27 MiB）。
        return 2 * out + 3 * b * h_q * s * s * 4 / 2**20
    return 2 * out


def time_once(fn, warmup, reps):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    ms = [s0.elapsed_time(e0) for s0, e0 in zip(starts, ends)]
    return statistics.median(ms), min(ms), max(ms)


def run_config(b, s, h_q, h_kv, d, dtype, backend, warmup, reps):
    q = torch.randn(b, h_q, s, d, device="cuda", dtype=dtype)
    k = torch.randn(b, h_kv, s, d, device="cuda", dtype=dtype)
    v = torch.randn(b, h_kv, s, d, device="cuda", dtype=dtype)
    gqa = h_q != h_kv

    def fn():
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=gqa)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    try:
        with sdpa_kernel(backend):
            out = fn()
            torch.cuda.synchronize()
            median, lo, hi = time_once(fn, warmup, reps)
            peak = torch.cuda.max_memory_allocated() - base
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        msg = str(e).splitlines()[0][:60]
        return {"status": "FAIL", "err": msg}

    ref = None
    with sdpa_kernel(SDPBackend.MATH):
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=gqa)
    diff = float((out.float() - ref.float()).abs().max())

    fl = causal_attn_flops(b, h_q, s, d)
    return {
        "status": "ok",
        "ms": round(median, 3),
        "ms_min": round(lo, 3),
        "ms_max": round(hi, 3),
        "tflops": round(fl / (median / 1e3) / 1e12, 2),
        "peak_MiB": round(peak / 2**20, 1),
        "max_diff_vs_math": round(diff, 5),
        "kv_MiB": round(2 * b * h_kv * s * d * q.element_size() / 2**20, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-q", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--h-kv", type=int, nargs="+", default=[8, 4, 2, 1])
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPE))
    ap.add_argument("--backends", nargs="+", default=["cudnn", "math"], choices=list(BACKENDS))
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--budget-mib", type=float, default=5000,
                    help="预估额外显存超过此值的配置直接跳过，避免 Windows 静默分页污染计时")
    ap.add_argument("--out", default="results/prefill")
    args = ap.parse_args()

    dt = DTYPE[args.dtype]
    elem = torch.tensor(0, dtype=dt).element_size()
    rows = []
    print(f"# {torch.cuda.get_device_name(0)} | torch {torch.__version__} | "
          f"H_q={args.head_q} D={args.head_dim} dtype={args.dtype} "
          f"warmup={args.warmup} reps={args.reps}\n", flush=True)
    hdr = (f"{'B':>3} {'S':>5} {'H_kv':>4} {'type':>5} {'backend':>8} "
           f"{'ms':>9} {'ms[min-max]':>16} {'TFLOPS':>7} {'peak/MiB':>9} {'KV/MiB':>8} {'status'}")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    for b in args.batch_sizes:
        for s in args.seq_lens:
            for h_kv in args.h_kv:
                kind = "MHA" if h_kv == args.head_q else ("MQA" if h_kv == 1 else "GQA")
                for bname in args.backends:
                    est = est_extra_mib(b, args.head_q, h_kv, s, args.head_dim, elem, bname)
                    if est > args.budget_mib:
                        r = {"status": "SKIP", "err": f"预估额外 {est:.0f} MiB 超预算"}
                        rows.append({"B": b, "S": s, "H_kv": h_kv, "type": kind,
                                     "backend": bname, "est_extra_MiB": round(est, 1), **r})
                        print(f"{b:>3} {s:>5} {h_kv:>4} {kind:>5} {bname:>8} "
                              f"{'-':>9} {'-':>16} {'-':>7} {'-':>9} {'-':>8} "
                              f"SKIP: 预估额外 {est:.0f} MiB 超预算", flush=True)
                        continue
                    r = run_config(b, s, args.head_q, h_kv, args.head_dim, dt,
                                   BACKENDS[bname], args.warmup, args.reps)
                    row = {"B": b, "S": s, "H_kv": h_kv, "type": kind, "backend": bname,
                           "est_extra_MiB": round(est, 1), **r}
                    rows.append(row)
                    if r["status"] == "ok":
                        spread = f"{r['ms_min']}-{r['ms_max']}"
                        print(f"{b:>3} {s:>5} {h_kv:>4} {kind:>5} {bname:>8} "
                              f"{r['ms']:>9} {spread:>16} "
                              f"{r['tflops']:>7} {r['peak_MiB']:>9} {r['kv_MiB']:>8} ok",
                              flush=True)
                    else:
                        print(f"{b:>3} {s:>5} {h_kv:>4} {kind:>5} {bname:>8} "
                              f"{'-':>9} {'-':>16} {'-':>7} {'-':>9} {'-':>8} {r['status']}: {r['err']}",
                              flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), "utf-8")
    print(f"\n[已写出] {out.with_suffix('.json')}  共 {len(rows)} 行")


if __name__ == "__main__":
    main()
