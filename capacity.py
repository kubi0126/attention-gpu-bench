"""容量建模：8 GiB 显存下，不同 H_kv 到底能撑多大的 batch × 多长的上下文。

KV 元素数是个公式（2·B·S_kv·H_kv·D）。这里不停在公式上：
先用公式预测，再真去分配 cache 并跑一步 decode，二分出实际能跑通的上限，
看公式和现实的差距有多大（差距本身就是要报告的内容：allocator 预留、workspace、
输出张量都不在公式里）。

运行：E:/ai_agent/.venv/Scripts/python.exe capacity.py
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16}


def kv_bytes(b, h_kv, s, d, elem):
    return 2 * b * h_kv * s * d * elem


def fits(b, h_q, h_kv, s, d, dtype, backend, budget_mib):
    """真分配 + 真跑一步 decode，返回峰值占用 MiB；跑不通返回 None。"""
    elem = torch.tensor(0, dtype=dtype).element_size()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        k = torch.zeros(b, h_kv, s, d, device="cuda", dtype=dtype)
        v = torch.zeros(b, h_kv, s, d, device="cuda", dtype=dtype)
        q = torch.randn(b, h_q, 1, d, device="cuda", dtype=dtype)
        with sdpa_kernel(backend):
            F.scaled_dot_product_attention(q, k, v, enable_gqa=(h_q != h_kv))
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 2**20
        ok = peak <= budget_mib
        del k, v, q
        return (round(peak, 1), ok)
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        return None


def bisect(probe, lo, hi):
    """在 [lo, hi] 里找最后一个能跑通的值。probe 返回 True/False。"""
    if not probe(lo):
        return None
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if probe(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-q", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--h-kv", type=int, nargs="+", default=[8, 4, 2, 1])
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPE))
    ap.add_argument("--backend", default="cudnn", choices=["cudnn", "math", "efficient"])
    ap.add_argument("--budget-mib", type=float, default=6500,
                    help="留给 KV 的显存预算，默认 6.3 GiB（笔记本还要留桌面与 allocator 开销）")
    ap.add_argument("--fixed-s", type=int, default=4096)
    ap.add_argument("--fixed-b", type=int, default=8)
    ap.add_argument("--max-b", type=int, default=8192, help="二分上界，撞到上界说明结果被截断")
    ap.add_argument("--max-s", type=int, default=1048576)
    args = ap.parse_args()

    backend = {"cudnn": SDPBackend.CUDNN_ATTENTION, "math": SDPBackend.MATH,
               "efficient": SDPBackend.EFFICIENT_ATTENTION}[args.backend]
    dt = DTYPE[args.dtype]
    elem = torch.tensor(0, dtype=dt).element_size()
    total = torch.cuda.get_device_properties(0).total_memory / 2**20
    print(f"# {torch.cuda.get_device_name(0)}  总显存 {total:.0f} MiB | "
          f"KV 预算 {args.budget_mib:.0f} MiB | dtype={args.dtype} backend={args.backend}\n")

    rows = []
    hdr = (f"{'H_kv':>4} {'type':>5} | {'固定S':>6} {'最大B实测':>8} {'最大B预测':>8} "
           f"{'误差%':>6} | {'固定B':>5} {'最大S实测':>8} {'最大S预测':>8} {'误差%':>6}")
    print(hdr)
    print("-" * len(hdr))

    for h_kv in args.h_kv:
        kind = "MHA" if h_kv == args.head_q else ("MQA" if h_kv == 1 else "GQA")
        per_b = 2 * h_kv * args.fixed_s * args.head_dim * elem  # 每个样本的 KV 字节

        def probe_b(b, h_kv=h_kv, per_b=per_b):
            r = fits(b, args.head_q, h_kv, args.fixed_s, args.head_dim, dt, backend,
                     args.budget_mib)
            return r is not None and r[1]

        def probe_s(s, h_kv=h_kv):
            r = fits(args.fixed_b, args.head_q, h_kv, s, args.head_dim, dt, backend,
                     args.budget_mib)
            return r is not None and r[1]

        max_b = bisect(probe_b, 1, args.max_b)
        max_s = bisect(probe_s, 128, args.max_s)
        # 撞到二分上界 = 真实上限还在外面，这个数字不能用，标出来
        tb = "!" if max_b == args.max_b else ""
        ts = "!" if max_s == args.max_s else ""
        pred_b = int(args.budget_mib * 2**20 // per_b)
        pred_s = int(args.budget_mib * 2**20 // (2 * h_kv * args.fixed_b * args.head_dim * elem))
        eb = round((max_b - pred_b) / pred_b * 100, 1) if max_b else None
        es = round((max_s - pred_s) / pred_s * 100, 1) if max_s else None
        rows.append({"H_kv": h_kv, "type": kind, "max_B_real": max_b, "max_B_pred": pred_b,
                     "B_truncated": bool(tb), "max_S_real": max_s, "max_S_pred": pred_s,
                     "S_truncated": bool(ts)})
        print(f"{h_kv:>4} {kind:>5} | {args.fixed_s:>6} {f'{max_b}{tb}':>8} {pred_b:>8} "
              f"{str(eb):>6} | {args.fixed_b:>5} {f'{max_s}{ts}':>8} {pred_s:>8} {str(es):>6}")

    print("\n[!] = 撞到二分上界，真实上限更大，该行不可用，需提高 --max-b / --max-s 重跑")

    Path("results").mkdir(exist_ok=True)
    Path("results/capacity.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), "utf-8")
    print("\n[已写出] results/capacity.json")
    print("[口径提醒] 这里只放 KV Cache + 一步 decode，不含模型权重。")


if __name__ == "__main__":
    main()
