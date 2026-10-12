"""Decode 阶段实测：固定上下文长度下的单步时延，以及 KV Cache 的收益。

和 bench_prefill.py 的分工：prefill 一次算完整段，算力受限；
decode 一次只算一个 token，但要把历史 KV 全读一遍，访存受限。
只测单一阶段、或只在 CPU 上测，都只能看到 KV 存储量的比例关系，看不到这个差别。

【两个必须写清楚的方法论坑，都是本机实测踩出来的】
1. 长度递增会让 cuDNN 每步重建执行计划：第一版让缓存随生成步数变长，
   结果单步 470 ms 且与 B/H_kv/S 无关；固定长度后同 shape 单步 0.116 ms，差 4054 倍。
   所以这里按服务引擎的常规做法固定上下文长度，shape 抖动的代价用 --show-shape-cost 单列。
2. 每步单独用 event 计时会被 CPU 下发开销污染（同一配置 median 0.24 ms / min 0.024 ms）。
   因此每个配置测两遍：CUDA Graph 重放（纯 kernel）与 eager 连续下发（含下发开销），
   两者之差就是"这个规模下 decode 到底是不是 launch-bound"的答案。

带宽换算统一用 CUDA Graph 口径的最小值，分母取 bw_probe.py 的实测峰值。

运行：E:/ai_agent/.venv/Scripts/python.exe bench_decode.py
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16}
BE = {"cudnn": SDPBackend.CUDNN_ATTENTION, "math": SDPBackend.MATH,
      "efficient": SDPBackend.EFFICIENT_ATTENTION}


def time_steps(step, backend, warmup, reps, inner, use_graph=True):
    """返回 (graph_ms, eager_ms)；graph_ms 为 None 表示该配置无法建图。"""
    with sdpa_kernel(backend):
        for _ in range(warmup):
            step()
        torch.cuda.synchronize()

        graph_ms = None
        if use_graph:
            g = None
            try:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    step()
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    step()
                ms = []
                for _ in range(reps):
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    g.replay()
                    e1.record()
                    torch.cuda.synchronize()
                    ms.append(e0.elapsed_time(e1))
                graph_ms = min(ms)
                del g
            except Exception:
                graph_ms = None
                del g

        ms = []
        for _ in range(reps):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(inner):
                step()
            e1.record()
            torch.cuda.synchronize()
            ms.append(e0.elapsed_time(e1) / inner)
    return graph_ms, statistics.median(ms)


def make_decode_step(b, h_q, h_kv, L, d, dtype):
    """固定缓存长度 L，返回"生成下一个 token"这一步的闭包与需要保活的张量。"""
    s = L + 1
    k_c = torch.randn(b, h_kv, s, d, device="cuda", dtype=dtype)
    v_c = torch.randn(b, h_kv, s, d, device="cuda", dtype=dtype)
    q = torch.randn(b, h_q, 1, d, device="cuda", dtype=dtype)
    k_new = torch.randn(b, h_kv, 1, d, device="cuda", dtype=dtype)
    v_new = torch.randn(b, h_kv, 1, d, device="cuda", dtype=dtype)
    gqa = h_q != h_kv

    def step():
        k_c[:, :, L:L + 1] = k_new
        v_c[:, :, L:L + 1] = v_new
        return F.scaled_dot_product_attention(q, k_c, v_c, enable_gqa=gqa)

    return step, (k_c, v_c, q, k_new, v_new)


def make_nocache_step(b, h_q, h_kv, L, d, dtype):
    """不用 Cache：每生成 1 个 token 就把整段 [0, L] 重算一遍。"""
    s = L + 1
    q = torch.randn(b, h_q, s, d, device="cuda", dtype=dtype)
    k = torch.randn(b, h_kv, s, d, device="cuda", dtype=dtype)
    v = torch.randn(b, h_kv, s, d, device="cuda", dtype=dtype)
    gqa = h_q != h_kv

    def step():
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=gqa)

    return step, (q, k, v)


def shape_cost(b, h_q, h_kv, d, dtype, backend, steps=8):
    """同一份 decode，长度递增 vs 长度固定，单步时延差多少。"""
    gqa = h_q != h_kv
    q1 = torch.randn(b, h_q, 1, d, device="cuda", dtype=dtype)
    grow = []
    with sdpa_kernel(backend):
        for L in range(512, 512 + steps * 8, 8):
            k = torch.randn(b, h_kv, L, d, device="cuda", dtype=dtype)
            v = torch.randn(b, h_kv, L, d, device="cuda", dtype=dtype)
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            F.scaled_dot_product_attention(q1, k, v, enable_gqa=gqa)
            e1.record()
            torch.cuda.synchronize()
            grow.append(e0.elapsed_time(e1))
            del k, v
        fixed = []
        k = torch.randn(b, h_kv, 512, d, device="cuda", dtype=dtype)
        v = torch.randn(b, h_kv, 512, d, device="cuda", dtype=dtype)
        for _ in range(steps):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            F.scaled_dot_product_attention(q1, k, v, enable_gqa=gqa)
            e1.record()
            torch.cuda.synchronize()
            fixed.append(e0.elapsed_time(e1))
    return grow, fixed


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-q", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--ctx-lens", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--h-kv", type=int, nargs="+", default=[8, 4, 2, 1])
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPE))
    ap.add_argument("--backends", nargs="+", default=["cudnn", "math"], choices=list(BE))
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--inner", type=int, default=20, help="eager 口径下连续下发多少步再取平均")
    ap.add_argument("--peak-gbps", type=float, default=None,
                    help="bw_probe.py 实测只读峰值带宽，用于算带宽利用率")
    ap.add_argument("--show-shape-cost", action="store_true")
    ap.add_argument("--out", default="results/decode")
    args = ap.parse_args()

    dt = DTYPE[args.dtype]
    elem = torch.tensor(0, dtype=dt).element_size()
    l2_mib = torch.cuda.get_device_properties(0).L2_cache_size / 2**20
    rows = []
    print(f"# {torch.cuda.get_device_name(0)} | torch {torch.__version__} | H_q={args.head_q} "
          f"D={args.head_dim} dtype={args.dtype} warmup={args.warmup} reps={args.reps} "
          f"inner={args.inner} | L2={l2_mib:.0f} MiB\n", flush=True)
    print(f"[口径] KV 总量小于 L2({l2_mib:.0f} MiB) 的配置标为 L2驻留：重复回放时数据根本不出显存，"
          f"算出来的 GB/s 是 L2 带宽，不能当显存带宽用。", flush=True)

    if args.show_shape_cost:
        for bname in args.backends:
            grow, fixed = shape_cost(1, args.head_q, 2, args.head_dim, dt, BE[bname])
            gm, fm = statistics.median(grow), statistics.median(fixed)
            print(f"[shape 代价] backend={bname}  长度递增 {gm:.2f} ms/步  |  "
                  f"长度固定 {fm:.3f} ms/步  |  比值 {gm/fm:.0f}x", flush=True)
            rows.append({"kind": "shape_cost", "backend": bname,
                         "growing_ms": round(gm, 3), "fixed_ms": round(fm, 3)})
        print(flush=True)

    hdr = (f"{'B':>3} {'ctx':>5} {'H_kv':>4} {'type':>5} {'backend':>8} {'graph/ms':>9} "
           f"{'eager/ms':>9} {'下发占比':>8} {'tok/s':>8} {'KV/MiB':>8} {'读KV GB/s':>10} "
           f"{'带宽利用':>8} {'nocache/ms':>10} {'Cache收益':>9}")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    for b in args.batch_sizes:
        for L in args.ctx_lens:
            for h_kv in args.h_kv:
                kind = "MHA" if h_kv == args.head_q else ("MQA" if h_kv == 1 else "GQA")
                for bname in args.backends:
                    try:
                        step, keep1 = make_decode_step(b, args.head_q, h_kv, L,
                                                       args.head_dim, dt)
                        g_ms, e_ms = time_steps(step, BE[bname], args.warmup, args.reps,
                                                args.inner)
                        nstep, keep2 = make_nocache_step(b, args.head_q, h_kv, L,
                                                         args.head_dim, dt)
                        # MATH 后端的 nocache 要实体化 [B,H,S,S]，大 ctx 下会超显存被静默分页
                        est_nc = 3 * b * args.head_q * (L + 1) ** 2 * 4 / 2**20
                        if bname == "math" and est_nc > 5000:
                            ng = None
                        else:
                            ng, _ = time_steps(nstep, BE[bname], args.warmup,
                                               max(3, args.reps // 3), args.inner)
                    except (RuntimeError, torch.cuda.OutOfMemoryError) as ex:
                        print(f"{b:>3} {L:>5} {h_kv:>4} {kind:>5} {bname:>8}  FAIL "
                              f"{str(ex).splitlines()[0][:60]}", flush=True)
                        rows.append({"B": b, "ctx": L, "H_kv": h_kv, "backend": bname,
                                     "status": "fail", "err": str(ex).splitlines()[0][:80]})
                        continue

                    base = g_ms if g_ms else e_ms
                    kv_mib = 2 * b * h_kv * (L + 1) * args.head_dim * elem / 2**20
                    gbps = (2 * b * h_kv * (L + 1) * args.head_dim * elem) / (base / 1e3) / 1e9
                    resident = kv_mib < 2 * l2_mib   # 留一倍余量：还有 Q/输出/权重同享 L2
                    util = "-" if not args.peak_gbps else ("L2驻留" if resident
                                                           else f"{gbps/args.peak_gbps*100:.0f}%")
                    launch = (e_ms - g_ms) / e_ms * 100 if g_ms else None
                    rows.append({"B": b, "ctx": L, "H_kv": h_kv, "type": kind,
                                 "backend": bname, "graph_ms": round(g_ms, 4) if g_ms else None,
                                 "eager_ms": round(e_ms, 4), "launch_overhead_pct":
                                     round(launch, 1) if launch is not None else None,
                                 "tpot_ms_used": round(base, 4),
                                 "tokens_per_s": round(1000 / base, 1),
                                 "kv_MiB": round(kv_mib, 1), "kv_read_GBps": round(gbps, 1),
                                 "l2_resident": resident,
                                 "bw_util": util,
                                 "nocache_ms": round(ng, 4) if ng else None,
                                 "cache_speedup": round(ng / base, 1) if ng else None})
                    nc_s = f"{ng:.4f}" if ng else "SKIP"
                    sp_s = f"{ng/base:.1f}x" if ng else "-"
                    print(f"{b:>3} {L:>5} {h_kv:>4} {kind:>5} {bname:>8} "
                          f"{(f'{g_ms:.4f}' if g_ms else 'no-graph'):>9} {e_ms:>9.4f} "
                          f"{(f'{launch:.0f}%' if launch is not None else '-'):>8} "
                          f"{1000/base:>8.1f} {kv_mib:>8.1f} {gbps:>10.1f} {util:>8} "
                          f"{nc_s:>10} {sp_s:>9}", flush=True)
                    del keep1, keep2, step, nstep
                    torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), "utf-8")
    print(f"\n[已写出] {out.with_suffix('.json')}  共 {len(rows)} 行", flush=True)


if __name__ == "__main__":
    main()
