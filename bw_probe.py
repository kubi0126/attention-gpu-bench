"""实测本机可达显存带宽，给 bench_decode.py 的 GB/s 做分母。

不引用显卡标称带宽（4060 Laptop 的官方口径与笔记本功耗墙强相关，拿来做分母会失真），
这里直接用大张量读出实测值。张量必须远大于 32 MiB 的 L2，否则测的是 L2 带宽。

运行：E:/ai_agent/.venv/Scripts/python.exe bw_probe.py
"""

import json
import statistics
from pathlib import Path

import torch


def timed(fn, warmup=5, reps=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ms = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ms.append(s.elapsed_time(e))
    return statistics.median(ms), min(ms)


def main():
    Path("results").mkdir(exist_ok=True)
    dt = torch.bfloat16
    el = torch.tensor(0, dtype=dt).element_size()
    res = {"device": torch.cuda.get_device_name(0), "dtype": "bf16", "tests": {}}
    print(f"# {torch.cuda.get_device_name(0)} | L2 = "
          f"{torch.cuda.get_device_properties(0).L2_cache_size / 2**20:.0f} MiB\n")

    for mib in (256, 512, 1024, 2048):
        n = mib * 2**20 // el
        x = torch.randn(n, device="cuda", dtype=dt)
        y = torch.empty_like(x)

        # 只读：reduction，全部元素过一遍显存
        med, lo = timed(lambda: torch.sum(x, dtype=dt))
        read = n * el / (med / 1e3) / 1e9
        # 读+写：copy，字节数按 2 倍算
        med2, lo2 = timed(lambda: y.copy_(x))
        copy = 2 * n * el / (med2 / 1e3) / 1e9
        res["tests"][f"{mib}MiB"] = {
            "read_GBps": round(read, 1), "read_GBps_best": round(n * el / (lo / 1e3) / 1e9, 1),
            "copy_GBps": round(copy, 1), "copy_GBps_best": round(2 * n * el / (lo2 / 1e3) / 1e9, 1),
        }
        print(f"{mib:>5} MiB  只读 {read:>7.1f} GB/s (best {n*el/(lo/1e3)/1e9:>6.1f})   "
              f"拷贝 {copy:>7.1f} GB/s (best {2*n*el/(lo2/1e3)/1e9:>6.1f})")
        del x, y
        torch.cuda.empty_cache()

    best = max(v["read_GBps"] for v in res["tests"].values())
    res["peak_read_GBps"] = best
    print(f"\n[口径] decode 有效带宽以此为分母，取只读峰值 {best:.1f} GB/s")

    # bf16 GEMM 峰值：attention 的"有效 TFLOPS"要有分母，否则 29 TFLOPS 是好是坏无从判断。
    # 不引用官方标称（笔记本会随功耗墙大幅波动），直接用大矩阵乘实测。
    print()
    for n in (4096, 8192):
        a = torch.randn(n, n, device="cuda", dtype=dt)
        b = torch.randn(n, n, device="cuda", dtype=dt)
        med, lo = timed(lambda: a @ b, warmup=3, reps=8)
        tf = 2 * n**3 / (med / 1e3) / 1e12
        res["tests"][f"gemm_{n}"] = {"bf16_TFLOPS": round(tf, 1),
                                     "bf16_TFLOPS_best": round(2 * n**3 / (lo / 1e3) / 1e12, 1)}
        print(f"bf16 GEMM {n}x{n}: {tf:>6.1f} TFLOPS (best {2*n**3/(lo/1e3)/1e12:.1f})")
        del a, b
        torch.cuda.empty_cache()
    peak_tf = max(v["bf16_TFLOPS"] for k, v in res["tests"].items() if k.startswith("gemm"))
    res["peak_bf16_TFLOPS"] = round(peak_tf, 1)
    print(f"[口径] attention 有效算力以此为分母，取实测峰值 {peak_tf:.1f} TFLOPS")

    torch.cuda.synchronize()
    print("[提示] 跑之前先关掉占用 GPU 的程序（游戏/浏览器硬加速），否则数字会被压低。")
    with open("results/bw_probe.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print("[已写出] results/bw_probe.json")


if __name__ == "__main__":
    main()
