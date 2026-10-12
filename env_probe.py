"""探测本机 GPU 与 SDPA 能力，为 benchmark 的写法定下依据。

运行：E:/ai_agent/.venv/Scripts/python.exe env_probe.py
"""

import json

import torch
import torch.nn.functional as F

out = {}

out["torch"] = torch.__version__
out["cuda_builtin"] = torch.version.cuda
out["cudnn"] = torch.backends.cudnn.version()

p = torch.cuda.get_device_properties(0)
out["device"] = {
    "name": p.name,
    "cc": f"{p.major}.{p.minor}",
    "sm_count": p.multi_processor_count,
    "total_mem_GiB": round(p.total_memory / 2**30, 2),
    "L2_cache_MiB": round(p.L2_cache_size / 2**20, 1),
    "clock_MHz": p.clock_rate,
    "mem_clock_MHz": p.memory_clock_rate,
    "mem_bus_width": getattr(p, "memory_bus_width", None),
}

# enable_gqa 是 GQA 直连 SDPA 的关键：不支持就得手动 expand，会污染显存测量
# 它是 C++ 内置算子，inspect.signature 拿不到，只能直接试调用
out["has_enable_gqa"] = True
try:
    _q = torch.randn(1, 2, 8, 4, device="cuda", dtype=torch.bfloat16)
    _k = torch.randn(1, 1, 8, 4, device="cuda", dtype=torch.bfloat16)
    F.scaled_dot_product_attention(_q, _k, _k, enable_gqa=True)
except TypeError as e:
    out["has_enable_gqa"] = False
    out["enable_gqa_error"] = str(e)[:120]

# 各后端在当前 shape 下是否真的可用（enabled 只是全局开关，可用性看 shape/dtype）
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

B, Hq, Hkv, S, D = 2, 8, 2, 1024, 128
q = torch.randn(B, Hq, S, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.bfloat16)

backends = {
    "FLASH": SDPBackend.FLASH_ATTENTION,
    "EFFICIENT": SDPBackend.EFFICIENT_ATTENTION,
    "MATH": SDPBackend.MATH,
}
avail = {}
for name, be in backends.items():
    try:
        with sdpa_kernel(be):
            F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        torch.cuda.synchronize()
        avail[name] = "ok"
    except Exception as e:
        avail[name] = f"{type(e).__name__}: {str(e)[:90]}"
out["backend_with_gqa"] = avail

# 数值一致性：enable_gqa 直连 vs 手动 repeat 成 MHA 形状
ref = F.scaled_dot_product_attention(
    q, k.repeat_interleave(Hq // Hkv, dim=1), v.repeat_interleave(Hq // Hkv, dim=1),
    is_causal=True,
)
got = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
out["gqa_max_abs_diff_vs_repeat"] = float((ref - got).abs().max())

# 是否命中 cudnn / flash 路径（provenance 能告诉我们实际走了哪个后端）
with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
    F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
out["sdpa_kernel_priority_list_ok"] = True

print(json.dumps(out, indent=2, ensure_ascii=False))
