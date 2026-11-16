"""Print the GPU indices ordered by free VRAM (freest first).

The resident-memory budget is taken from device[0] only, and _device_own_bytes()
cannot see our own allocation on this host (NVML broken), so device[0] must be the
card with the most free memory at the moment the fit starts.

Only the devices SLURM actually gave us are probed (CUDA_VISIBLE_DEVICES), so the
returned indices are always valid for the allocation."""
import subprocess, sys

CODE = ("import torch,sys; i=int(sys.argv[1]); "
        "assert i < torch.cuda.device_count(), 'not visible'; "
        "f,t=torch.cuda.mem_get_info(i); print(int(f))")

probe = subprocess.run([sys.executable, "-c",
                        "import torch; print(torch.cuda.device_count())"],
                       capture_output=True, text=True, timeout=120)
try:
    n_vis = int((probe.stdout or "0").strip().splitlines()[-1])
except Exception:
    n_vis = 0
if n_vis <= 0:
    n_vis = 1

best = []
for g in range(n_vis):
    try:
        out = subprocess.run([sys.executable, "-c", CODE, str(g)],
                             capture_output=True, text=True, timeout=120)
        free = int(out.stdout.strip() or 0)
    except Exception:
        free = 0
    best.append((free, g))

best.sort(reverse=True)
sys.stderr.write("  visible=%d  free VRAM: %s\n" % (
    n_vis, "  ".join("gpu%d=%.1fGiB" % (g, f / 2**30) for f, g in best)))
print(",".join(str(g) for f, g in best))
