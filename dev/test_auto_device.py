#!/usr/bin/env python3
"""[FIX AUTO-DEVICE-TESTS] contract of gpu_backend._auto_device (no CUDA needed).

With PHEASY_GPU_DEVICE unset the backend no longer means cuda:0: it picks the
visible card with the most free VRAM once per process.  Tests that pinned a
literal "cuda:0" failed on a shared box whenever card 0 was the busy one; this
pins the selection itself on mocked cards instead.
"""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import gpu_backend as gb    # noqa: E402

try:
    import torch
except ImportError:
    torch = None

GIB = 2 ** 30


@unittest.skipIf(torch is None, "Torch not installed")
class AutoDeviceTests(unittest.TestCase):
    def cards(self, free_gib, env=None):
        """Mock visible cards with the given free VRAM; reset the per-process cache."""
        free = {d: None if f is None else int(f * GIB) for d, f in enumerate(free_gib)}
        stack = [patch.object(gb, "_auto_device_cache", None),
                 patch.object(torch.cuda, "is_available", return_value=True),
                 patch.object(torch.cuda, "device_count", return_value=len(free_gib)),
                 patch.object(gb, "_device_free_bytes", side_effect=lambda d: free.get(int(d))),
                 patch.dict(os.environ, env or {})]
        for p in stack:
            p.start()
            self.addCleanup(p.stop)
        for key in ("PHEASY_GPU_DEVICE", "PHEASY_GPU_AUTO_DEVICE", "PHEASY_GPU_DEVICES",
                    "PHEASY_GPU_NGPU", "PHEASY_GPU_SM_DEVICES"):
            if key not in (env or {}):
                os.environ.pop(key, None)
        return free

    def test_unset_device_picks_the_card_with_most_free_vram(self):
        self.cards([2.0, 20.0, 9.0])
        self.assertEqual(str(gb.device()), "cuda:1")
        self.assertEqual(gb.resident_device_ids(), [1])
        self.assertEqual([str(d) for d in gb._resident_cv_devices()], ["cuda:1"])

    def test_choice_is_cached_for_the_process(self):
        free = self.cards([2.0, 20.0, 9.0])
        self.assertEqual(str(gb.device()), "cuda:1")
        free[1] = 0                 # our own upload filled card 1
        self.assertEqual(str(gb.device()), "cuda:1")

    def test_explicit_device_and_opt_out_win(self):
        self.cards([2.0, 20.0, 9.0], {"PHEASY_GPU_DEVICE": "2"})
        self.assertEqual(str(gb.device()), "cuda:2")
        os.environ.pop("PHEASY_GPU_DEVICE")
        os.environ["PHEASY_GPU_AUTO_DEVICE"] = "0"
        self.assertEqual(str(gb.device()), "cuda:0")

    def test_unqueryable_card_is_skipped(self):
        self.cards([None, 4.0])     # card 0 full: mem_get_info cannot even make a context
        self.assertEqual(str(gb.device()), "cuda:1")

    def test_single_card_is_cuda0(self):
        self.cards([1.0])
        self.assertEqual(str(gb.device()), "cuda:0")


if __name__ == "__main__":
    unittest.main()
