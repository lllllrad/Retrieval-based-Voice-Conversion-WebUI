"""Streaming accompaniment / reverb removal for the live page.

Uses the PyMSS Roformer models from the WebUI's separation tab. Each live
block runs the models on a short rolling window of recent input (default
1 s) and keeps only the newest part; consecutive outputs are crossfaded.
The models never see the future, so a small lookahead (default 50 ms of
extra delay) is used to keep the newest audio away from the window edge.
"""

import os
import sys
import threading
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as AF

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = PROJECT_ROOT / "assets" / "pymss_weights"
MODEL_SR = 44100

# kind -> (model_type, checkpoint, config, target stem)
SPECS = {
    "vocals": ("bs_roformer", "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
               "model_bs_roformer_ep_368_sdr_12.9628.yaml", "vocals"),
    "dereverb": ("mel_band_roformer", "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt",
                 "dereverb_mel_band_roformer_anvuew.yaml", "noreverb"),
}
ORDER = ("vocals", "dereverb")  # remove accompaniment first, then reverb


def missing_weights():
    return [kind for kind, spec in SPECS.items() if not (WEIGHTS / spec[1]).is_file()]


class Separators:
    """Lazily loaded, shared Roformer models (one copy per server)."""

    def __init__(self, device):
        self.device = torch.device(device)
        self.models = {}
        self.lock = threading.Lock()

    def get(self, kind):
        with self.lock:
            if kind not in self.models:
                self.models[kind] = self._load(kind)
            return self.models[kind]

    def _load(self, kind):
        tools_dir = str(PROJECT_ROOT / "tools")
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        from pymss import MSSeparator

        model_type, ckpt, cfg, target = SPECS[kind]
        sep = MSSeparator(
            model_type=model_type, model_path=str(WEIGHTS / ckpt), config_path=str(WEIGHTS / cfg),
            device="cuda" if self.device.type == "cuda" else "cpu",
            device_ids=[self.device.index or 0], output_format="wav", use_tta=False,
            store_dirs={}, audio_params={}, logger=None, debug=False, progress_callback=None,
            inference_params={"batch_size": 1, "chunk_size": 352800, "overlap_size": 176400,
                              "standardize": False, "normalize": False,
                              "use_amp": self.device.type == "cuda", "model_dtype": "auto",
                              "cuda_attention_backend": "default"},
        )
        model = getattr(sep.model, "module", sep.model).eval()
        stems = list(sep.config.training.instruments)
        target_stem = getattr(sep.config.training, "target_instrument", None)
        # Single-target models return just that stem; others return all stems.
        stem_index = None if target_stem else stems.index(target)
        return model, stem_index

    @torch.no_grad()
    def run(self, kind, x):
        """x: (T,) float32 tensor at 44.1 kHz on device -> cleaned (T,)."""
        model, stem_index = self.get(kind)
        inp = x[None, None, :].expand(1, 2, -1).contiguous()
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
            y = model(inp)
        if stem_index is not None:
            y = y[:, stem_index]
        return y.float().reshape(-1, 2, y.shape[-1])[0].mean(0)


class StreamCleaner:
    """Per-connection rolling window; process() keeps the block length."""

    def __init__(self, separators, sr, block_frames, window=1.0, lookahead=0.05, fade=0.01):
        self.seps = separators
        self.sr = int(sr)
        self.bf = int(block_frames)
        self.dev = separators.device
        self.look = int(round(lookahead * self.sr))
        self.fade = max(1, int(round(fade * self.sr)))
        # Window must hold the block, the fade, the lookahead and some context.
        self.wn = max(int(round(window * self.sr)), self.bf + self.fade + self.look + self.sr // 4)
        self.ring = torch.zeros(self.wn, device=self.dev)
        self.tail = torch.zeros(self.fade, device=self.dev)
        self.fade_in = torch.linspace(0, 1, self.fade, device=self.dev)
        self.enabled = set()

    def set_enabled(self, kinds):
        kinds = {k for k in kinds if k in SPECS}
        if kinds and not self.enabled:
            self.tail.zero_()
        self.enabled = kinds

    @property
    def delay_ms(self):
        return round(1000 * (self.look + self.fade) / self.sr) if self.enabled else 0

    def process(self, block):
        n = len(block)
        x = torch.from_numpy(np.ascontiguousarray(block, dtype=np.float32)).to(self.dev)
        self.ring = torch.cat([self.ring[n:], x])
        if not self.enabled:
            return block
        y = AF.resample(self.ring, self.sr, MODEL_SR) if self.sr != MODEL_SR else self.ring
        for kind in ORDER:
            if kind in self.enabled:
                y = self.seps.run(kind, y)
        y = AF.resample(y, MODEL_SR, self.sr) if self.sr != MODEL_SR else y
        y = y[: self.wn]
        if y.shape[0] < self.wn:
            y = torch.nn.functional.pad(y, (self.wn - y.shape[0], 0))
        end = self.wn - self.look
        seg = y[end - n - self.fade: end]
        out = seg[:n].clone()
        out[: self.fade] = self.tail * (1 - self.fade_in) + seg[: self.fade] * self.fade_in
        self.tail = seg[n: n + self.fade].clone()
        return out.cpu().numpy()
