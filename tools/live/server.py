"""Live voice conversion over WebSocket.

The browser streams mic audio (16-bit PCM) to /ws, the server converts it
block by block with RVC (same algorithm as realtime_gui.py: rolling context,
SOLA crossfade) and streams the result back. The voice is a weighted blend of
one or more models, mixed in GPU memory, so ratio changes apply to the next
block.

Run: python -m tools.live.server --port 7867
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
for _k, _v in {
    "weight_root": "assets/weights",
    "index_root": "logs",
    "outside_index_root": "assets/indices",
    "rmvpe_root": "assets/rmvpe",
    "RVC_CUDA_GRAPH": "0",
}.items():
    os.environ.setdefault(_k, _v)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchaudio.transforms as tat  # noqa: E402
from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse  # noqa: E402

from tools.live.stt import MODELS as STT_MODELS, LiveSTT, api_key as stt_api_key  # noqa: E402

logger = logging.getLogger("rvc-live")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PAGE = Path(__file__).with_name("index.html")
MAX_MODELS = 8
INDEX_VECTORS = 30000


def _load_config():
    saved = sys.argv
    sys.argv = saved[:1]
    try:
        from configs.config import Config

        return Config()
    finally:
        sys.argv = saved


class Engine:
    """Shared models: HuBERT, pitch extractor and the blended synthesizer."""

    def __init__(self, config):
        from infer.hubert import load_hubert_model
        from infer.rtrvc import RVC

        self.config = config
        self.device = config.device
        self.dtype = torch.float16 if config.is_half else torch.float32
        self.lock = threading.Lock()
        self.raw = {}       # model file -> raw checkpoint weights on device
        self.meta = {}      # model file -> (version, f0, config)
        self.vectors = {}   # model file -> np.ndarray | None
        self.struct = None  # structure key of current net_g
        self.mix = {"models": [], "weights": [], "index_models": []}

        rvc = RVC.__new__(RVC)  # reuse RVC.infer / get_f0* without its file loader
        rvc.config = config
        rvc.device = config.device
        rvc.is_half = config.is_half
        rvc.f0_up_key = 0
        rvc.formant_shift = 0.0
        rvc.f0_min, rvc.f0_max = 50, 1100
        rvc.f0_mel_min = 1127 * np.log(1 + rvc.f0_min / 700)
        rvc.f0_mel_max = 1127 * np.log(1 + rvc.f0_max / 700)
        rvc.index_rate = 0.0
        rvc.infer_count = 0
        rvc.resample_kernel = {}
        rvc.model = load_hubert_model(self.device, self.is_half)
        rvc.net_g = None
        rvc.tgt_sr = None
        rvc.if_f0 = 1
        rvc.version = "v2"
        rvc.cache_pitch = torch.zeros(1024, device=self.device, dtype=torch.long)
        rvc.cache_pitchf = torch.zeros(1024, device=self.device, dtype=torch.float32)
        self.rvc = rvc

    @property
    def is_half(self):
        return self.config.is_half

    def separators(self):
        """Shared accompaniment/reverb removal models (loaded on first use)."""
        if getattr(self, "_separators", None) is None:
            from tools.live.preproc import Separators

            self._separators = Separators(self.device)
        return self._separators

    # ---- model data -------------------------------------------------
    def _load(self, name):
        if name in self.raw:
            return
        path = Path(os.environ["weight_root"]) / name
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        self.raw[name] = {k: v.to(self.device, self.dtype) for k, v in ckpt["weight"].items()}
        self.meta[name] = (ckpt.get("version", "v1"), ckpt.get("f0", 1), list(ckpt["config"]))

    def _index_vectors(self, name):
        if name not in self.vectors:
            from infer.vc.utils import get_index_path_from_model
            from tools.blend_tab import read_index_vectors

            path = get_index_path_from_model(name, 0)
            try:
                self.vectors[name] = read_index_vectors(path) if path else None
            except Exception:
                logger.warning("index read failed for %s\n%s", name, traceback.format_exc())
                self.vectors[name] = None
        return self.vectors[name]

    def forget(self, names):
        for n in list(self.raw):
            if n not in names:
                self.raw.pop(n, None)
                self.meta.pop(n, None)

    def _build_net(self, name):
        from infer.module.models import (
            SynthesizerTrnMs256NSFsid,
            SynthesizerTrnMs256NSFsid_nono,
            SynthesizerTrnMs768NSFsid,
            SynthesizerTrnMs768NSFsid_nono,
        )

        version, f0, config = self.meta[name]
        config = list(config)
        config[-3] = int(self.raw[name]["emb_g.weight"].shape[0])
        cls = {
            ("v1", 1): SynthesizerTrnMs256NSFsid,
            ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
            ("v2", 1): SynthesizerTrnMs768NSFsid,
            ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
        }[(version, f0)]
        self.rvc.net_g = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Weight norm is kept (not removed) so the blended raw weights load
        # exactly as a saved blend would.
        net_g = cls(*config, is_half=self.is_half)
        del net_g.enc_q
        net_g.eval().to(self.device)
        self.rvc.net_g = net_g.half() if self.is_half else net_g.float()
        self.rvc.tgt_sr = config[-1]
        self.rvc.if_f0 = f0
        self.rvc.version = version

    # ---- mixing -----------------------------------------------------
    def set_mix(self, models, weights, pitch, index_rate):
        pairs = [(m, float(w)) for m, w in zip(models, weights) if float(w) > 0][:MAX_MODELS]
        if not pairs:
            raise ValueError("no model with ratio > 0")
        names = [m for m, _ in pairs]
        for m in names:
            self._load(m)
        keys = {json.dumps([self.meta[m][0], self.meta[m][1], self.meta[m][2]]) for m in names}
        if len(keys) > 1:
            raise ValueError("models are not compatible (version / sample rate differ)")
        key = keys.pop()
        total = sum(w for _, w in pairs)
        ws = [w / total for _, w in pairs]

        tensors = [self.raw[m] for m in names]
        mixed = {}
        for k in tensors[0]:
            if k.startswith("enc_q"):
                continue
            if k == "emb_g.weight":
                n = min(t[k].shape[0] for t in tensors)
                mixed[k] = sum(w * t[k][:n] for w, t in zip(ws, tensors))
            else:
                mixed[k] = sum(w * t[k] for w, t in zip(ws, tensors))

        index = big = None
        used = []
        if index_rate > 0:
            import faiss

            avail = [(m, w) for m, w in zip(names, ws) if self._index_vectors(m) is not None]
            aw = sum(w for _, w in avail)
            rng = np.random.default_rng(0)
            parts = []
            for m, w in avail:
                v = self.vectors[m]
                n = max(1, int(round(INDEX_VECTORS * w / aw)))
                parts.append(v[rng.choice(len(v), n, replace=n > len(v))])
                used.append(m)
            if parts:
                big = np.concatenate(parts).astype(np.float32)
                index = faiss.IndexFlatL2(big.shape[1])
                index.add(big)

        with self.lock:
            if key != self.struct:
                self._build_net(names[0])
                self.struct = key
            self.rvc.net_g.load_state_dict(mixed, strict=False)
            self.rvc.f0_up_key = int(pitch)
            if index is not None:
                self.rvc.index, self.rvc.big_npy = index, big
                self.rvc.index_rate = float(index_rate)
            else:
                if hasattr(self.rvc, "index"):
                    del self.rvc.index
                self.rvc.index_rate = 0.0
            self.mix = {"models": names, "weights": ws, "index_models": used}
        self.forget(set(names))
        return {
            "tgt_sr": self.rvc.tgt_sr,
            "version": self.rvc.version,
            "models": names,
            "weights": ws,
            "index_models": used,
        }

    def set_pitch(self, pitch):
        with self.lock:
            self.rvc.f0_up_key = int(pitch)


class Session:
    """Per-connection buffers; mirrors realtime_gui.GUI.start_vc/audio_callback."""

    def __init__(self, engine, sr, block_time=0.25, extra_time=2.0, crossfade_time=0.05):
        self.engine = engine
        self.dev = engine.device
        self.sr = int(sr)
        self.zc = self.sr // 100
        # User settings: change them only through apply() (see there).
        self.threshold = -60.0
        self.f0method = "rmvpe"
        self.rms_mix = 0.0
        self.bypass = False
        self.cleaner = None       # StreamCleaner (accompaniment/reverb removal), created on demand
        self.last_input = None    # block after cleaning; what the transcript compares against
        zc = self.zc
        self.block_frame = int(round(block_time * self.sr / zc)) * zc
        self.block_frame_16k = 160 * self.block_frame // zc
        self.crossfade_frame = int(round(crossfade_time * self.sr / zc)) * zc
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * zc)
        self.sola_search_frame = zc
        self.extra_frame = int(round(extra_time * self.sr / zc)) * zc
        n = self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame
        self.input_wav = torch.zeros(n, device=self.dev, dtype=torch.float32)
        self.input_wav_res = torch.zeros(160 * n // zc, device=self.dev, dtype=torch.float32)
        self.rms_buffer = np.zeros(4 * zc, dtype="float32")
        self.sola_buffer = torch.zeros(self.sola_buffer_frame, device=self.dev, dtype=torch.float32)
        self.sola_den_kernel = torch.ones(1, 1, self.sola_buffer_frame, device=self.dev)
        self.skip_head = self.extra_frame // zc
        self.return_length = (self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // zc
        self.fade_in = (
            torch.sin(0.5 * np.pi * torch.linspace(0.0, 1.0, self.sola_buffer_frame, device=self.dev)) ** 2
        )
        self.fade_out = 1 - self.fade_in
        self.resampler = tat.Resample(self.sr, 16000, dtype=torch.float32).to(self.dev)
        self.resampler2 = None
        self.resampler2_sr = None
        self.cache_pitch = torch.zeros(1024, device=self.dev, dtype=torch.long)
        self.cache_pitchf = torch.zeros(1024, device=self.dev, dtype=torch.float32)

    def _out_resampler(self, tgt_sr):
        if tgt_sr == self.sr:
            return None
        if self.resampler2_sr != tgt_sr:
            self.resampler2 = tat.Resample(tgt_sr, self.sr, dtype=torch.float32).to(self.dev)
            self.resampler2_sr = tgt_sr
        return self.resampler2

    # ---- settings ---------------------------------------------------
    # Every user setting of a session goes through apply() and is reported by
    # state(). The connection keeps the settings it was sent and re-applies all
    # of them to each new Session, and the client compares state() (sent with
    # every block's stats) to its own controls and re-sends what differs. So a
    # setting cannot silently drop when a session is rebuilt or a message is
    # lost; a new setting only needs a branch here and an entry in state().
    def apply(self, settings):
        """Apply any subset of the settings. "clean" goes last because it can
        fail (missing weights) or take a while (model load)."""
        if "threshold" in settings:
            self.threshold = float(settings["threshold"])
        if "rmsMix" in settings:
            self.rms_mix = float(settings["rmsMix"])
        if "bypass" in settings:
            self.bypass = bool(settings["bypass"])
        if "f0method" in settings:
            self.f0method = str(settings["f0method"])
        if "clean" in settings:
            self.set_clean(list(settings["clean"]))

    def state(self):
        return {
            "threshold": self.threshold,
            "rmsMix": self.rms_mix,
            "bypass": self.bypass,
            "f0method": self.f0method,
            "clean": self.clean_kinds(),
            "clean_delay_ms": self.cleaner.delay_ms if self.cleaner else 0,
        }

    def clean_kinds(self):
        from tools.live.preproc import ORDER

        enabled = self.cleaner.enabled if self.cleaner else ()
        return [k for k in ORDER if k in enabled]

    def set_clean(self, kinds):
        """Enable/disable input cleaning ("vocals" = remove accompaniment,
        "dereverb" = remove reverb).
        Loads the models here (slow the first time), not in process()."""
        from tools.live.preproc import SPECS, StreamCleaner, missing_weights

        kinds = [k for k in kinds if k in SPECS]
        missing = [k for k in kinds if k in missing_weights()]
        if missing:
            raise ValueError("separation weights missing in assets/pymss_weights: %s" % ", ".join(missing))
        if kinds and self.cleaner is None:
            # A longer window separates better but costs more per block.
            window = 2.0 if self.block_frame / self.sr >= 0.3 else 1.0
            self.cleaner = StreamCleaner(self.engine.separators(), self.sr, self.block_frame, window=window)
        for k in kinds:
            self.engine.separators().get(k)
        if self.cleaner is not None:
            self.cleaner.set_enabled(set(kinds))

    def process(self, indata):
        import librosa

        zc = self.zc
        indata = indata.astype(np.float32)
        t_clean = 0.0
        if self.cleaner is not None and self.cleaner.enabled:
            t0 = time.perf_counter()
            indata = self.cleaner.process(indata)
            t_clean = time.perf_counter() - t0
        self.last_input = indata
        if self.threshold > -60:
            x = np.append(self.rms_buffer, indata)
            rms = librosa.feature.rms(y=x, frame_length=4 * zc, hop_length=zc)[:, 2:]
            self.rms_buffer[:] = x[-4 * zc:]
            x = x[2 * zc - zc // 2:]
            quiet = librosa.amplitude_to_db(rms, ref=1.0)[0] < self.threshold
            for i in range(quiet.shape[0]):
                if quiet[i]:
                    x[i * zc:(i + 1) * zc] = 0
            indata = x[zc // 2:]
        clean_stats = {"clean_ms": round(t_clean * 1000, 1)} if t_clean else {}
        if self.bypass:
            return indata[: self.block_frame].copy(), clean_stats
        bf = self.block_frame
        self.input_wav[:-bf] = self.input_wav[bf:].clone()
        self.input_wav[-indata.shape[0]:] = torch.from_numpy(indata).to(self.dev)
        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[self.block_frame_16k:].clone()
        self.input_wav_res[-160 * (indata.shape[0] // zc + 1):] = self.resampler(
            self.input_wav[-indata.shape[0] - 2 * zc:]
        )[160:]

        eng = self.engine
        t0 = time.perf_counter()
        with eng.lock:
            rvc = eng.rvc
            if rvc.net_g is None:
                return np.zeros(bf, dtype=np.float32), {}
            rvc.cache_pitch, rvc.cache_pitchf = self.cache_pitch, self.cache_pitchf
            infer_wav = rvc.infer(
                self.input_wav_res, self.block_frame_16k, self.skip_head,
                self.return_length, self.f0method,
            )
            tgt_sr = rvc.tgt_sr
        t_infer = time.perf_counter() - t0
        rs = self._out_resampler(tgt_sr)
        if rs is not None:
            infer_wav = rs(infer_wav)
        infer_wav = infer_wav.float()

        if self.rms_mix < 1:
            input_wav = self.input_wav[self.extra_frame:]
            rms1 = librosa.feature.rms(
                y=input_wav[: infer_wav.shape[0]].cpu().numpy(), frame_length=4 * zc, hop_length=zc
            )
            rms1 = F.interpolate(
                torch.from_numpy(rms1).to(self.dev).unsqueeze(0),
                size=infer_wav.shape[0] + 1, mode="linear", align_corners=True,
            )[0, 0, :-1]
            rms2 = librosa.feature.rms(y=infer_wav.cpu().numpy(), frame_length=4 * zc, hop_length=zc)
            rms2 = F.interpolate(
                torch.from_numpy(rms2).to(self.dev).unsqueeze(0),
                size=infer_wav.shape[0] + 1, mode="linear", align_corners=True,
            )[0, 0, :-1]
            rms2 = torch.clamp(rms2, min=1e-3)
            infer_wav = infer_wav * torch.pow(rms1 / rms2, 1.0 - self.rms_mix)

        sb = self.sola_buffer_frame
        conv_input = infer_wav[None, None, : sb + self.sola_search_frame]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(F.conv1d(conv_input ** 2, self.sola_den_kernel) + 1e-8)
        offset = int(torch.argmax(cor_nom[0, 0] / cor_den[0, 0]))
        infer_wav = infer_wav[offset:]
        infer_wav[:sb] *= self.fade_in
        infer_wav[:sb] += self.sola_buffer * self.fade_out
        self.sola_buffer[:] = infer_wav[bf: bf + sb]
        out = infer_wav[:bf].cpu().numpy()
        return out, {"infer_ms": round(t_infer * 1000, 1), **clean_stats}


# ---------------------------------------------------------------- app
config = None
engine = None
executor = ThreadPoolExecutor(max_workers=1)
router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def page():
    return PAGE.read_text(encoding="utf-8")


@router.get("/api/models")
def api_models():
    from tools import blend_tab, compare_tab
    from infer.vc.utils import get_index_path_from_model

    groups = {}
    for name in compare_tab.weight_names():
        key, label = blend_tab.model_meta(name)
        if key is None:
            continue
        idx = get_index_path_from_model(name, 0)
        groups.setdefault(key, {"label": label, "models": []})["models"].append({
            "file": name,
            "name": name[:-4],
            "short": blend_tab.short(name),
            "index": "none" if not idx else ("self" if blend_tab.SELF_INDEX_SUFFIX in idx else "yes"),
        })
    return {"groups": list(groups.values()), "current": engine.mix if engine else None}


@router.post("/api/save")
async def api_save(body: dict):
    from tools import blend_tab, compare_tab
    import re as _re

    models = list(body.get("models") or [])
    weights = [float(w) for w in body.get("weights") or []]
    name = str(body.get("name") or "")
    overwrite = bool(body.get("overwrite"))
    if len(models) < 2:
        return JSONResponse({"ok": False, "message": blend_tab.T["need_two"]})
    padded = (weights + [0] * blend_tab.MAX_MODELS)[: blend_tab.MAX_MODELS]

    def run():
        last = ""
        for h in blend_tab.run_blend(
            config, models, *padded, name, overwrite, blend_tab.T["index_blend"],
            compare_tab.T["upload"], None, None, 0, False,
        ):
            last = h
        return last

    html = await asyncio.get_running_loop().run_in_executor(None, run)
    text = _re.sub(r"<br\s*/?>", "\n", html)
    text = _re.sub(r"<[^>]+>", "", text).strip()
    ok = "pth" in text and blend_tab.T["failed"] not in text
    return {"ok": ok, "message": text}


@router.get("/api/stt")
def api_stt():
    return {"configured": bool(stt_api_key()), "models": list(STT_MODELS)}


@router.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()
    session = None
    stt = None
    stt_cfg = {}
    settings = {}  # every session setting this client sent; re-applied to each new Session
    pending = np.zeros(0, dtype=np.float32)
    dropped = 0
    send_lock = asyncio.Lock()

    async def send_json(obj):
        async with send_lock:
            await ws.send_text(json.dumps(obj, ensure_ascii=False))

    async def apply_settings(changes):
        """Apply settings to the session and always answer with one "state"
        message (the client counts them to know when it is in sync)."""
        try:
            settings.update(changes)
            if changes.get("clean") and changes["clean"] != session.clean_kinds():
                await send_json({"type": "clean_loading"})
            try:
                await loop.run_in_executor(None, session.apply, changes)
            except Exception as error:
                logger.warning("applying settings failed\n%s", traceback.format_exc())
                # Keep only what actually took effect so the next session matches.
                if "clean" in changes:
                    settings["clean"] = session.clean_kinds()
                await send_json({"type": "error", "scope": "clean" if "clean" in changes else "params",
                                 "message": str(error)})
        finally:
            await send_json({"type": "state", **session.state()})

    async def send_json_quiet(obj):
        try:
            await send_json(obj)
        except Exception:
            pass

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("text") is not None:
                data = json.loads(msg["text"])
                kind = data.get("type")
                try:
                    if kind == "start":
                        session = Session(
                            engine, data["sampleRate"],
                            block_time=float(data.get("blockTime", 0.25)),
                            extra_time=float(data.get("extraTime", 2.0)),
                        )
                        pending = np.zeros(0, dtype=np.float32)
                        if stt is not None:
                            await stt.close()
                        stt = LiveSTT(send_json_quiet, session.sr, session.block_frame)
                        stt.configure(stt_cfg)
                        await send_json({"type": "started", "blockFrames": session.block_frame,
                                         "sampleRate": session.sr})
                        await apply_settings({**settings, **(data.get("settings") or {})})
                    elif kind == "mix":
                        info = await loop.run_in_executor(
                            None, engine.set_mix, data["models"], data["weights"],
                            data.get("pitch", 0), float(data.get("indexRate", 0)),
                        )
                        await send_json({"type": "mix", **info})
                    elif kind == "stt":
                        stt_cfg.update({k: v for k, v in data.items() if k not in ("type", "reset", "flush")})
                        if stt is not None:
                            stt.configure(data)
                    elif kind == "pitch":
                        # Pitch belongs to the shared engine (like the mix), not the session.
                        engine.set_pitch(data["pitch"])
                    elif kind == "params":
                        changes = {k: v for k, v in data.items() if k != "type"}
                        if session is None:
                            settings.update(changes)  # applied when the session starts
                        else:
                            await apply_settings(changes)
                except Exception as error:
                    logger.warning("control message failed\n%s", traceback.format_exc())
                    await send_json({"type": "error", "message": str(error)})
                continue

            payload = msg.get("bytes")
            if session is None or not payload:
                continue
            chunk = np.frombuffer(payload, dtype=np.int16).astype(np.float32) / 32768.0
            pending = np.concatenate([pending, chunk])
            bf = session.block_frame
            # Keep latency bounded: if we fall behind, drop the oldest input.
            if len(pending) > 3 * bf:
                drop = len(pending) - bf
                drop -= drop % bf
                pending = pending[drop:]
                dropped += drop // bf
            while len(pending) >= bf:
                block, pending = pending[:bf], pending[bf:]
                t0 = time.perf_counter()
                out, stats = await loop.run_in_executor(executor, session.process, block)
                if stt is not None:
                    stt.feed(session.last_input if session.last_input is not None else block,
                             out, session.bypass)
                pcm = (np.clip(out, -1, 1) * 32767).astype(np.int16).tobytes()
                async with send_lock:
                    await ws.send_bytes(pcm)
                await send_json({
                    "type": "stats",
                    "proc_ms": round((time.perf_counter() - t0) * 1000, 1),
                    "block_ms": round(1000 * bf / session.sr),
                    "backlog_ms": round(1000 * len(pending) / session.sr),
                    "dropped": dropped,
                    **stats,
                    "state": session.state(),
                    **({"stt_buf": round(stt.seg.buffered, 1)} if stt is not None and stt.enabled else {}),
                })
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.warning("websocket error\n%s", traceback.format_exc())
    finally:
        if stt is not None:
            await stt.close()
        logger.info("client disconnected")


def create_app():
    app = FastAPI(title="RVC Live")
    app.include_router(router)
    app.include_router(router, prefix="/live")

    @app.get("/live")
    def _live_slash():
        return RedirectResponse("/live/")

    return app


def main():
    global config, engine
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7867)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    config = _load_config()
    logger.info("device %s, half %s", config.device, config.is_half)
    engine = Engine(config)
    # Load the pitch model now so the first live block does not stall.
    engine.rvc.get_f0(torch.zeros(16000, device=config.device), 0, "rmvpe")
    logger.info("ready")

    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
