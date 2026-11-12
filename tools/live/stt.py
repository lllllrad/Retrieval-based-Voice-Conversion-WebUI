"""Live transcript comparison: original mic audio vs converted audio.

The live loop feeds every (input block, output block) pair to a Segmenter.
Speech is collected into batches of about `target` seconds (silence between
utterances trimmed); each batch is sent to Groq Whisper once per side, and
the two transcripts are compared at character level (CER-based match rate).
This is not meant to be instant; batching keeps API calls few.
"""

import asyncio
import io
import os
import re
import time
import unicodedata
from difflib import SequenceMatcher
from math import gcd
from pathlib import Path

import numpy as np

GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
MODELS = ("whisper-large-v3-turbo", "whisper-large-v3")
STT_SR = 16000


def api_key():
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    env = Path(__file__).resolve().parents[2] / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:].lstrip()
            if line.startswith("GROQ_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
                if key:
                    os.environ["GROQ_API_KEY"] = key
                return key
    return ""


# ---------------------------------------------------------------- segmenting
class Segmenter:
    """Collects speech from the block stream into batches for transcription.

    Utterances are found with input loudness; silence between them is trimmed
    to ``pad`` seconds, and utterances are queued until about ``target``
    seconds of audio are collected. One request per side then covers many
    sentences (Groq bills at least 10 s per request, and longer audio gives
    Whisper more context). A batch is also sent after ``idle`` seconds without
    speech, so nothing waits forever.
    """

    def __init__(self, sr, block_frames, vad_db=-45.0, gap=0.7, max_len=12.0,
                 min_voiced=0.4, target=20.0, idle=8.0, pad=0.4):
        self.sr = sr
        self.bf = block_frames
        self.block_s = block_frames / sr
        self.vad_db = vad_db
        self.gap = gap
        self.max_len = max_len
        self.min_voiced = min_voiced
        self.target = target
        self.idle = idle
        self.pad = pad
        self.batch = []          # finished utterances: (inp, out)
        self.batch_len = 0.0
        self.batch_t = None
        self.idle_s = 0.0
        self.reset()

    def reset(self, keep_batch=False):
        self.pre = None          # last silent block (pre-roll)
        self.inp, self.out = [], []
        self.voiced = 0.0
        self.silence = 0.0
        self.t_start = None
        if not keep_batch:
            self.batch, self.batch_len, self.batch_t, self.idle_s = [], 0.0, None, 0.0

    @property
    def buffered(self):
        """Seconds of audio waiting to be sent (queued + in progress)."""
        return self.batch_len + len(self.inp) * self.block_s

    def level_db(self, block):
        rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float64)) + 1e-12))
        return 20 * np.log10(rms)

    def feed(self, inp, out):
        """Returns (input_audio, output_audio, start_time) when a batch is ready."""
        loud = self.level_db(inp) > self.vad_db
        if not self.inp:
            if not loud:
                self.pre = (inp, out)
                if self.batch:
                    self.idle_s += self.block_s
                    if self.idle_s >= self.idle:
                        return self._take()
                return None
            self.t_start = time.time() - (self.block_s if self.pre else 0)
            if self.pre:
                self.inp.append(self.pre[0])
                self.out.append(self.pre[1])
            self.pre = None
        self.inp.append(inp)
        self.out.append(out)
        if loud:
            self.voiced += self.block_s
            self.silence = 0.0
        else:
            self.silence += self.block_s
        length = len(self.inp) * self.block_s
        if self.silence >= self.gap or length >= self.max_len:
            self._end_utterance()
            if self.batch_len >= self.target:
                return self._take()
        return None

    def _end_utterance(self):
        if self.inp and self.voiced >= self.min_voiced:
            keep = len(self.inp)
            extra = int(round((self.silence - self.pad) / self.block_s))
            if extra > 0:
                keep -= extra
            self.batch.append((np.concatenate(self.inp[:keep]), np.concatenate(self.out[:keep])))
            self.batch_len += keep * self.block_s
            if self.batch_t is None:
                self.batch_t = self.t_start
        self.idle_s = 0.0
        self.reset(keep_batch=True)

    def _take(self):
        if not self.batch:
            return None
        seg = (np.concatenate([b[0] for b in self.batch]),
               np.concatenate([b[1] for b in self.batch]), self.batch_t)
        self.batch, self.batch_len, self.batch_t, self.idle_s = [], 0.0, None, 0.0
        return seg

    def flush(self):
        """Send everything collected so far, including an unfinished utterance."""
        self._end_utterance()
        return self._take()


# ---------------------------------------------------------------- audio / API
def to_wav16k(x, sr):
    import soundfile as sf
    from scipy.signal import resample_poly

    x = np.asarray(x, dtype=np.float32)
    if sr != STT_SR:
        g = gcd(int(sr), STT_SR)
        x = resample_poly(x, STT_SR // g, int(sr) // g).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, np.clip(x, -1, 1), STT_SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class GroqError(Exception):
    pass


async def transcribe(client, wav, model, language):
    data = {"model": model, "temperature": "0", "response_format": "json"}
    if language:
        data["language"] = language
    for attempt in range(2):
        r = await client.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key()}"},
            data=data,
            files={"file": ("audio.wav", wav, "audio/wav")},
        )
        if r.status_code == 429 and attempt == 0:
            try:
                wait = float(r.headers.get("retry-after", "2"))
            except ValueError:
                wait = 2.0
            if wait <= 10:
                await asyncio.sleep(wait)
                continue
        if r.status_code != 200:
            try:
                msg = r.json().get("error", {}).get("message") or r.text
            except Exception:
                msg = r.text
            raise GroqError(f"Groq {r.status_code}: {msg[:300]}")
        return (r.json().get("text") or "").strip()
    raise GroqError("Groq 429: rate limit")


# ---------------------------------------------------------------- comparison
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def normalize(text):
    text = unicodedata.normalize("NFC", text).lower()
    text = _PUNCT.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def compare(ref, hyp):
    """Character-level comparison, ignoring spaces and punctuation.

    Returns edit count, reference length, match rate and diff ops for display
    (ops are computed on the normalized text with spaces kept).
    """
    nr, nh = normalize(ref), normalize(hyp)
    r, h = nr.replace(" ", ""), nh.replace(" ", "")
    edits = levenshtein(r, h)
    denom = max(len(r), 1)
    rate = max(0.0, 1.0 - edits / denom) if (r or h) else 1.0
    ops = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, nr, nh, autojunk=False).get_opcodes():
        if tag == "equal":
            ops.append(["=", nr[i1:i2]])
        else:
            if i2 > i1:
                ops.append(["-", nr[i1:i2]])
            if j2 > j1:
                ops.append(["+", nh[j1:j2]])
    return {"edits": edits, "ref_len": len(r), "rate": round(rate, 4), "ops": ops}


# ---------------------------------------------------------------- per-connection
class LiveSTT:
    """Owns the segmenter, HTTP client and running totals for one websocket."""

    def __init__(self, send_json, sr, block_frames):
        import httpx

        self.send_json = send_json
        self.sr = sr
        self.enabled = False
        self.model = MODELS[0]
        self.language = "ko"
        self.seg = Segmenter(sr, block_frames)
        self.client = httpx.AsyncClient(timeout=30.0)
        self.sem = asyncio.Semaphore(2)
        self.tasks = set()
        self.next_id = 0
        self.total_edits = 0
        self.total_ref = 0

    def configure(self, data):
        if "enabled" in data:
            enabled = bool(data["enabled"])
            if self.enabled and not enabled:
                self.flush()
            self.enabled = enabled
        if data.get("model") in MODELS:
            self.model = data["model"]
        if "language" in data:
            self.language = str(data["language"] or "")
        if "vadDb" in data:
            self.seg.vad_db = float(data["vadDb"])
        if "gap" in data:
            self.seg.gap = float(data["gap"])
        if "maxLen" in data:
            self.seg.max_len = float(data["maxLen"])
        if "target" in data:
            self.seg.target = float(data["target"])
            if self.enabled and self.seg.batch_len >= self.seg.target:
                self._spawn(self.seg._take())
        if data.get("flush") and self.enabled:
            self.flush()
        if data.get("reset"):
            self.seg.reset()
            self.total_edits = self.total_ref = 0

    def feed(self, inp, out, bypass):
        """Called once per processed block (inside the event loop)."""
        if not self.enabled:
            return
        if bypass:
            self.seg.reset(keep_batch=True)
            return
        seg = self.seg.feed(inp, out)
        if seg is not None:
            self._spawn(seg)

    def _spawn(self, seg):
        self.next_id += 1
        task = asyncio.create_task(self._run(self.next_id, *seg))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def _run(self, sid, inp, out, t_start):
        dur = round(len(inp) / self.sr, 2)
        await self.send_json({"type": "stt", "id": sid, "state": "pending", "t": t_start, "dur": dur})
        loop = asyncio.get_running_loop()
        try:
            if not api_key():
                raise GroqError("GROQ_API_KEY가 없습니다 (.env 확인)")
            async with self.sem:
                t0 = time.perf_counter()
                wav_in, wav_out = await asyncio.gather(
                    loop.run_in_executor(None, to_wav16k, inp, self.sr),
                    loop.run_in_executor(None, to_wav16k, out, self.sr),
                )
                src, conv = await asyncio.gather(
                    transcribe(self.client, wav_in, self.model, self.language),
                    transcribe(self.client, wav_out, self.model, self.language),
                )
                ms = round((time.perf_counter() - t0) * 1000)
            cmp = compare(src, conv)
            self.total_edits += cmp["edits"]
            self.total_ref += cmp["ref_len"]
            total = max(0.0, 1.0 - self.total_edits / self.total_ref) if self.total_ref else None
            await self.send_json({
                "type": "stt", "id": sid, "state": "done", "t": t_start, "dur": dur,
                "src": src, "conv": conv, "stt_ms": ms, **cmp,
                "total_rate": None if total is None else round(total, 4),
                "total_edits": self.total_edits, "total_ref": self.total_ref,
            })
        except Exception as error:
            await self.send_json({"type": "stt", "id": sid, "state": "error", "t": t_start,
                                  "dur": dur, "message": str(error)})

    def flush(self):
        """Send whatever utterance is in progress (e.g. when the user stops STT)."""
        seg = self.seg.flush()
        if seg is not None:
            self._spawn(seg)

    async def close(self):
        for task in list(self.tasks):
            task.cancel()
        if self.tasks:
            await asyncio.wait(set(self.tasks), timeout=5)
        await self.client.aclose()
