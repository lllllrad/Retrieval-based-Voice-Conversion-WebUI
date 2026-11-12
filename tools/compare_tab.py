"""Model comparison tab: convert one input with many models x pitches and
listen to all results side by side."""

import html
import logging
import os
import re
import threading
import time
import traceback
from pathlib import Path

import gradio as gr
import soundfile as sf

from i18n.i18n import I18nAuto
from infer.vc.modules import VC
from infer.vc.utils import get_index_path_from_model

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "compare" / "out"

TEXT = {
    "ko_KR": {
        "tab": "모델 비교",
        "intro": "입력 음성 하나를 여러 모델과 피치값으로 변환해 결과를 나란히 들어봅니다. "
        "모델 융합(ckpt 처리 탭)으로 만든 모델도 새로고침하면 목록에 나타납니다.",
        "input": "입력 음성",
        "source_mode": "입력 방식",
        "upload": "파일",
        "mic": "마이크",
        "mic_input": "마이크 녹음 (녹음 후 '비교 시작')",
        "no_mic": "마이크로 녹음한 음성이 없습니다. 녹음을 마친 뒤 다시 시도하세요.",
        "models": "비교할 모델",
        "refresh": "목록 새로고침",
        "all": "전체 선택",
        "none": "선택 해제",
        "pitches": "피치 이동값 (반음, 공백으로 구분)",
        "pitches_hint": "예: 0 6 12  (남성 → 여성 모델은 +6~+12)",
        "f0": "피치 추출 방식",
        "index_rate": "인덱스 비율",
        "protect": "무성음 보호",
        "rms": "음량 엔벨로프 혼합 비율",
        "run": "비교 시작",
        "stop": "중지",
        "model": "모델",
        "pending": "대기",
        "running": "변환 중…",
        "failed": "실패",
        "no_input": "입력 음성을 올려 주세요.",
        "no_models": "모델을 하나 이상 선택해 주세요.",
        "busy": "다른 비교 작업이 실행 중입니다. 먼저 중지하세요.",
        "stopped": "중지되었습니다.",
        "stop_req": "중지 요청됨. 현재 항목이 끝나면 멈춥니다.",
        "progress": "진행: %d/%d · 저장 위치: %s",
        "done": "완료: %d개 성공, %d개 실패 · 저장 위치: %s",
        "source": "원본",
    },
    "en_US": {
        "tab": "Model Compare",
        "intro": "Convert one input with several models and pitch shifts, then listen side by side. "
        "Models made with Model Fusion (ckpt tab) appear here after refresh.",
        "input": "Input audio",
        "source_mode": "Input source",
        "upload": "File",
        "mic": "Microphone",
        "mic_input": "Microphone recording (record, then 'Run comparison')",
        "no_mic": "No microphone recording yet. Finish recording and try again.",
        "models": "Models to compare",
        "refresh": "Refresh list",
        "all": "Select all",
        "none": "Clear",
        "pitches": "Pitch shifts (semitones, space separated)",
        "pitches_hint": "e.g. 0 6 12  (male voice -> female model: +6 to +12)",
        "f0": "Pitch extraction",
        "index_rate": "Index rate",
        "protect": "Protect voiceless consonants",
        "rms": "Volume envelope mix rate",
        "run": "Run comparison",
        "stop": "Stop",
        "model": "Model",
        "pending": "pending",
        "running": "converting…",
        "failed": "failed",
        "no_input": "Upload an input audio file.",
        "no_models": "Select at least one model.",
        "busy": "Another comparison is running. Stop it first.",
        "stopped": "Stopped.",
        "stop_req": "Stop requested. It will stop after the current item.",
        "progress": "Progress: %d/%d · saved to: %s",
        "done": "Done: %d ok, %d failed · saved to: %s",
        "source": "Source",
    },
}

_lang = I18nAuto().language
T = TEXT.get(_lang, TEXT["en_US"])

_run_lock = threading.Lock()
_stop = threading.Event()
_vc = None


def weight_names():
    root = os.environ.get("weight_root", "assets/weights")
    return sorted(n for n in os.listdir(root) if n.endswith(".pth"))


def parse_pitches(text):
    values = []
    for token in re.findall(r"[+-]?\d+", str(text or "")):
        value = max(-24, min(24, int(token)))
        if value not in values:
            values.append(value)
    return values or [0]


def _audio_tag(path):
    url = "file=%s?t=%d" % (html.escape(str(path)), int(os.path.getmtime(path)))
    return '<audio controls preload="none" src="%s" style="width:220px"></audio>' % url


def _render(models, pitches, cells, source, message):
    head = "".join("<th>%+d</th>" % p for p in pitches)
    rows = []
    for m in models:
        tds = []
        for p in pitches:
            state = cells.get((m, p), ("pending", None))
            kind, value = state
            if kind == "ok":
                tds.append("<td>%s</td>" % _audio_tag(value))
            elif kind == "failed":
                tds.append(
                    '<td title="%s" style="color:#b91c1c">%s</td>'
                    % (html.escape(value or ""), T["failed"])
                )
            else:
                tds.append('<td style="color:#64748b">%s</td>' % T[kind])
        rows.append("<tr><td><b>%s</b></td>%s</tr>" % (html.escape(m[:-4]), "".join(tds)))
    src = ""
    if source:
        src = "<p>%s: %s</p>" % (T["source"], _audio_tag(source))
    return (
        "<p>%s</p>%s"
        '<div style="overflow-x:auto"><table style="border-collapse:collapse">'
        "<thead><tr><th>%s</th>%s</tr></thead><tbody>%s</tbody></table></div>"
        % (html.escape(message), src, T["model"], head, "".join(rows))
    )


def run_compare(config, input_path, models, pitch_text, f0_method, index_rate, protect, rms_mix_rate, order=None):
    global _vc
    if not input_path:
        yield "<p>%s</p>" % T["no_input"]
        return
    if not models:
        yield "<p>%s</p>" % T["no_models"]
        return
    if not _run_lock.acquire(blocking=False):
        yield "<p>%s</p>" % T["busy"]
        return
    try:
        _stop.clear()
        pitches = parse_pitches(pitch_text)
        available = set(weight_names())
        if order:
            models = [m for m in order if m in available]
        else:
            models = [m for m in weight_names() if m in set(models)]
        stem = re.sub(r"[^\w.-]+", "_", Path(input_path).stem)[:40] or "input"
        out_dir = OUT_ROOT / ("%s_%s" % (time.strftime("%Y%m%d-%H%M%S"), stem))
        out_dir.mkdir(parents=True, exist_ok=True)
        source = out_dir / ("source" + Path(input_path).suffix.lower())
        source.write_bytes(Path(input_path).read_bytes())
        rel = os.path.relpath(out_dir, PROJECT_ROOT)

        if _vc is None:
            _vc = VC(config)
        cells = {}
        total = len(models) * len(pitches)
        done = ok = 0
        for m in models:
            if _stop.is_set():
                break
            try:
                _vc.get_vc(m)
                index = get_index_path_from_model(m, 0) if index_rate > 0 else ""
            except Exception:
                err = traceback.format_exc()
                logger.warning("compare: failed to load %s\n%s", m, err)
                for p in pitches:
                    cells[(m, p)] = ("failed", err.strip().splitlines()[-1])
                    done += 1
                continue
            for p in pitches:
                if _stop.is_set():
                    break
                cells[(m, p)] = ("running", None)
                yield _render(models, pitches, cells, source, T["progress"] % (done, total, rel))
                try:
                    status, result = _vc.vc_single(
                        0, str(source), p, f0_method, index, index_rate, 0, rms_mix_rate, protect
                    )
                    if not result or result[0] is None or result[1] is None:
                        raise RuntimeError(str(status).strip().splitlines()[-1])
                    out = out_dir / ("%s_p%+d.wav" % (m[:-4], p))
                    sf.write(str(out), result[1], result[0])
                    cells[(m, p)] = ("ok", out)
                    ok += 1
                except Exception as error:
                    logger.warning("compare: %s pitch %s failed: %s", m, p, error)
                    cells[(m, p)] = ("failed", str(error))
                done += 1
        msg = T["stopped"] if _stop.is_set() else T["done"] % (ok, done - ok, rel)
        yield _render(models, pitches, cells, source, msg)
    finally:
        _run_lock.release()


def request_stop():
    _stop.set()
    return "<p>%s</p>" % T["stop_req"]


def build_compare_tab(config):
    with gr.TabItem(T["tab"]):
        gr.Markdown(T["intro"])
        with gr.Row():
            with gr.Column(scale=1):
                source_mode = gr.Radio(
                    label=T["source_mode"], choices=[T["upload"], T["mic"]], value=T["upload"]
                )
                audio = gr.Audio(label=T["input"], type="filepath")
                mic = gr.Audio(
                    label=T["mic_input"], source="microphone", type="filepath", visible=False
                )
                source_mode.change(
                    lambda mode: (
                        {"visible": mode != T["mic"], "__type__": "update"},
                        {"visible": mode == T["mic"], "__type__": "update"},
                    ),
                    [source_mode],
                    [audio, mic],
                    queue=False,
                )
                pitches = gr.Textbox(label=T["pitches"], value="0", placeholder=T["pitches_hint"])
                gr.Markdown(T["pitches_hint"])
                f0 = gr.Radio(label=T["f0"], choices=["rmvpe", "pm"], value="rmvpe")
                index_rate = gr.Slider(0, 1, value=0.75, step=0.05, label=T["index_rate"])
                protect = gr.Slider(0, 0.5, value=0.33, step=0.01, label=T["protect"])
                rms = gr.Slider(0, 1, value=0.25, step=0.05, label=T["rms"])
            with gr.Column(scale=2):
                models = gr.CheckboxGroup(label=T["models"], choices=weight_names(), value=[])
                with gr.Row():
                    refresh = gr.Button(T["refresh"])
                    select_all = gr.Button(T["all"])
                    select_none = gr.Button(T["none"])
                with gr.Row():
                    run = gr.Button(T["run"], variant="primary")
                    stop = gr.Button(T["stop"], variant="stop")
        result = gr.HTML()

        refresh.click(lambda: {"choices": weight_names(), "__type__": "update"}, [], [models])
        select_all.click(
            lambda: {"choices": weight_names(), "value": weight_names(), "__type__": "update"},
            [], [models],
        )
        select_none.click(lambda: {"value": [], "__type__": "update"}, [], [models])
        def on_run(mode, upload_path, mic_path, *args):
            if mode == T["mic"]:
                if not mic_path:
                    yield "<p>%s</p>" % T["no_mic"]
                    return
                yield from run_compare(config, mic_path, *args)
            else:
                yield from run_compare(config, upload_path, *args)

        run.click(
            on_run,
            [source_mode, audio, mic, models, pitches, f0, index_rate, protect, rms],
            [result],
            api_name="compare_models",
        )
        stop.click(request_stop, [], [result], queue=False)
