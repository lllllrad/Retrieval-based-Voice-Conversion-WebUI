"""Voice blend tab: pick several compatible models, set a ratio for each,
merge them into a new model (plus a matching blended index) and preview it
next to the source models."""

import html
import json
import logging
import os
import re
import threading
import traceback
from collections import OrderedDict
from pathlib import Path

import gradio as gr
import numpy as np
import torch

from infer.vc.utils import get_index_path_from_model
from tools import compare_tab

logger = logging.getLogger(__name__)

MAX_MODELS = 8
PALETTE = ["#2563eb", "#db2777", "#16a34a", "#ea580c", "#7c3aed", "#0891b2", "#ca8a04", "#64748b"]

TEXT = {
    "ko_KR": {
        "tab": "목소리 합성",
        "intro": "같은 묶음(버전·샘플레이트)의 모델을 여러 개 골라 비율을 정하면, 가중 평균으로 새 모델을 만듭니다. "
        "입력 음성을 넣으면 합성 결과를 구성 모델들과 나란히 들어볼 수 있습니다. "
        "만든 모델은 다른 탭의 모델 목록에도 나타납니다.",
        "group": "① 모델 묶음 (서로 섞을 수 있는 모델끼리)",
        "models": "② 섞을 모델 선택 (최대 %d개)" % MAX_MODELS,
        "weights": "③ 비율 조절",
        "even": "균등하게",
        "refresh": "목록 새로고침",
        "name": "④ 새 모델 이름 (비우면 자동)",
        "overwrite": "같은 이름이 있으면 덮어쓰기",
        "index_mode": "인덱스",
        "index_blend": "비율대로 합친 인덱스 만들기",
        "index_none": "인덱스 없이",
        "preview": "⑤ 미리듣기 입력 (선택)",
        "with_sources": "구성 모델 결과도 함께 듣기",
        "pitch": "피치 이동 (반음)",
        "run": "합성하고 들어보기",
        "need_two": "모델을 두 개 이상 선택하세요.",
        "too_many": "모델은 최대 %d개까지 섞을 수 있습니다." % MAX_MODELS,
        "zero": "비율 합이 0입니다. 슬라이더를 올려 주세요.",
        "exists": "'%s' 모델이 이미 있습니다. 이름을 바꾸거나 '덮어쓰기'를 체크하세요.",
        "conflict": "이 이름을 쓰면 다른 모델의 인덱스 연결이 바뀌어서 취소했습니다: %s. 다른 이름을 쓰세요.",
        "busy": "다른 합성 작업이 실행 중입니다.",
        "merging": "모델 합성 중…",
        "indexing": "인덱스 합성 중… (모델 크기에 따라 수십 초)",
        "saved": "저장됨: %s",
        "index_saved": "인덱스: %s (%s개 벡터)",
        "index_skipped": "인덱스 없음",
        "index_missing": "인덱스도 참고 음성도 없어 인덱스에서 제외된 모델: %s (미리듣기 입력을 넣고 다시 실행하면 포함됩니다)",
        "self_index": "인덱스가 없는 모델의 대체 인덱스 생성 중: %s",
        "self_index_done": "대체 인덱스 생성: %s (참고 음성 %.0f초, %s개 벡터)",
        "no_preview": "미리듣기 입력이 없어 모델만 만들었습니다. 입력을 넣고 다시 누르거나 '모델 비교' 탭에서 들어보세요.",
        "auto_name": "자동 이름",
        "failed": "합성 실패",
    },
    "en_US": {
        "tab": "Voice Blend",
        "intro": "Pick several models from the same group (version and sample rate), set a ratio for each, "
        "and they are merged into a new model by weighted average. Add input audio to hear the blend "
        "next to its source models. New models also appear in the other tabs.",
        "group": "1. Model group (models that can be mixed)",
        "models": "2. Models to blend (up to %d)" % MAX_MODELS,
        "weights": "3. Ratios",
        "even": "Even split",
        "refresh": "Refresh list",
        "name": "4. New model name (blank = automatic)",
        "overwrite": "Overwrite if the name exists",
        "index_mode": "Index",
        "index_blend": "Build a blended index by ratio",
        "index_none": "No index",
        "preview": "5. Preview input (optional)",
        "with_sources": "Also play the source models",
        "pitch": "Pitch shift (semitones)",
        "run": "Blend and preview",
        "need_two": "Select at least two models.",
        "too_many": "You can blend up to %d models." % MAX_MODELS,
        "zero": "Ratios sum to 0. Raise a slider.",
        "exists": "Model '%s' already exists. Rename it or tick 'Overwrite'.",
        "conflict": "Cancelled: this name would change which index other models use: %s. Pick another name.",
        "busy": "Another blend is running.",
        "merging": "Blending model…",
        "indexing": "Blending index… (may take tens of seconds)",
        "saved": "Saved: %s",
        "index_saved": "Index: %s (%s vectors)",
        "index_skipped": "No index",
        "index_missing": "Left out of the index (no index and no reference audio): %s. Add preview input and run again to include them.",
        "self_index": "Building substitute index for: %s",
        "self_index_done": "Substitute index built: %s (%.0fs reference audio, %s vectors)",
        "no_preview": "No preview input, so only the model was created. Add input and run again, or use the Model Compare tab.",
        "auto_name": "Auto name",
        "failed": "Blend failed",
    },
}
T = TEXT.get(compare_tab._lang, TEXT["en_US"])

_lock = threading.Lock()
_meta_cache = {}


def _weight_root():
    return Path(os.environ.get("weight_root", "assets/weights"))


def _index_root():
    return Path(os.environ.get("outside_index_root", "assets/indices"))


def model_meta(name):
    """(group_key, label) for a model file; cached by mtime."""
    path = _weight_root() / name
    mtime = path.stat().st_mtime
    hit = _meta_cache.get(name)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
        version = ckpt.get("version", "v1")
        key = json.dumps([version, ckpt.get("f0", 1), ckpt["config"]])
        label = "%s · %s%s" % (version, ckpt.get("sr"), "" if ckpt.get("f0", 1) else " · no-f0")
        meta = (key, label)
    except Exception as error:
        logger.warning("blend: cannot read %s: %s", name, error)
        meta = (None, None)
    _meta_cache[name] = (mtime, meta)
    return meta


def groups():
    """{group label (with count): [model names]} for groups with 2+ models."""
    by_key = OrderedDict()
    labels = {}
    for name in compare_tab.weight_names():
        key, label = model_meta(name)
        if key is None:
            continue
        by_key.setdefault(key, []).append(name)
        labels[key] = label
    out = OrderedDict()
    seen = {}
    for key, names in by_key.items():
        if len(names) < 2:
            continue
        label = labels[key]
        seen[label] = seen.get(label, 0) + 1
        if seen[label] > 1:
            label = "%s (%d)" % (label, seen[label])
        out["%s  [%d]" % (label, len(names))] = names
    return out


def short(name):
    stem = name[:-4] if name.endswith(".pth") else name
    return re.sub(r"^kr_", "", stem)


def auto_name(models, weights):
    total = sum(weights) or 1
    parts = ["%s%d" % (short(m), round(100 * w / total)) for m, w in zip(models, weights)]
    return "mix_" + "_".join(parts)


def clean_name(text):
    return re.sub(r"[^\w.-]+", "_", str(text or "").strip()).strip("._")[:80]


def ratio_bar(models, weights):
    total = sum(weights)
    if not models:
        return ""
    if total <= 0:
        return "<p>%s</p>" % T["zero"]
    segs, legend = [], []
    for i, (m, w) in enumerate(zip(models, weights)):
        pct = 100 * w / total
        color = PALETTE[i % len(PALETTE)]
        if pct > 0:
            segs.append(
                '<div style="width:%.2f%%;background:%s;color:#fff;font-size:12px;'
                'text-align:center;overflow:hidden;white-space:nowrap">%s</div>'
                % (pct, color, html.escape(short(m)) if pct >= 8 else "")
            )
        legend.append(
            '<span style="margin-right:12px"><span style="display:inline-block;width:10px;height:10px;'
            'background:%s;margin-right:4px"></span>%s <b>%.0f%%</b></span>'
            % (color, html.escape(short(m)), pct)
        )
    return (
        '<div style="display:flex;height:26px;border-radius:6px;overflow:hidden;line-height:26px">%s</div>'
        '<div style="margin-top:6px">%s</div><div style="margin-top:6px;color:#64748b">%s: <code>%s</code></div>'
        % ("".join(segs), "".join(legend), T["auto_name"], html.escape(auto_name(models, weights)))
    )


def merge_models(models, weights, out_path):
    total = float(sum(weights))
    ws = [w / total for w in weights]
    ckpts = [
        torch.load(str(_weight_root() / m), map_location="cpu", weights_only=True) for m in models
    ]
    base = ckpts[0]
    tensors = [c["weight"] for c in ckpts]
    keys = list(tensors[0].keys())
    for m, t in zip(models[1:], tensors[1:]):
        if sorted(t.keys()) != sorted(keys):
            raise ValueError("model structure differs: %s" % m)
    out = OrderedDict()
    for key in keys:
        if key == "emb_g.weight":
            n = min(t[key].shape[0] for t in tensors)
            out[key] = sum(w * t[key][:n].float() for w, t in zip(ws, tensors)).half()
            continue
        shape = tensors[0][key].shape
        for m, t in zip(models, tensors):
            if t[key].shape != shape:
                raise ValueError("shape mismatch in %s for %s" % (m, key))
        out[key] = sum(w * t[key].float() for w, t in zip(ws, tensors)).half()
    config = list(base["config"])
    config[-3] = int(out["emb_g.weight"].shape[0])
    recipe = ", ".join("%s %.0f%%" % (short(m), 100 * w) for m, w in zip(models, ws))
    opt = OrderedDict()
    opt["weight"] = out
    opt["config"] = config
    opt["sr"] = base["sr"]
    opt["f0"] = base.get("f0", 1)
    opt["version"] = base.get("version", "v1")
    opt["info"] = "blend: " + recipe
    torch.save(opt, str(out_path))
    return opt["version"], recipe


SELF_INDEX_SUFFIX = "_selfidx"
MAX_REFERENCE_SECONDS = 600


def write_ivf_index(vectors, version, out_path):
    import faiss

    n = vectors.shape[0]
    n_ivf = max(1, min(int(16 * np.sqrt(n)), n // 39))
    index = faiss.index_factory(256 if version == "v1" else 768, "IVF%d,Flat" % n_ivf)
    faiss.extract_index_ivf(index).nprobe = 1
    index.train(vectors)
    for start in range(0, n, 8192):
        index.add(vectors[start : start + 8192])
    faiss.write_index(index, str(out_path))


def read_index_vectors(path):
    import faiss

    index = faiss.read_index(str(path))
    faiss.extract_index_ivf(index).make_direct_map()
    return index.reconstruct_n(0, index.ntotal).astype(np.float32)


def reference_sources(extra=None):
    """Speech to build substitute indexes from: the given input plus earlier
    comparison inputs, newest first, up to MAX_REFERENCE_SECONDS."""
    import soundfile as sf

    paths = [Path(extra)] if extra else []
    paths += sorted(compare_tab.OUT_ROOT.glob("*/source.*"), reverse=True)
    out, total, seen = [], 0.0, set()
    for path in paths:
        try:
            key = (path.stat().st_size, path.read_bytes()[:4096])
            if key in seen:
                continue
            seen.add(key)
            seconds = sf.info(str(path)).duration
        except Exception:
            continue
        out.append(path)
        total += seconds
        if total >= MAX_REFERENCE_SECONDS:
            break
    return out, total


def build_self_index(config, model, sources, pitches):
    """Make a substitute index for a model that has none: convert reference
    speech with the model itself and index HuBERT features of the result."""
    from infer.hubert import extract_hubert_features

    if compare_tab._vc is None:
        compare_tab._vc = compare_tab.VC(config)
    vc = compare_tab._vc
    vc.get_vc(model)
    version = vc.version
    feats = []
    for src in sources:
        for pitch in pitches:
            _, result = vc.vc_single(0, str(src), pitch, "rmvpe", "", 0, 16000, 0.25, 0.33)
            if not result or result[1] is None:
                continue
            sr, audio = result
            audio = np.asarray(audio)
            if np.issubdtype(audio.dtype, np.integer):
                audio = audio.astype(np.float32) / 32768.0
            if sr != 16000:
                import librosa

                audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
            for start in range(0, len(audio), 16000 * 30):
                chunk = torch.from_numpy(np.ascontiguousarray(audio[start : start + 16000 * 30]))
                if chunk.numel() < 16000:
                    continue
                chunk = chunk.to(config.device).view(1, -1)
                chunk = chunk.half() if config.is_half else chunk.float()
                with torch.no_grad():
                    f = extract_hubert_features(vc.hubert_model, chunk, version)
                feats.append(f[0].float().cpu().numpy())
    if not feats:
        return None, 0
    vectors = np.concatenate(feats).astype(np.float32)
    out_path = _index_root() / ("%s%s_%s.index" % (model[:-4], SELF_INDEX_SUFFIX, version))
    write_ivf_index(vectors, version, out_path)
    return out_path, vectors.shape[0]


def needs_self_index(model, reference_seconds):
    """True if the model has no index, or only a substitute index built from
    clearly less reference audio than is available now."""
    path = get_index_path_from_model(model, 0)
    if not path:
        return True
    if SELF_INDEX_SUFFIX not in Path(path).name:
        return False
    meta = Path(path).with_suffix(".json")
    try:
        built = float(json.loads(meta.read_text())["reference_seconds"])
    except Exception:
        built = 0.0
    return reference_seconds >= max(built * 1.5, built + 30)


def blend_index(models, weights, version, out_path, seed=0):
    vecs, used, missing = [], [], []
    for m, w in zip(models, weights):
        path = get_index_path_from_model(m, 0)
        if not path or w <= 0:
            if w > 0:
                missing.append(short(m))
            continue
        vecs.append(read_index_vectors(path))
        used.append(w)
    if not vecs:
        return None, 0, missing
    # Give each model a share of vectors equal to its ratio. Small indexes are
    # resampled with replacement so they do not shrink everyone else's share.
    total_w = sum(used)
    target = min(200000, sum(len(v) for v in vecs))
    rng = np.random.default_rng(seed)
    picked = []
    for v, w in zip(vecs, used):
        n = max(1, int(round(target * w / total_w)))
        picked.append(v[rng.choice(len(v), n, replace=n > len(v))])
    big = np.concatenate(picked)
    rng.shuffle(big)
    write_ivf_index(big, version, out_path)
    return out_path, big.shape[0], missing


def run_blend(config, models, *rest):
    weights = list(rest[:MAX_MODELS])
    name, overwrite, index_mode, mode, upload, mic, pitch, with_sources = rest[MAX_MODELS:]
    models = list(models or [])
    weights = [float(w or 0) for w in weights[: len(models)]]
    if len(models) < 2:
        yield "<p>%s</p>" % T["need_two"]
        return
    if len(models) > MAX_MODELS:
        yield "<p>%s</p>" % T["too_many"]
        return
    if sum(weights) <= 0:
        yield "<p>%s</p>" % T["zero"]
        return
    name = clean_name(name) or clean_name(auto_name(models, weights))
    out_path = _weight_root() / (name + ".pth")
    if out_path.exists() and not overwrite:
        yield "<p>%s</p>" % (T["exists"] % html.escape(name))
        return
    if not _lock.acquire(blocking=False):
        yield "<p>%s</p>" % T["busy"]
        return
    try:
        log = []
        before = {
            m: get_index_path_from_model(m, 0) or ""
            for m in compare_tab.weight_names() if m != out_path.name
        }
        yield "<p>%s</p>" % T["merging"]
        version, recipe = merge_models(models, weights, out_path)
        _meta_cache.pop(out_path.name, None)
        log.append(T["saved"] % html.escape("%s.pth  (%s)" % (name, recipe)))

        for old in _index_root().glob("added_%s_v*.index" % name):
            old.unlink()
        source = mic if mode == compare_tab.T["mic"] else upload
        if index_mode == T["index_blend"]:
            sources, seconds = reference_sources(source)
            need = [m for m, w in zip(models, weights) if w > 0 and needs_self_index(m, seconds)]
            if need:
                if sources:
                    yield "<p>%s</p><p>%s</p>" % (
                        "<br>".join(log), T["self_index"] % html.escape(", ".join(short(m) for m in need))
                    )
                    if not compare_tab._run_lock.acquire(blocking=False):
                        yield "<p>%s</p>" % compare_tab.T["busy"]
                        return
                    try:
                        pitches = sorted({0, int(pitch or 0)})
                        for m in need:
                            path, count = build_self_index(config, m, sources, pitches)
                            if path:
                                path.with_suffix(".json").write_text(
                                    json.dumps({"reference_seconds": seconds, "vectors": count})
                                )
                                log.append(T["self_index_done"] % (html.escape(path.name), seconds, count))
                    finally:
                        compare_tab._run_lock.release()
            yield "<p>%s</p><p>%s</p>" % ("<br>".join(log), T["indexing"])
            index_path = _index_root() / ("added_%s_%s.index" % (name, version))
            path, count, missing = blend_index(models, weights, version, index_path)
            log.append(T["index_saved"] % (html.escape(path.name), count) if path else T["index_skipped"])
            if missing:
                log.append(T["index_missing"] % html.escape(", ".join(missing)))
        else:
            log.append(T["index_skipped"])
        changed = [m for m, idx in before.items() if (get_index_path_from_model(m, 0) or "") != idx]
        if changed:
            out_path.unlink(missing_ok=True)
            for old in _index_root().glob("added_%s_v*.index" % name):
                old.unlink()
            yield "<p>%s</p>" % (T["conflict"] % html.escape(", ".join(m[:-4] for m in changed)))
            return
        summary = "<p>%s</p>" % "<br>".join(log)

        if not source:
            yield summary + "<p>%s</p>" % T["no_preview"]
            return
        to_play = [out_path.name] + (models if with_sources else [])
        index_rate = 0.75 if index_mode == T["index_blend"] else 0.0
        for table in compare_tab.run_compare(
            config, source, to_play, str(int(pitch or 0)), "rmvpe", index_rate, 0.33, 0.25,
            order=to_play,
        ):
            yield summary + table
    except Exception:
        logger.warning("blend failed\n%s", traceback.format_exc())
        yield "<p><b>%s</b></p><pre>%s</pre>" % (
            T["failed"], html.escape(traceback.format_exc()[-2000:])
        )
    finally:
        _lock.release()


def build_blend_tab(config):
    def slider_updates(models, weights=None):
        models = list(models or [])[:MAX_MODELS]
        ups = []
        for i in range(MAX_MODELS):
            if i < len(models):
                up = {"visible": True, "label": short(models[i]), "__type__": "update"}
                if weights is not None:
                    up["value"] = weights[i]
                ups.append(up)
            else:
                ups.append({"visible": False, "__type__": "update"})
        return ups

    with gr.TabItem(T["tab"]):
        gr.Markdown(T["intro"])
        init_groups = groups()
        first = next(iter(init_groups), None)
        with gr.Row():
            with gr.Column(scale=1):
                group = gr.Dropdown(label=T["group"], choices=list(init_groups), value=first)
                refresh = gr.Button(T["refresh"])
                models = gr.CheckboxGroup(
                    label=T["models"], choices=init_groups.get(first, []), value=[]
                )
            with gr.Column(scale=1):
                gr.Markdown("**%s**" % T["weights"])
                sliders = [
                    gr.Slider(0, 100, value=50, step=1, label="", visible=False)
                    for _ in range(MAX_MODELS)
                ]
                even = gr.Button(T["even"])
                bar = gr.HTML()
        with gr.Row():
            with gr.Column(scale=1):
                name = gr.Textbox(label=T["name"], placeholder="mix_...")
                overwrite = gr.Checkbox(label=T["overwrite"], value=False)
                index_mode = gr.Radio(
                    label=T["index_mode"],
                    choices=[T["index_blend"], T["index_none"]],
                    value=T["index_blend"],
                )
            with gr.Column(scale=1):
                gr.Markdown("**%s**" % T["preview"])
                mode = gr.Radio(
                    choices=[compare_tab.T["upload"], compare_tab.T["mic"]],
                    value=compare_tab.T["upload"],
                    label=compare_tab.T["source_mode"],
                )
                upload = gr.Audio(label=compare_tab.T["input"], type="filepath")
                mic = gr.Audio(
                    label=compare_tab.T["mic_input"], source="microphone", type="filepath",
                    visible=False,
                )
                with gr.Row():
                    pitch = gr.Number(label=T["pitch"], value=0, precision=0)
                    with_sources = gr.Checkbox(label=T["with_sources"], value=True)
        run = gr.Button(T["run"], variant="primary")
        result = gr.HTML()

        def on_group(label):
            names = groups().get(label, [])
            return [{"choices": names, "value": [], "__type__": "update"}] + slider_updates([]) + [""]

        def on_refresh(label):
            g = groups()
            label = label if label in g else next(iter(g), None)
            return [
                {"choices": list(g), "value": label, "__type__": "update"},
                {"choices": g.get(label, []), "value": [], "__type__": "update"},
            ] + slider_updates([]) + [""]

        def on_models(selected, *weights):
            selected = list(selected or [])[:MAX_MODELS]
            even_w = [50] * len(selected)
            return slider_updates(selected, even_w) + [ratio_bar(selected, even_w)]

        def on_weights(selected, *weights):
            selected = list(selected or [])[:MAX_MODELS]
            return ratio_bar(selected, [float(w or 0) for w in weights[: len(selected)]])

        def on_even(selected):
            selected = list(selected or [])[:MAX_MODELS]
            even_w = [50] * len(selected)
            return slider_updates(selected, even_w) + [ratio_bar(selected, even_w)]

        def on_run(*args):
            yield from run_blend(config, *args)

        group.change(on_group, [group], [models] + sliders + [bar], queue=False)
        refresh.click(on_refresh, [group], [group, models] + sliders + [bar], queue=False)
        models.change(on_models, [models] + sliders, sliders + [bar], queue=False)
        for s in sliders:
            s.change(on_weights, [models] + sliders, [bar], queue=False)
        even.click(on_even, [models], sliders + [bar], queue=False)
        mode.change(
            lambda m: (
                {"visible": m != compare_tab.T["mic"], "__type__": "update"},
                {"visible": m == compare_tab.T["mic"], "__type__": "update"},
            ),
            [mode], [upload, mic], queue=False,
        )
        run.click(
            on_run,
            [models] + sliders + [name, overwrite, index_mode, mode, upload, mic, pitch, with_sources],
            [result],
            api_name="blend_models",
        )
