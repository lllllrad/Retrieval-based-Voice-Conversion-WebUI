"""Model management tab: list voice models, rename or delete them together
with their index files."""

import html
import json
import logging
import os
import re
from pathlib import Path

import gradio as gr
import torch

from infer.vc.utils import get_index_path_from_model
from tools import blend_tab, compare_tab

logger = logging.getLogger(__name__)

TEXT = {
    "ko_KR": {
        "tab": "모델 관리",
        "intro": "음성 모델의 이름을 바꾸거나 삭제합니다. 연결된 인덱스 파일도 함께 처리됩니다. "
        "다른 탭의 모델 목록은 각 탭의 새로고침 버튼을 누르면 반영됩니다.",
        "model": "모델",
        "refresh": "목록 새로고침",
        "new_name": "새 이름",
        "rename": "이름 변경",
        "confirm": "삭제 확인 (되돌릴 수 없음)",
        "delete": "삭제",
        "col_name": "이름",
        "col_type": "형식",
        "col_index": "인덱스",
        "col_info": "정보",
        "col_size": "크기",
        "none": "없음",
        "self": " (대체)",
        "pick": "모델을 선택하세요.",
        "bad_name": "사용할 수 없는 이름입니다. 영문·숫자·한글·_ - . 만 쓸 수 있습니다.",
        "same": "현재 이름과 같습니다.",
        "exists": "'%s' 모델이 이미 있습니다.",
        "busy": "비교나 합성 작업이 실행 중입니다. 끝난 뒤 다시 시도하세요.",
        "conflict": "이 이름을 쓰면 다른 모델의 인덱스 연결이 바뀌어서 취소했습니다: %s",
        "renamed": "이름 변경: %s → %s",
        "renamed_idx": "인덱스 이름 변경: %s → %s",
        "need_confirm": "삭제하려면 '삭제 확인'을 체크하세요.",
        "deleted": "삭제됨: %s",
        "kept_idx": "다른 모델도 쓰는 인덱스라 남겨 둠: %s",
    },
    "en_US": {
        "tab": "Manage Models",
        "intro": "Rename or delete voice models. Linked index files are handled too. "
        "Other tabs pick up changes when you press their refresh buttons.",
        "model": "Model",
        "refresh": "Refresh list",
        "new_name": "New name",
        "rename": "Rename",
        "confirm": "Confirm delete (cannot be undone)",
        "delete": "Delete",
        "col_name": "Name",
        "col_type": "Type",
        "col_index": "Index",
        "col_info": "Info",
        "col_size": "Size",
        "none": "none",
        "self": " (substitute)",
        "pick": "Select a model.",
        "bad_name": "Invalid name. Use letters, digits, _ - . only.",
        "same": "That is already the name.",
        "exists": "Model '%s' already exists.",
        "busy": "A compare or blend job is running. Try again when it finishes.",
        "conflict": "Cancelled: this name would change which index other models use: %s",
        "renamed": "Renamed: %s → %s",
        "renamed_idx": "Index renamed: %s → %s",
        "need_confirm": "Tick 'Confirm delete' to delete.",
        "deleted": "Deleted: %s",
        "kept_idx": "Kept index still used by another model: %s",
    },
}
T = TEXT.get(compare_tab._lang, TEXT["en_US"])


def _wr():
    return Path(os.environ.get("weight_root", "assets/weights"))


def _ir():
    return Path(os.environ.get("outside_index_root", "assets/indices"))


def index_map():
    """{model file: resolved index path or ''} for every model."""
    return {m: get_index_path_from_model(m, 0) or "" for m in compare_tab.weight_names()}


def model_table():
    rows = []
    for m in compare_tab.weight_names():
        path = _wr() / m
        try:
            ckpt = torch.load(str(path), map_location="cpu", weights_only=True, mmap=True)
            kind = "%s · %s" % (ckpt.get("version", "v1"), ckpt.get("sr"))
            info = str(ckpt.get("info", ""))
        except Exception as error:
            kind, info = "?", str(error)[:60]
        idx = get_index_path_from_model(m, 0)
        if idx:
            idx_label = html.escape(Path(idx).name)
            if blend_tab.SELF_INDEX_SUFFIX in Path(idx).name:
                idx_label += T["self"]
        else:
            idx_label = '<span style="color:#b45309">%s</span>' % T["none"]
        rows.append(
            "<tr><td><b>%s</b></td><td>%s</td><td>%s</td><td>%s</td><td>%.0f MB</td></tr>"
            % (html.escape(m[:-4]), kind, idx_label, html.escape(info[:80]), path.stat().st_size / 1e6)
        )
    head = "".join(
        "<th style='text-align:left;padding-right:16px'>%s</th>" % T[k]
        for k in ("col_name", "col_type", "col_index", "col_info", "col_size")
    )
    return (
        '<div style="overflow-x:auto"><table style="border-collapse:collapse">'
        "<thead><tr>%s</tr></thead><tbody>%s</tbody></table></div>" % (head, "".join(rows))
    )


def _linked_indexes(model, mapping):
    """Index files that belong to this model: the resolved one plus any
    substitute index and its metadata."""
    stem = model[:-4]
    files = set()
    if mapping.get(model):
        files.add(Path(mapping[model]).resolve())
    for p in _ir().glob("%s%s_v*.*" % (stem, blend_tab.SELF_INDEX_SUFFIX)):
        files.add(p.resolve())
    return sorted(files)


def _new_index_name(path, old_stem, new_stem):
    name = path.name
    experiment = re.sub(r"_e\d+_s\d+$", "", old_stem, flags=re.IGNORECASE)
    for old in (old_stem, experiment):
        if old and old in name:
            return name.replace(old, new_stem, 1)
    # Index matched by a looser rule: give it a name that matches exactly.
    return "%s%s" % (new_stem, path.suffix)


def _choices_update(value=None):
    names = [m[:-4] for m in compare_tab.weight_names()]
    return {"choices": names, "value": value if value in names else None, "__type__": "update"}


def _locked():
    if not compare_tab._run_lock.acquire(blocking=False):
        return False
    if not blend_tab._lock.acquire(blocking=False):
        compare_tab._run_lock.release()
        return False
    return True


def _unlock():
    blend_tab._lock.release()
    compare_tab._run_lock.release()


def rename_model(selected, new_name):
    if not selected:
        return T["pick"], model_table(), _choices_update(), ""
    new_stem = blend_tab.clean_name(new_name)
    if not new_stem or new_stem != str(new_name).strip():
        return T["bad_name"], model_table(), _choices_update(selected), new_name
    if new_stem == selected:
        return T["same"], model_table(), _choices_update(selected), new_name
    old_file, new_file = selected + ".pth", new_stem + ".pth"
    if (_wr() / new_file).exists():
        return T["exists"] % html.escape(new_stem), model_table(), _choices_update(selected), new_name
    if not _locked():
        return T["busy"], model_table(), _choices_update(selected), new_name
    try:
        before = index_map()
        moves = [(_wr() / old_file, _wr() / new_file)]
        for p in _linked_indexes(old_file, before):
            target = p.with_name(_new_index_name(p, selected, new_stem))
            if target != p:
                moves.append((p, target))
        if any(dst.exists() for _, dst in moves):
            clash = next(dst.name for _, dst in moves if dst.exists())
            return T["exists"] % html.escape(clash), model_table(), _choices_update(selected), new_name
        done = []
        try:
            for src, dst in moves:
                src.rename(dst)
                done.append((src, dst))
        except Exception:
            for src, dst in reversed(done):
                dst.rename(src)
            raise
        after = index_map()
        changed = [
            m for m in before
            if m != old_file and Path(after.get(m, "")).name != Path(before[m]).name
        ]
        moved_index = {Path(s).name: Path(d).name for s, d in done[1:]}
        expected = moved_index.get(Path(before[old_file]).name, Path(before[old_file]).name) if before[old_file] else ""
        if Path(after.get(new_file, "")).name != expected:
            changed.append(new_file)
        if changed:
            for src, dst in reversed(done):
                dst.rename(src)
            return (
                T["conflict"] % html.escape(", ".join(c[:-4] for c in changed)),
                model_table(), _choices_update(selected), new_name,
            )
        blend_tab._meta_cache.pop(old_file, None)
        log = [T["renamed"] % (html.escape(selected), html.escape(new_stem))]
        log += [T["renamed_idx"] % (html.escape(s.name), html.escape(d.name)) for s, d in done[1:]]
        return "<br>".join(log), model_table(), _choices_update(new_stem), ""
    finally:
        _unlock()


def delete_model(selected, confirmed):
    if not selected:
        return T["pick"], model_table(), _choices_update(), False
    if not confirmed:
        return T["need_confirm"], model_table(), _choices_update(selected), False
    if not _locked():
        return T["busy"], model_table(), _choices_update(selected), False
    try:
        model_file = selected + ".pth"
        mapping = index_map()
        others = {Path(v).resolve() for m, v in mapping.items() if m != model_file and v}
        log = []
        for p in _linked_indexes(model_file, mapping):
            if p.resolve() in others:
                log.append(T["kept_idx"] % html.escape(p.name))
                continue
            p.unlink(missing_ok=True)
            log.append(T["deleted"] % html.escape(p.name))
        (_wr() / model_file).unlink(missing_ok=True)
        blend_tab._meta_cache.pop(model_file, None)
        log.insert(0, T["deleted"] % html.escape(model_file))
        return "<br>".join(log), model_table(), _choices_update(), False
    finally:
        _unlock()


def build_manage_tab():
    with gr.TabItem(T["tab"]):
        gr.Markdown(T["intro"])
        with gr.Row():
            with gr.Column(scale=1):
                model = gr.Dropdown(label=T["model"], choices=[m[:-4] for m in compare_tab.weight_names()])
                refresh = gr.Button(T["refresh"])
            with gr.Column(scale=1):
                new_name = gr.Textbox(label=T["new_name"])
                rename = gr.Button(T["rename"], variant="primary")
            with gr.Column(scale=1):
                confirm = gr.Checkbox(label=T["confirm"], value=False)
                delete = gr.Button(T["delete"], variant="stop")
        status = gr.HTML()
        table = gr.HTML(value=model_table())

        def on_refresh(current):
            return model_table(), _choices_update(current)

        refresh.click(on_refresh, [model], [table, model], queue=False)
        model.change(lambda m: m or "", [model], [new_name], queue=False)
        rename.click(rename_model, [model, new_name], [status, table, model, new_name])
        delete.click(delete_model, [model, confirm], [status, table, model, confirm])
