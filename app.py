# app.py
import os
import shutil
import uuid

import gradio as gr
import spatchat_ui
import spatchat_files as sfiles
import spatchat_layout
from gradio.context import LocalContext

from welcome import WELCOME_MESSAGE
from raster_io import notify_upload
from handlers import analyze_raster
from data_views import class_table, raster_plots, metric_plots_from_reply

try:
    from spatchat_gate import metered_chat
except ImportError:  # launched without spatchat-gateway's client/ on PYTHONPATH
    def metered_chat(**_kwargs):
        return lambda fn: fn


# ---------------------------------------------------------------------------------------------
# Per-visitor workspace: sessions/<id>/uploads and sessions/<id>/outputs, removed when the tab closes.
# ---------------------------------------------------------------------------------------------
SESSIONS_ROOT = "sessions"
shutil.rmtree(SESSIONS_ROOT, ignore_errors=True)  # leftovers of a previous process belong to nobody
os.makedirs(SESSIONS_ROOT, exist_ok=True)
_HASH_TO_SID: dict[str, str] = {}
_ACTIVE: dict[str, str] = {}  # session id -> path of the raster the analyses run on


def _active(sid):
    """The visitor's active raster, if its file still exists."""
    path = _ACTIVE.get(sid or "")
    return path if path and os.path.isfile(path) else None


def _request_hash():
    try:
        request = LocalContext.request.get()
    except LookupError:
        return None
    return str(getattr(request, "session_hash", "") or "") or None


def _session(sid):
    """(session id, uploads dir, outputs dir), created on first use."""
    sid = (sid or "").strip() or uuid.uuid4().hex
    uploads = os.path.join(SESSIONS_ROOT, sid, "uploads")
    outputs = os.path.join(SESSIONS_ROOT, sid, "outputs")
    os.makedirs(uploads, exist_ok=True)
    os.makedirs(outputs, exist_ok=True)
    rh = _request_hash()
    if rh:
        _HASH_TO_SID[rh] = sid
    return sid, uploads, outputs


def _cleanup_browser_session():
    rh = _request_hash()
    sid = _HASH_TO_SID.pop(rh, None) if rh else None
    if sid:
        _ACTIVE.pop(sid, None)
        shutil.rmtree(os.path.join(SESSIONS_ROOT, sid), ignore_errors=True)


def _filepath(file):
    return getattr(file, "name", file)


def _adopt(file, sid):
    """Copy an uploaded raster into this visitor's uploads folder. Returns (managed path, session id)."""
    sid, uploads, _ = _session(sid)
    src = _filepath(file)
    if not src or not os.path.isfile(src):
        return None, sid
    name = os.path.basename(src)
    stem, ext = os.path.splitext(name)
    dst, n = os.path.join(uploads, name), 2
    while os.path.exists(dst):
        dst = os.path.join(uploads, f"{stem}_{n}{ext}")
        n += 1
    shutil.copy2(src, dst)
    return dst, sid


def _chat_denied(markdown, file, user_msg, history):
    """Shown instead of an answer when the gateway refuses the message (limit reached, session expired)."""
    hist = list(history or []) + [
        {"role": "user", "content": str(user_msg or "")},
        {"role": "assistant", "content": markdown},
    ]
    return hist, ""


_HERE = os.path.dirname(os.path.abspath(__file__))
_SAMPLE_DIR = os.path.join(_HERE, "sample_data")


def _sample_raster() -> str:
    """The demo land-cover raster, placed near Boulder, Colorado so it shows on the map.
    Built on first use from the class pattern in dummy_lulc.tif."""
    dst = os.path.join(_SAMPLE_DIR, "demo_landcover.tif")
    if not os.path.exists(dst):
        import rasterio
        from rasterio.transform import from_origin

        os.makedirs(_SAMPLE_DIR, exist_ok=True)
        with rasterio.open(os.path.join(_HERE, "dummy_lulc.tif")) as src:
            data, profile = src.read(1), src.profile.copy()
        profile.update(crs="EPSG:32613", transform=from_origin(476000, 4437000, 30, 30), driver="GTiff")
        with rasterio.open(dst, "w", **profile) as out:
            out.write(data, 1)
    return dst

_EMPTY_SHEET = spatchat_ui.empty_state_html(
    "table", "No raster loaded",
    "Click + in the chat, drag a GeoTIFF onto this panel, or choose &ldquo;Try sample data&rdquo;. "
    "A table of land-cover classes will appear here.",
)
_EMPTY_PLOTS = spatchat_ui.empty_state_html(
    "plot", "No plots yet",
    "Load a land-cover raster and the class composition and patch-size charts will appear here.",
)


# ---------------------------------------------------------------------------------------------
# What the Datasheet, Plots and Files tabs show
# ---------------------------------------------------------------------------------------------
def _entries(sid):
    """(uploaded, produced) for the Files tab."""
    if not sid:
        return [], []
    _, uploads, outputs = _session(sid)
    active = os.path.abspath(_active(sid) or "")

    def up_note(path):
        return "Land-cover raster" + (" (active)" if os.path.abspath(path) == active else "")

    def out_note(path):
        ext = os.path.splitext(path)[1].lower()
        return {".png": "Chart", ".csv": "Table"}.get(ext, "")

    uploaded = sfiles.scan(uploads, sfiles.UPLOADED, note_for=up_note)
    produced = sfiles.scan(outputs, sfiles.PRODUCED, note_for=out_note)
    sfiles.stamp_sources(produced, os.path.join(SESSIONS_ROOT, sid, ".provenance.json"),
                         os.path.basename(_active(sid) or ""))
    return uploaded, produced


def _views(sid, open_table=False):
    """Plots + Files for the current raster. Reads what is on disk; writes nothing."""
    outputs = _session(sid)[2] if sid else None
    gal, gal_empty = spatchat_ui.gallery_updates(gr, spatchat_ui.list_images(outputs), _EMPTY_PLOTS)
    file = _active(sid)
    uploaded, produced = _entries(sid)
    summary = next((e["path"] for e in produced if e["name"] == "class_summary.csv"), None)
    return (gal, gal_empty,
            *files_panel.updates(gr, uploaded, produced, preview_path=summary if (open_table and file) else None,
                                 reset_preview=not file))


def _views_open(sid):
    """After an upload: refresh, and open the class table."""
    return _views(sid, open_table=True)


def _build_outputs(file, sid):
    """Charts and class table for a newly active raster, written to the visitor's outputs folder."""
    if not file or not sid:
        return
    outputs = _session(sid)[2]
    try:
        raster_plots(file, outputs)
    except Exception as exc:
        print("[landmetrics] could not summarise raster:", exc)


def _message_text(message) -> str:
    """Plain text of a chat message. Gradio 6 delivers content as a list of blocks, Gradio 5 as a string."""
    content = message.get("content") if isinstance(message, dict) else message
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "\n".join(
            _message_text(part.get("text") if isinstance(part, dict) and "text" in part else part)
            for part in content if part)
    return "" if content is None else str(content)


def _after_chat(history, sid):
    """Chart (and save) the table in the newest assistant reply."""
    last = next((m for m in reversed(history or []) if m.get("role") == "assistant"), None)
    if _active(sid) and last:
        metric_plots_from_reply(_message_text(last), _session(sid)[2])


def _on_upload(file, sid):
    managed, sid = _adopt(file, sid)
    if managed:
        _ACTIVE[sid] = managed
        _build_outputs(managed, sid)
    return sid


def _on_sample(sid):
    managed, sid = _adopt(_sample_raster(), sid)
    _ACTIVE[sid] = managed
    _build_outputs(managed, sid)
    return sid


def _map_for(sid):
    """Map tab: the active raster with its class legend, or the standard empty map."""
    path = _active(sid)
    if not path:
        return spatchat_layout.empty_map("Drop a land-cover GeoTIFF here, or click + to start",
                                         "Your raster appears on this map, with its classes in the legend.")
    try:
        return spatchat_layout.raster_map(path, label=os.path.basename(path))
    except Exception as exc:  # no CRS, text labels, ...: explain instead of showing a blank map
        return spatchat_layout.empty_map("This raster can't be shown on the map", str(exc)[:200])


def _notify(sid, history):
    return notify_upload(_active(sid), history)


def _chat(sid, user_msg, history):
    """The metrics chat, always run on this visitor's active raster."""
    return analyze_raster(_active(sid), user_msg, history)


def _chat_denied_for(markdown, sid, user_msg, history):
    return _chat_denied(markdown, None, user_msg, history)


def _files_remove(rows, paths, sid, history):
    """Delete the ticked files; if the active raster goes, fall back to the newest one left (or none)."""
    sid, uploads, outputs = _session(sid)
    root = os.path.join(SESSIONS_ROOT, sid)
    removed, problems = sfiles.remove_paths(files_panel.ticked(rows, paths), [root])
    gone = {os.path.abspath(p) for p in removed}
    active = _active(sid)
    history = list(history or [])
    if (active and os.path.abspath(active) in gone) or (not active and sid in _ACTIVE):
        left = sfiles.scan(uploads, sfiles.UPLOADED)
        new_active = left[0]["path"] if left else None
        if new_active:
            _ACTIVE[sid] = new_active
            _build_outputs(new_active, sid)
        else:
            _ACTIVE.pop(sid, None)
        note = f"Now using **{os.path.basename(new_active)}**." if new_active else "No raster loaded."
        history.append({"role": "assistant", "content": f"{sfiles.removal_message(removed)}. {note}"})
    elif gone:
        history.append({"role": "assistant", "content": sfiles.removal_message(removed)})
    return (*files_panel.updates(gr, *_entries(sid), message=sfiles.removal_message(removed), problems=problems,
                                 reset_preview=bool(removed)),
            history, sid)


def _files_download(rows, paths, sid):
    root = os.path.join(SESSIONS_ROOT, sid) if sid else None
    return files_panel.download_update(gr, rows, paths, [root])


# ---------------------------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------------------------
_SUGGESTION_CHIPS = [
    "PLAND for all classes",
    "SHDI at the landscape level",
    "Largest patch index (LPI)",
    "Edge density (ED)",
]

with gr.Blocks(title="Spatchat: Landscape Metrics") as iface:
    ui = spatchat_layout.build(
        gr,
        title="Landscape Metrics",
        subtitle="Upload a land-cover GeoTIFF and ask for landscape metrics in plain language.",
        room="landmetrics",
        cite="Wan, H.Y. &amp; Hysen, L. (2025). <i>Spatchat: Landscape Metrics.</i>",
        welcome=WELCOME_MESSAGE,
        chips=_SUGGESTION_CHIPS,
        placeholder="e.g. calculate edge density",
        file_types=[".tif", ".tiff"],
        map_html=_map_for(None),
        plots_empty_html=_EMPTY_PLOTS,
    )
    chatbot, sample_btn, file_input, txt_in, send_btn = ui.chatbot, ui.sample_btn, ui.file_input, ui.user_input, ui.send_btn
    gallery, gallery_empty, files_panel = ui.gallery, ui.gallery_empty, ui.files
    session_state, map_out = ui.session_state, ui.map_out

    iface.queue(max_size=16)
    iface.unload(_cleanup_browser_session)

    VIEW_OUTPUTS = [gallery, gallery_empty, *files_panel.outputs]

    file_input.upload(_on_upload, inputs=[file_input, session_state], outputs=session_state
    ).then(_map_for, inputs=session_state, outputs=map_out
    ).then(_notify, inputs=[session_state, chatbot], outputs=[chatbot, txt_in]
    ).then(_views_open, inputs=session_state, outputs=VIEW_OUTPUTS)

    sample_btn.click(_on_sample, inputs=session_state, outputs=session_state
    ).then(_map_for, inputs=session_state, outputs=map_out
    ).then(_notify, inputs=[session_state, chatbot], outputs=[chatbot, txt_in]
    ).then(_views_open, inputs=session_state, outputs=VIEW_OUTPUTS)

    files_panel.refresh_btn.click(_views, inputs=session_state, outputs=VIEW_OUTPUTS)
    files_panel.remove_btn.click(
        _files_remove,
        inputs=[*files_panel.selections, session_state, chatbot],
        outputs=[*files_panel.outputs, chatbot, session_state],
    ).then(_map_for, inputs=session_state, outputs=map_out
    ).then(_views, inputs=session_state, outputs=VIEW_OUTPUTS, show_progress="hidden")
    files_panel.table.input(_files_download, inputs=[*files_panel.selections, session_state],
                            outputs=files_panel.download_btn, show_progress="hidden")

    gr.on(triggers=[txt_in.submit, send_btn.click],
          fn=metered_chat(room="landmetrics", message_index=1, deny=_chat_denied_for)(_chat),
          inputs=[session_state, txt_in, chatbot],
          outputs=[chatbot, txt_in]
    ).then(_after_chat, inputs=[chatbot, session_state], outputs=None
    ).then(_views, inputs=session_state, outputs=VIEW_OUTPUTS, show_progress="hidden")

iface.launch(ssr_mode=False, head=spatchat_ui.HEAD)
