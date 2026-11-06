# app.py
import os
import gradio as gr
import spatchat_ui

from welcome import WELCOME_MESSAGE
from raster_io import preview_raster, clear_raster, notify_upload, no_raster_fig
from handlers import analyze_raster
from data_views import class_table, raster_plots, metric_plots_from_reply

try:
    from spatchat_gate import metered_chat
except ImportError:  # launched without spatchat-gateway's client/ on PYTHONPATH
    def metered_chat(**_kwargs):
        return lambda fn: fn


def _chat_denied(markdown, file, user_msg, history):
    """Shown instead of an answer when the gateway refuses the message (limit reached, session expired)."""
    hist = list(history or []) + [
        {"role": "user", "content": str(user_msg or "")},
        {"role": "assistant", "content": markdown},
    ]
    return hist, ""


def _refresh_views(file):
    """Datasheet + Plots tabs for the current raster (empty states when nothing is loaded)."""
    if not file:
        sheet, sheet_empty = spatchat_ui.datasheet_updates(gr, None, _EMPTY_SHEET)
        gal, gal_empty = spatchat_ui.gallery_updates(gr, [], _EMPTY_PLOTS)
        return sheet, sheet_empty, gal, gal_empty, []
    try:
        paths = raster_plots(file)
        sheet, sheet_empty = spatchat_ui.datasheet_updates(gr, class_table(file), _EMPTY_SHEET)
        gal, gal_empty = spatchat_ui.gallery_updates(gr, paths, _EMPTY_PLOTS)
    except Exception as exc:  # unreadable raster: say so instead of showing stale views
        msg = spatchat_ui.empty_state_html("table", "Could not summarise this raster", str(exc)[:200])
        sheet, sheet_empty = gr.update(value=None), gr.update(value=msg)
        gal, gal_empty = gr.update(value=None), gr.update(value=msg)
        paths = []
    return sheet, sheet_empty, gal, gal_empty, paths


def _plot_last_result(history, paths, file):
    """Chart the table in the newest assistant reply (class- or patch-level metrics) into the Plots tab."""
    paths = list(paths or [])
    last = next((m for m in reversed(history or []) if m.get("role") == "assistant"), None)
    if not file or not last:
        return gr.skip(), gr.skip(), paths
    new = metric_plots_from_reply(str(last.get("content") or ""))
    if not new:
        return gr.skip(), gr.skip(), paths
    paths = new + [p for p in paths if p not in new]
    gal, gal_empty = spatchat_ui.gallery_updates(gr, paths, _EMPTY_PLOTS)
    return gal, gal_empty, paths


_EMPTY_SHEET = spatchat_ui.empty_state_html(
    "table", "No raster loaded",
    "Click + in the chat, drag a GeoTIFF onto this panel, or choose &ldquo;Try sample data&rdquo;. "
    "A table of land-cover classes will appear here.",
)
_EMPTY_PLOTS = spatchat_ui.empty_state_html(
    "plot", "No plots yet",
    "Load a land-cover raster and the class composition and patch-size charts will appear here.",
)

# --- UI setup & launch ---
initial_history = [{"role": "assistant", "content": WELCOME_MESSAGE}]

_SAMPLE_RASTER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dummy_lulc.tif")

_SUGGESTION_CHIPS = [
    "PLAND for all classes",
    "SHDI at the landscape level",
    "Largest patch index (LPI)",
    "Edge density (ED)",
]

with gr.Blocks(title="Spatchat: Landscape Metrics", head=spatchat_ui.HEAD) as iface:
    spatchat_ui.build_header(
        gr,
        "Landscape Metrics",
        "Upload a land-cover GeoTIFF and ask for landscape metrics in plain language.",
        "landmetrics",
    )

    with gr.Row(elem_id="spatchat-workarea"):
        with gr.Column(scale=3, min_width=0, elem_id="spatchat-sidebar"):
            chatbot = gr.Chatbot(**spatchat_ui.chatbot_kwargs(gr, WELCOME_MESSAGE))
            sample_btn = spatchat_ui.chip_row_with_sample(gr, _SUGGESTION_CHIPS)
            # "+" uploads the GeoTIFF; it behaves like the old upload box.
            file_input, txt_in, send_btn = spatchat_ui.composer(
                gr, "e.g. calculate edge density", [".tif", ".tiff"]
            )
        with gr.Column(scale=5, min_width=0, elem_id="spatchat-mapcol"):
            with gr.Tabs(elem_id="spatchat-tabs"):
                with gr.Tab("Raster", id="raster"):
                    raster_out = gr.Plot(value=no_raster_fig(), label="Raster Preview")
                    clear_btn = gr.Button("Clear Raster")
                sheet, sheet_empty = spatchat_ui.datasheet_tab(gr, "Datasheet", _EMPTY_SHEET)
                gallery, gallery_empty = spatchat_ui.plots_tab(gr, "Plots", _EMPTY_PLOTS)
            gallery_state = gr.State([])

    spatchat_ui.build_footer(gr, "Wan, H.Y. &amp; Hysen, L. (2025). <i>Spatchat: Landscape Metrics.</i>")

    iface.queue(max_size=16)

    file_input.upload(preview_raster, inputs=file_input, outputs=raster_out).then(
        _refresh_views, inputs=file_input, outputs=[sheet, sheet_empty, gallery, gallery_empty, gallery_state])
    file_input.upload(notify_upload, inputs=[file_input, chatbot],
                      outputs=[chatbot, txt_in])
    sample_btn.click(lambda: (_SAMPLE_RASTER, preview_raster(_SAMPLE_RASTER)),
                     inputs=None, outputs=[file_input, raster_out]
    ).then(notify_upload, inputs=[file_input, chatbot], outputs=[chatbot, txt_in]
    ).then(_refresh_views, inputs=file_input, outputs=[sheet, sheet_empty, gallery, gallery_empty, gallery_state])
    clear_btn.click(clear_raster, inputs=None,
                    outputs=[file_input, raster_out]).then(
        _refresh_views, inputs=file_input, outputs=[sheet, sheet_empty, gallery, gallery_empty, gallery_state])
    gr.on(triggers=[txt_in.submit, send_btn.click],
          fn=metered_chat(room="landmetrics", message_index=1, deny=_chat_denied)(analyze_raster),
          inputs=[file_input, txt_in, chatbot],
          outputs=[chatbot, txt_in]).then(
        _plot_last_result, inputs=[chatbot, gallery_state, file_input],
        outputs=[gallery, gallery_empty, gallery_state])

iface.launch(ssr_mode=False)
