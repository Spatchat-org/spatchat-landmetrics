# app.py
import gradio as gr
import spatchat_ui

from welcome import WELCOME_MESSAGE
from raster_io import preview_raster, clear_raster, notify_upload, no_raster_fig
from handlers import analyze_raster

# --- UI setup & launch ---
initial_history = [{"role": "assistant", "content": WELCOME_MESSAGE}]

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
            spatchat_ui.chip_row(gr, _SUGGESTION_CHIPS)
            # "+" uploads the GeoTIFF; it behaves like the old upload box.
            file_input, txt_in, send_btn = spatchat_ui.composer(
                gr, "e.g. calculate edge density", [".tif", ".tiff"]
            )
        with gr.Column(scale=5, min_width=0, elem_id="spatchat-mapcol"):
            with gr.Tabs(elem_id="spatchat-tabs"):
                with gr.Tab("Raster", id="raster"):
                    raster_out = gr.Plot(label="Raster Preview")
                    clear_btn = gr.Button("Clear Raster")

    spatchat_ui.build_footer(gr, "Wan, H.Y. &amp; Hysen, L. (2025). <i>Spatchat: Landscape Metrics.</i>")

    iface.queue(max_size=16)

    file_input.upload(preview_raster, inputs=file_input, outputs=raster_out)
    file_input.upload(notify_upload, inputs=[file_input, chatbot],
                      outputs=[chatbot, txt_in])
    clear_btn.click(clear_raster, inputs=None,
                    outputs=[file_input, raster_out])
    gr.on(triggers=[txt_in.submit, send_btn.click], fn=analyze_raster,
          inputs=[file_input, txt_in, chatbot],
          outputs=[chatbot, txt_in])

iface.launch(ssr_mode=False)
