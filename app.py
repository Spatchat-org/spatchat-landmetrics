# app.py
import gradio as gr

from welcome import WELCOME_MESSAGE
from raster_io import preview_raster, clear_raster, notify_upload, no_raster_fig
from handlers import analyze_raster

# --- UI setup & launch ---
initial_history = [{"role": "assistant", "content": WELCOME_MESSAGE}]

with gr.Blocks(title="Spatchat") as iface:
    gr.HTML('<head><link rel="icon" href="logo1.png"></head>')
    gr.Image(value="logo_long1.png", type="filepath",
             show_label=False, show_download_button=False,
             show_share_button=False, elem_id="logo-img")
    gr.HTML("<style>#logo-img img{height:90px;margin:10px;border-radius:6px;}</style>")
    gr.Markdown("## 🌲 Spatchat: Landscape Metrics Assistant {landmetrics}")
    gr.HTML("""
      <div style="margin-top: -10px; margin-bottom: 15px;">
        <input type="text" value="https://spatchat.org/browse/?room=landmetrics" id="shareLink" readonly style="width: 50%; padding: 5px; background-color: #f8f8f8; color: #222; font-weight: 500; border: 1px solid #ccc; border-radius: 4px;">
        <button onclick="navigator.clipboard.writeText(document.getElementById('shareLink').value)" style="padding: 5px 10px; background-color: #007BFF; color: white; border: none; border-radius: 4px; cursor: pointer;">
          📋 Copy Share Link
        </button>
        <div style="margin-top: 10px; font-size: 14px;">
          <b>Share:</b>
          <a href="https://twitter.com/intent/tweet?text=Checkout+Spatchat!&url=https://spatchat.org/browse/?room=landmetrics" target="_blank">🐦 Twitter</a> |
          <a href="https://www.facebook.com/sharer/sharer.php?u=https://spatchat.org/browse/?room=landmetrics" target="_blank">📘 Facebook</a>
        </div>
      </div>
    """)
    gr.HTML("""
      <div style="font-size: 14px;">
      © 2025 Ho Yi Wan & Logan Hysen. All rights reserved.<br>
      If you use Spatchat in research, please cite:<br>
      <b>Wan, H.Y.</b> & <b>Hysen, L.</b> (2025). <i>Spatchat: Landscape Metrics.</i>
      </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload GeoTIFF", type="filepath",
                                  file_types=[".tif", ".tiff"])
            raster_out = gr.Plot(label="Raster Preview")
            clear_btn = gr.Button("Clear Raster")
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(value=initial_history, type="messages",
                                 label="Spatchat Dialog")
            txt_in = gr.Textbox(placeholder="e.g. calculate edge density",
                                lines=1)

    iface.queue(max_size=16)

    file_input.change(preview_raster, inputs=file_input, outputs=raster_out)
    file_input.change(notify_upload, inputs=[file_input, chatbot],
                      outputs=[chatbot, txt_in])
    clear_btn.click(clear_raster, inputs=None,
                    outputs=[file_input, raster_out])
    txt_in.submit(analyze_raster,
                  inputs=[file_input, txt_in, chatbot],
                  outputs=[chatbot, txt_in])

iface.launch(ssr_mode=False)
