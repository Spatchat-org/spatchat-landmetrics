---
title: Spatchat
emoji: 🌲
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "5.25.2"
app_file: app.py
pinned: false
license: mit
---

# 🌲 Spatchat: Conversational Spatial Intelligence

**Spatchat** is an open-source platform for AI-powered spatial modeling and analysis.  
It enables natural language interaction with complex geospatial tools — from landscape metrics to species distribution models and beyond.

## 🚀 What is Spatchat?

Spatchat is a modular, extensible framework that allows researchers, educators, and developers to build chat-driven spatial analysis tools. It represents a paradigm shift in spatial modeling: from code-heavy pipelines to seamless, conversational interfaces powered by LLMs.

This Space hosts the **Spatchat Landscape Metrics Assistant** — the first module in a growing ecosystem.

## 💡 Key Features

- 🔍 Natural language querying of spatial metrics
- 🧠 LLM-powered intent parsing and metric reasoning
- 🗺️ Real-time landscape analysis from raster input
- 🧩 Modular architecture for future tools
- ☁️ Cloud-native, no local setup required

## ✨ Try It

1. Upload a GeoTIFF raster
2. Ask questions like:
   - “What is edge density?”
   - “Compare patch density between class 1 and 3”
   - “Define Shannon evenness index”

## 🛠 Architecture

Each Spatchat module consists of:
- A lightweight chat UI (Gradio)
- An analysis engine (e.g., PyLandStats, Circuitscape)
- An LLM dispatcher (e.g., Together, OpenAI, DeepSeek)
- Optional memory or session context

## 🤝 Call for Collaborators

We're building more than a chatbot — we're building an ecosystem.

If you’re developing:
- Species distribution models (SDMs)
- Connectivity modeling workflows
- Resistance mapping tools
- Remote sensing pipelines

…and want to integrate your model into a friendly chat interface, we’d love to collaborate!

📬 Contact: [Ho Yi Wan](mailto:hoyiwan@gmail.com)

## 📜 License

This project is released under the [MIT License](LICENSE), but other modules may use different licenses.

---

Spatchat is built on the belief that spatial insight should be a conversation — not a configuration file.
>>>>>>> 0773e6a (Deploy Spatchat platform)
