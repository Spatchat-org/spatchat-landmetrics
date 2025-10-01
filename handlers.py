# handlers.py
import re, json
from llm_router import llm
from parser import SYSTEM_PROMPT, parse_request_rule_based, _looks_like_hallucinated_table
from metrics_catalog import cross_level, landscape_only, class_only, patch_only, metric_map, patch_metric_map
from compute import (
    run_metadata, run_count_classes, run_list_metrics, run_compute_metrics,
    compute_landscape_only_text, compute_class_only_text, compute_patch_only_text,
    compute_multiple_metrics_text
)

def analyze_raster(file, user_msg, history):
    clear = ""  # clear the textbox after submit
    hist = history + [{"role":"user","content":user_msg}]
    lower = user_msg.lower()

    if file is None:
        hist.append({"role":"assistant","content":
                     "Please upload a GeoTIFF before asking anything."})
        return hist, clear

    # shortcuts
    if re.search(r"\ball landscape(?: level)? metrics\b", lower):
        keys = sorted((cross_level | landscape_only))
        reply = compute_landscape_only_text(file, keys)
        hist.append({"role":"assistant","content":reply})
        return hist, clear

    if re.search(r"\ball class(?: level)? metrics\b", lower):
        keys = sorted(class_only | cross_level)
        reply = compute_class_only_text(file, keys)
        hist.append({"role":"assistant","content":reply})
        return hist, clear

    if re.search(r"\ball patch(?: level)? metrics\b", lower):
        keys = sorted(patch_only)
        reply = compute_patch_only_text(file, keys)
        hist.append({"role":"assistant","content":reply})
        return hist, clear

    if re.search(r"\ball metrics\b", lower):
        keys = sorted(set(metric_map.keys()) | set(patch_metric_map.keys()))
        reply = compute_multiple_metrics_text(file, keys)
        hist.append({"role":"assistant","content":reply})
        return hist, clear

    # 1) Deterministic parser (no LLM)
    lvl_rb, mets_rb = parse_request_rule_based(user_msg)
    if lvl_rb is not None and mets_rb:
        reply = run_compute_metrics(file, mets_rb, lvl_rb)
        if _looks_like_hallucinated_table(reply):
            reply = "Let’s specify the metric explicitly (e.g., `class: pland`)."
        hist.append({"role":"assistant","content":reply})
        return hist, clear

    # 2) LLM router
    resp = llm.chat(
        messages=[{"role":"system","content":SYSTEM_PROMPT}] + hist,
        temperature=0.0,
        max_tokens=128,
        stream=False
    )

    try:
        call = json.loads(resp)
        if not isinstance(call, dict):
            raise ValueError("non-dict")
        tool = call.get("tool")
        if tool not in {"metadata","count_classes","list_metrics","compute_metrics"}:
            raise ValueError("bad tool")
    except Exception:
        hist.append({"role":"assistant","content":
                     "Which metric and level would you like? For example: `patch: area`, `class: pland`, or `landscape: edge density`."})
        return hist, clear

    if tool == "metadata":
        reply = run_metadata(file)
    elif tool == "count_classes":
        reply = run_count_classes(file)
    elif tool == "list_metrics":
        reply = run_list_metrics(file)
    elif tool == "compute_metrics":
        raw_metrics = call.get("metrics", [])
        level       = call.get("level", "both")
        mapped, unknowns = [], []
        from metrics_catalog import metric_definitions, reverse_synonyms
        for m in (raw_metrics or []):
            ml = str(m).lower()
            if ml in metric_definitions:
                mapped.append(ml)
            elif ml in reverse_synonyms:
                mapped.append(reverse_synonyms[ml])
            else:
                unknowns.append(m)
        if unknowns or not mapped:
            reply = "I didn’t recognize those metrics. Try one of: area (patch), pland (class), edge density (landscape)."
        else:
            reply = run_compute_metrics(file, mapped, level)
    else:
        reply = "I didn’t catch which metric you want. Try: `class: pland`."

    if _looks_like_hallucinated_table(reply):
        reply = "Let’s specify the metric explicitly (e.g., `class: pland`)."

    hist.append({"role":"assistant","content":reply})
    return hist, clear
