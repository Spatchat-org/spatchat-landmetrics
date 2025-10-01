# parser.py
import re, json
from metrics_catalog import (
    metric_definitions, synonyms, reverse_synonyms,
    metric_map, patch_metric_map
)

SYSTEM_PROMPT = """
You are a tool router. Output ONLY a single JSON object with this exact schema:
{"tool":"metadata"|"count_classes"|"list_metrics"|"compute_metrics",
 "level":"landscape"|"class"|"patch"|"both",
 "metrics":[<lowercase metric codes>]}
Rules:
- NEVER include any other text, Markdown, or numbers.
- NEVER estimate or compute values.
- If unsure, choose {"tool":"compute_metrics","level":"both","metrics":[]}.
""".strip()

LEVEL_HINTS = {
    "class": re.compile(r"\bclass\b|\bby class\b|\bclass[-\s]*level\b"),
    "landscape": re.compile(r"\blandscape\b|\boverall\b|\bwhole map\b|\blandscape[-\s]*level\b"),
    "patch": re.compile(r"\bpatch\b|\bpatch[-\s]*level\b")
}

def parse_request_rule_based(user_msg: str):
    lower = user_msg.lower()

    want_class = bool(LEVEL_HINTS["class"].search(lower))
    want_land  = bool(LEVEL_HINTS["landscape"].search(lower))
    want_patch = bool(LEVEL_HINTS["patch"].search(lower))

    if sum([want_class, want_land, want_patch]) > 1:
        level = "both"
    elif want_patch:
        level = "patch"
    elif want_class:
        level = "class"
    elif want_land:
        level = "landscape"
    else:
        level = "both"

    found = set()
    text = f" {lower} "

    # phrase matches (synonyms may contain spaces)
    for canon, syns in synonyms.items():
        for s in sorted(syns, key=len, reverse=True):
            if f" {s} " in text:
                found.add(canon); break

    # codes inside parentheses, e.g. "(pland)"
    for par in re.findall(r"\(([a-zA-Z_]+)\)", lower):
        if par in metric_definitions:
            found.add(par)
        elif par in reverse_synonyms:
            found.add(reverse_synonyms[par])

    # single tokens (allow (), |)
    for tok in re.findall(r"[a-zA-Z_%()|]+", lower):
        if tok in metric_definitions:
            found.add(tok)
        elif tok in reverse_synonyms:
            found.add(reverse_synonyms[tok])

    mapped = []
    for m in found:
        if m == "core_p" or (m == "core" and level == "patch"):
            mapped.append("core_p")
        elif m == "core":
            mapped.append("tca")
        else:
            mapped.append(m)

    mapped = [m for m in mapped if (m in metric_map or m in patch_metric_map)]
    return (level if mapped else None, mapped)

def _looks_like_hallucinated_table(txt: str) -> bool:
    suspicious = ["area_km2", "area percent", "percent landscape"]
    return any(s in txt.lower() for s in suspicious) and ("| class" in txt.lower() or "\t" in txt)
