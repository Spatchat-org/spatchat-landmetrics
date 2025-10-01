# welcome.py
WELCOME_MESSAGE = """\
👋 Hi! I’m Spatchat. I am an expert in quantifying landscape metrics.

This is a list of metrics I can calculate:

Patch level
- Area/Edge: AREA, PERIM
- Shape: PARA, SHAPE, FRAC
- Core: CORE, NCORE, CAI
- Aggregation: ENN

Class level
- Area/Edge: TA, PLAND, NP, PD, LPI, TE, ED
- Core: TCA, CPLAND, NDCA, DCAD
- Aggregation/Connectivity: LSI, MESH

Landscape level
- Area/Edge summaries: TA, NP, PD, LPI, TE, ED, LSI
- Core summaries: TCA, CPLAND, NDCA, DCAD
- Richness/Diversity: H (entropy), SHDI
- Contagion/Interspersion: H(X,Y) (joint entropy), H(Y|X) (conditional entropy), MI (mutual information), RMI (relative MI), CONTAG

You can ask me in natural language. For example:
- "What is the percentage of landscape (PLAND) for class 1 at the class level?"
- "Give me PLAND for all classes at the class level."
- "What is Shannon’s diversity index (SHDI) at the landscape level?"
- "Tell me the largest patch index (LPI) at the landscape level."
- "Compute patch area (AREA) at the patch level."

If I’m not sure what you mean, I’ll suggest close matches and ask you to confirm.
Upload a GeoTIFF to begin — then ask me for any landscape metric.
"""
