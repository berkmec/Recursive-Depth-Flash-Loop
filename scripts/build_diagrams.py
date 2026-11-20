"""Generate and XML-validate publication-grade SVG diagrams from audited JSON artifacts."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagram_architecture import build_architecture_svg
from scripts.diagram_benchmarks import build_benchmarks_1_2m_svg, build_self_capabilities_svg

OUT_DIR = ROOT / "docs" / "diagrams"
SUMMARY_PATH = ROOT / "results" / "audited_summary.json"
GPU_PROFILE_PATH = ROOT / "results" / "raw" / "gpu_profil_55m.json"


def main() -> int:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    gpu_profile = json.loads(GPU_PROFILE_PATH.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        "rdfl_architecture.svg": build_architecture_svg(),
        "benchmarks_1_2m.svg": build_benchmarks_1_2m_svg(summary, gpu_profile),
        "self_capabilities_v2.svg": build_self_capabilities_svg(summary),
    }
    for name, content in files.items():
        ET.fromstring(content)
        path = OUT_DIR / name
        path.write_text(content, encoding="utf-8")
        print(f"Wrote and XML-validated {path} ({len(content)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
