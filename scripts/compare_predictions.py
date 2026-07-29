#!/usr/bin/env python3
"""Compare frozen self-test predictions with the pre-build baseline."""

from __future__ import annotations

import json
from pathlib import Path
import sys


TOLERANCE = 1e-6


def main() -> int:
    if len(sys.argv) != 3:
        print("用法：compare_predictions.py BASELINE.json FROZEN.json")
        return 2

    baseline = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    frozen = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    if not frozen.get("ok"):
        print(f"冻结应用自检失败：{frozen.get('error')}")
        return 1

    baseline_by_name = {
        Path(item["image"]).name: item for item in baseline["predictions"]
    }
    failed = False
    for actual in frozen["predictions"]:
        name = Path(actual["image"]).name
        expected = baseline_by_name[name]
        shape_matches = (
            actual["mask_height"] == expected["mask_height"]
            and actual["mask_width"] == expected["mask_width"]
        )
        difference = abs(actual["coverage"] - expected["coverage"])
        print(
            f"{name}: baseline={expected['coverage']:.10f}% "
            f"frozen={actual['coverage']:.10f}% diff={difference:.10g} "
            f"shape={'OK' if shape_matches else 'FAIL'}"
        )
        if not shape_matches or difference > TOLERANCE:
            failed = True

    if frozen.get("model_load_count") != 1:
        print(
            "模型加载次数错误："
            f"{frozen.get('model_load_count')}（期望 1）"
        )
        failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
