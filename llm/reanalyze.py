# -*- coding: utf-8 -*-
"""
Re-analyze a result JSON saved by eval_plate.py:
- Apply clean_plate to strip separators
- Compare accuracy before/after cleaning
- Report how many samples were fixed
- Classify remaining errors by type
"""

import json
import re
import sys
from typing import Optional

# All non-plate characters
_NON_PLATE_RE = re.compile(
    r'[^京津沪渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新'
    r'A-Za-z0-9]'
)


def clean_plate(plate: Optional[str]) -> Optional[str]:
    if plate is None:
        return None
    cleaned = _NON_PLATE_RE.sub('', plate)
    return cleaned if cleaned else None


def main():
    if len(sys.argv) < 2:
        json_path = "llm/results/qwen3.5_2b_20260716_213715.json"
    else:
        json_path = sys.argv[1]

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    total = len(results)

    before_correct = 0
    after_correct = 0
    fixed_by_clean = 0  # wrong before, right after cleaning
    still_wrong = 0     # still wrong after cleaning
    became_wrong = 0    # right before, wrong after cleaning (should not happen)
    fixed_examples = []
    still_wrong_examples = []

    for r in results:
        label = r["label"]
        raw_plate = r["pred"]["plate"]
        cleaned_plate = clean_plate(raw_plate)

        raw_correct = (raw_plate == label)
        cleaned_correct = (cleaned_plate == label)

        if raw_correct:
            before_correct += 1
        if cleaned_correct:
            after_correct += 1

        if not raw_correct and cleaned_correct:
            fixed_by_clean += 1
            if len(fixed_examples) < 30:
                fixed_examples.append((r["img_path"], label, raw_plate, cleaned_plate))
        elif not cleaned_correct:
            still_wrong += 1
            if len(still_wrong_examples) < 30:
                still_wrong_examples.append((r["img_path"], label, raw_plate, cleaned_plate))
        elif raw_correct and not cleaned_correct:
            became_wrong += 1

    # Classify error types among still-wrong samples
    separator_errors = 0      # separator still present after cleaning
    length_errors = 0         # length mismatch
    char_errors = 0           # character error
    province_errors = 0       # province mistake
    null_errors = 0           # predicted as null

    for _, label, raw, cleaned in still_wrong_examples:
        if cleaned is None:
            null_errors += 1
        elif _NON_PLATE_RE.search(str(raw)):
            separator_errors += 1
        elif len(cleaned) != len(label):
            length_errors += 1
        elif cleaned[0] != label[0]:
            province_errors += 1
        else:
            char_errors += 1

    print(f"{'='*60}")
    print(f"File: {json_path}")
    print(f"Total samples: {total}")
    print()
    print(f"=== Accuracy Comparison ===")
    print(f"Before cleaning: {before_correct}/{total} = {before_correct/total*100:.2f}%")
    print(f"After cleaning:  {after_correct}/{total} = {after_correct/total*100:.2f}%")
    print(f"Improvement:     +{(after_correct - before_correct)/total*100:.2f}%")
    print()
    print(f"=== Breakdown ===")
    print(f"Fixed by cleaning:     {fixed_by_clean} samples (separator error → correct)")
    print(f"Still wrong:           {still_wrong} samples")
    print(f"Broken by cleaning:    {became_wrong} samples (should not happen)")
    print()
    if still_wrong > 0:
        print(f"=== Still-wrong error types (sampled {min(30, len(still_wrong_examples))}) ===")
        print(f"  Separator residue:  {separator_errors}")
        print(f"  Length mismatch:    {length_errors}")
        print(f"  Province error:     {province_errors}")
        print(f"  Character error:    {char_errors}")
        print(f"  Predicted null:     {null_errors}")
    print()
    print(f"=== Fixed by cleaning (first 15) ===")
    for path, label, raw, cleaned in fixed_examples[:15]:
        print(f"  {raw} → {cleaned}  (label: {label})  {path.split('/')[-1]}")
    print()
    print(f"=== Still wrong (first 15) ===")
    for path, label, raw, cleaned in still_wrong_examples[:15]:
        print(f"  pred: {cleaned}  label: {label}  raw: {raw}  {path.split('/')[-1]}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
