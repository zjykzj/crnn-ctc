# -*- coding: utf-8 -*-
"""
对 eval_plate.py 保存的结果 JSON 进行重新分析：
- 应用 clean_plate 清洗分隔符
- 对比清洗前后的准确率
- 输出修复了多少条
"""

import json
import re
import sys
from typing import Optional

# 所有非车牌字符
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
    fixed_by_clean = 0  # 清洗前错，清洗后对
    still_wrong = 0     # 清洗后仍然错
    became_wrong = 0    # 清洗前对，清洗后错（理论上不应该发生）
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

    # 分类 still_wrong 的错误类型
    separator_errors = 0      # 清洗后还有分隔符
    length_errors = 0         # 长度不匹配
    char_errors = 0           # 字符错误
    province_errors = 0       # 省份错误
    null_errors = 0           # 预测为 null

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
    print(f"文件: {json_path}")
    print(f"总样本: {total}")
    print()
    print(f"=== 准确率对比 ===")
    print(f"清洗前: {before_correct}/{total} = {before_correct/total*100:.2f}%")
    print(f"清洗后: {after_correct}/{total} = {after_correct/total*100:.2f}%")
    print(f"提升:   +{(after_correct - before_correct)/total*100:.2f}%")
    print()
    print(f"=== 变化明细 ===")
    print(f"清洗修复:      {fixed_by_clean} 条（分隔符导致错误 → 清洗后正确）")
    print(f"仍然错误:      {still_wrong} 条")
    print(f"清洗破坏:      {became_wrong} 条（不应发生）")
    print()
    if still_wrong > 0:
        print(f"=== 仍错误样本的错误类型（采样 {min(30, len(still_wrong_examples))} 条） ===")
        print(f"  分隔符残留:   {separator_errors}")
        print(f"  长度不匹配:   {length_errors}")
        print(f"  省份错误:     {province_errors}")
        print(f"  字符错误:     {char_errors}")
        print(f"  预测为 null:  {null_errors}")
    print()
    print(f"=== 清洗修复样例（前 15 条） ===")
    for path, label, raw, cleaned in fixed_examples[:15]:
        print(f"  {raw} → {cleaned}  (label: {label})  {path.split('/')[-1]}")
    print()
    print(f"=== 仍然错误样例（前 15 条） ===")
    for path, label, raw, cleaned in still_wrong_examples[:15]:
        print(f"  pred: {cleaned}  label: {label}  raw: {raw}  {path.split('/')[-1]}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
