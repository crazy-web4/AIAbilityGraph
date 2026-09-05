"""
推理能力蒸馏：用大推理模型生成数据，SFT 小模型

流程（教学骨架，离线可跑通数据构造逻辑）：
  1. 准备"可验证"题目（数学/代码：对错可机械判定）
  2. 让推理模型生成 思维链 + 答案
  3. 用验证器过滤，只保留答案正确的轨迹
  4. 导出为 messages 格式 SFT 数据集（后续可用 QLoRA 微调小模型）

真实环境：
  - 把 MockReasoner 换成 special-07/reasoning_api.py 的真实客户端
  - 导出的 jsonl 可用 LLaMA-Factory / trl SFTTrainer + QLoRA 训练
运行：
  python distill_reasoning.py        # 离线 Mock，产出 /tmp/distill_sft.jsonl
"""

from __future__ import annotations

import json
import os
import random

OUTPUT = "/tmp/distill_sft.jsonl"


# ---------------------------------------------------------------------------
# 可验证题目（这里用简单算术演示；真实场景换数学题库/编程题+单测）
# ---------------------------------------------------------------------------

PROBLEMS = [
    {"question": "计算 23 * 17 = ?", "gold": 391},
    {"question": "计算 144 / 12 = ?", "gold": 12},
    {"question": "计算 7! = ?", "gold": 5040},
    {"question": "计算 2^10 = ?", "gold": 1024},
]


class MockReasoner:
    """模拟推理模型：大概率答对，偶尔算错（用于演示验证过滤）。"""

    def generate(self, q: str) -> tuple[str, str, int]:
        gold = next(p["gold"] for p in PROBLEMS if p["question"] == q)
        if random.random() < 0.75:           # 75% 答对
            answer, n = gold, gold
        else:                                # 答错的轨迹将被丢弃
            answer, n = gold + random.choice([-1, 2]), gold + 1
        reasoning = (
            f"<think>题目：{q}\n"
            f"我先列出运算步骤……\n"
            f"等等，检查一下进位/量级……\n"
            f"验算结果与 gold 一致。</think>"
        )
        return reasoning, str(answer), n


def verifier(pred_answer: str, gold) -> bool:
    """规则验证器：答案可机械判定（RLVR 的核心）。"""
    try:
        return int("".join(ch for ch in pred_answer if ch.isdigit()) or -999) == gold
    except ValueError:
        return False


def build_dataset(samples_per_problem: int = 4) -> list[dict]:
    reasoner = MockReasoner()
    dataset, kept, dropped = [], 0, 0
    for prob in PROBLEMS:
        for _ in range(samples_per_problem):
            reasoning, answer, _ = reasoner.generate(prob["question"])
            if not verifier(answer, prob["gold"]):
                dropped += 1
                continue                        # 只保留正确轨迹
            kept += 1
            dataset.append({
                "messages": [
                    {"role": "user", "content": prob["question"]},
                    {"role": "assistant",
                     "content": f"{reasoning}\n答案：{answer}"},
                ]
            })
    print(f"生成完成：保留 {kept} 条正确轨迹，丢弃 {dropped} 条错误轨迹")
    return dataset


def main() -> None:
    random.seed(0)
    dataset = build_dataset()
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"已导出 SFT 数据集：{OUTPUT}（{len(dataset)} 条）")
    print("下一步：用 LLaMA-Factory / trl + QLoRA 微调 7B/14B 小模型，"
          "再在评测集上验证。")


if __name__ == "__main__":
    main()
