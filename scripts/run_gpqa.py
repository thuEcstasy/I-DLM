#!/usr/bin/env python3
"""
Run GPQA Diamond inference against a single locally-running sglang server.

Usage:
  python run_gpqa.py                        # all GPQA Diamond problems
  python run_gpqa.py --num-problems 10      # quick smoke test
  python run_gpqa.py --concurrency 16       # tune in-flight requests
  python run_gpqa.py --output-dir out/gpqa  # save details + summary
"""

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from string import ascii_uppercase

import requests
from inspect_ai.dataset import Sample
from inspect_ai.solver import multiple_choice
from inspect_ai.scorer import choice

# Solver/scorer 全局初始化
solver = multiple_choice(cache=True)
scorer = choice()

GPQA_HF_DATASET = "Idavidrein/gpqa"

# --- 构造完整选择题 prompt
def construct_mc_prompt(record):
    # 固定正确答案位置为 B（索引=1），保证 reference 一致
    gold_index = 1
    choices = [
        record["Incorrect Answer 1"],
        record["Incorrect Answer 2"],
        record["Incorrect Answer 3"],
    ]
    choices.insert(gold_index, record["Correct Answer"])
    prompt_template = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()
    prompt = prompt_template.format(
        Question=record["Pre-Revision Question"],
        A=choices[0],
        B=choices[1],
        C=choices[2],
        D=choices[3],
    )
    return prompt, gold_index, choices

# # --- 官方 Lighteval GPQA Diamond 验证流程
# def verify_gpqa_diamond(prediction: str, gold_index: int, choices: list):
#     pred_clean = prediction.strip().split("Answer:")[-1].strip()  # "B"
#     letter_to_index = {"A":0,"B":1,"C":2,"D":3}
#     pred_idx = letter_to_index.get(pred_clean, None)
#     if pred_idx is not None:
#         pred_text = choices[pred_idx]  # 映射到选项文本
#     else:
#         pred_text = pred_clean

#     sample = Sample(input="", choices=choices, target=choices[gold_index])
#     predicted_choice = solver.solve(sample, pred_text)
#     ok = scorer.score(sample, predicted_choice)
#     return ok, None

import re

def verify_gpqa_diamond(prediction: str, golden_letter: str):
    """
    直接取 prediction 中最后一个大写字母 A-D 作为模型答案，
    与 golden_letter 比较
    """
    # 匹配所有大写 A-D
    matches = re.findall(r"[A-D]", prediction)
    if matches:
        pred_letter = matches[-1]  # 取最后一个
    else:
        pred_letter = ""  # 没找到
    ok = pred_letter.upper() == golden_letter.upper()
    return ok, None

# --- 调用本地模型 server
def call_server(idx, record, host, port, max_tokens, temperature, top_p, top_k, timeout):
    prompt, gold_index, choices = construct_mc_prompt(record)
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": "sdar",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }
    try:
        r = requests.post(url, json=payload, timeout=timeout).json()
        pred = r["choices"][0]["message"]["content"]
        comp = r.get("usage", {}).get("completion_tokens", 0)
        rid = r.get("id")
        return idx, pred, comp, rid, None, gold_index, prompt, choices
    except Exception as e:
        return idx, "", 0, None, str(e), gold_index, prompt, choices

# --- 加载 GPQA Diamond 问题
def load_problems(num):
    from datasets import load_dataset
    ds = load_dataset(GPQA_HF_DATASET, 'gpqa_diamond', split="train")
    problems = [
        {
            "Pre-Revision Question": it["Pre-Revision Question"],
            "Correct Answer": it["Correct Answer"] if "Correct Answer" in it else it["Pre-Revision Correct Answer"],
            "Incorrect Answer 1": it["Incorrect Answer 1"],
            "Incorrect Answer 2": it["Incorrect Answer 2"],
            "Incorrect Answer 3": it["Incorrect Answer 3"],
        }
        for it in ds
    ]
    if num:
        problems = problems[: min(num, len(problems))]
    return problems

def lookup_stats(stats_file, rid, timeout=5.0):
    if not stats_file or not rid:
        return None
    import time as _t
    deadline = _t.time() + timeout
    while _t.time() < deadline:
        if os.path.exists(stats_file):
            try:
                with open(stats_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            d = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if d.get("rid") == rid:
                            return d
            except OSError:
                pass
        _t.sleep(0.05)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--num-problems", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--stats-file", default="/tmp/idlm_stats.jsonl")
    ap.add_argument("--rounds-file", default="/tmp/idlm_rounds.jsonl")
    args = ap.parse_args()

    # truncate stats/rounds
    if args.stats_file:
        with open(args.stats_file, "w"):
            pass
    if args.rounds_file:
        with open(args.rounds_file, "w"):
            pass

    problems = load_problems(args.num_problems)
    N = len(problems)
    print(f"GPQA Diamond: {N} problems → http://{args.host}:{args.port}, concurrency={args.concurrency}")

    t0 = time.time()
    results = [None] * N
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                call_server,
                i,
                record,
                args.host,
                args.port,
                args.max_tokens,
                args.temperature,
                args.top_p,
                args.top_k,
                args.timeout,
            ): i
            for i, record in enumerate(problems)
        }
        done = 0
        for fut in as_completed(futures):
            idx, pred, comp, rid, err, gold_index, prompt, choices = fut.result()
            golden_letter = ascii_uppercase[gold_index]  # 例如 "B"
            ok, verr = verify_gpqa_diamond(pred, golden_letter)
            stats = lookup_stats(args.stats_file, rid) if args.stats_file else None
            results[idx] = (pred, comp, err, ok, verr, stats, gold_index, prompt, choices)
            done += 1

            ok_tag = "?" if ok is None else ("✓" if ok else "✗")
            sep = "─" * 78
            stats_str = ""
            if stats:
                stats_str = (
                    f"  acc={stats['avg_accept_len']:.2f}/{stats['verify_num_specs']}"
                    f"  tok/fwd={stats['tok_per_fwd']:.2f}"
                    f"  fwds={stats['forwards']}"
                )
            elif args.stats_file:
                stats_str = "  acc=?(no stats line)"
            print(f"\n{sep}")
            print(f"[{done}/{N}] idx={idx}  {ok_tag}  tokens={comp}  t={time.time()-t0:.0f}s{stats_str}")
            print(f"PROMPT:\n{'…' if len(prompt)>500 else ''}{prompt[-500:]}")
            print(f"PREDICTION:\n{'…' if len(pred)>500 else ''}{pred[-500:]}")
            print(f"REFERENCE: {ascii_uppercase[gold_index]}")

    # 汇总统计
    total_tok = 0
    correct = 0
    errors = 0
    verifier_available = True
    details = []
    for i, (record, (pred, comp, err, ok, verr, stats, gold_index, prompt, choices)) in enumerate(zip(problems, results)):
        total_tok += comp
        if err:
            errors += 1
        if ok is None:
            verifier_available = False
            ok = False
        elif ok:
            correct += 1
        details.append(
            {
                "idx": i,
                "problem": record["Pre-Revision Question"],
                "reference": ascii_uppercase[gold_index],
                "prediction": pred,
                "completion_tokens": comp,
                "correct": ok,
                "error": err,
                "verify_error": verr,
                "stats": stats,
            }
        )

    elapsed = time.time() - t0
    print("=" * 60)
    if verifier_available:
        print(f"Accuracy:    {correct / N * 100:.1f}%  ({correct}/{N})")
    else:
        print("Accuracy:    [verifier not available — skipped]")
    print(f"Total tokens: {total_tok:,}")
    print(f"Wall time:    {elapsed:.1f}s")
    print(f"Throughput:   {total_tok / elapsed:.1f} tok/s" if elapsed > 0 else "")
    print(f"Errors:       {errors}")

    # mean acc len / mean tok/fwd
    valid_stats = [d["stats"] for d in details if d.get("stats")]
    if valid_stats:
        mean_acc = sum(s["avg_accept_len"] for s in valid_stats) / len(valid_stats)
        mean_tpf = sum(s["tok_per_fwd"] for s in valid_stats) / len(valid_stats)
        denom = valid_stats[0]["verify_num_specs"]
        print(f"Mean acc len: {mean_acc:.2f}/{denom}  (across {len(valid_stats)}/{N} reqs)")
        print(f"Mean tok/fwd: {mean_tpf:.2f}")

    # pre_conf -> accept_len table
    bin_summary = None
    if args.rounds_file and os.path.exists(args.rounds_file):
        from collections import defaultdict
        bins = defaultdict(list)
        n_total = 0
        denom_r = 0
        with open(args.rounds_file) as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                vns_r = r.get("verify_num_specs")
                if vns_r and vns_r > denom_r:
                    denom_r = vns_r
                if r.get("pre_conf") is None or r.get("accept_len") is None:
                    continue
                b = round(int(r["pre_conf"] * 10) / 10, 1)
                bins[b].append(r["accept_len"])
                n_total += 1
        if bins:
            print()
            print(f"Pre-forward conf → accept_len  (n={n_total} rounds, V+R only)")
            print(f"{'conf_bin':>10}  {'n':>7}  {'mean_acc':>9}  {'std':>6}  bar")
            bin_summary = []
            for b in sorted(bins):
                xs = bins[b]
                m = sum(xs) / len(xs)
                if len(xs) > 1:
                    var = sum((x - m) ** 2 for x in xs) / len(xs)
                    sd = var ** 0.5
                else:
                    sd = 0.0
                bar = "█" * int(m / max(denom_r, 1) * 30) if denom_r else ""
                print(f"{b:>10.1f}  {len(xs):>7}  {m:>9.2f}  {sd:>6.2f}  {bar}")
                bin_summary.append({"conf_bin": b, "n": len(xs), "mean_acc": m, "std": sd})

    # 保存 JSON
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        tag = f"_{args.tag}" if args.tag else ""
        summary = {
            "score": (correct / N * 100) if verifier_available else None,
            "correct": correct if verifier_available else None,
            "total": N,
            "errors": errors,
            "total_completion_tokens": total_tok,
            "wall_time_s": elapsed,
            "config": vars(args),
            "bin_summary": bin_summary,
        }
        with open(os.path.join(args.output_dir, f"gpqa_summary{tag}.json"), "w") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(args.output_dir, f"gpqa_details{tag}.json"), "w") as f:
            json.dump(details, f, indent=2, ensure_ascii=False)
        if args.rounds_file and os.path.exists(args.rounds_file):
            import shutil
            shutil.copy(args.rounds_file,
                        os.path.join(args.output_dir, f"gpqa_rounds{tag}.jsonl"))
        print(f"Saved to {args.output_dir}/")


if __name__ == "__main__":
    main()