"""
Run MATH-500 inference against a single locally-running sglang server.

Assumes a server is already running (e.g. via scripts/launch_server.sh on port 30000).

Usage:
  python scripts/run_math500.py                       # all 500 problems
  python scripts/run_math500.py --num-problems 10     # quick smoke test
  python scripts/run_math500.py --concurrency 16      # tune in-flight requests
  python scripts/run_math500.py --output-dir out/m500 # save details + summary
"""
import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


MATH_HF_DATASET = "HuggingFaceH4/MATH-500"


def strip_thinking(text):
    if not text:
        return ""
    return re.sub(r"^.*</think>\s*", "", text, count=1, flags=re.DOTALL)


def try_verify(prediction, reference):
    """Return (verified_bool, error_str_or_None). Falls back to None if math_verify missing."""
    try:
        from math_verify import (
            ExprExtractionConfig,
            LatexExtractionConfig,
            parse,
            verify,
        )
    except ImportError:
        return None, "math_verify not installed"

    prediction = strip_thinking(prediction)
    gold = parse(
        f"${reference}$",
        extraction_mode="first_match",
        extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
    )
    if not gold:
        return False, "gold parse failed"
    pred = parse(
        prediction,
        extraction_config=[
            LatexExtractionConfig(boxed_match_priority=0),
            ExprExtractionConfig(),
        ],
    )
    if not pred:
        return False, "pred parse failed"
    try:
        return bool(verify(gold, pred)), None
    except Exception as e:
        return False, f"verify error: {e}"


def call_server(idx, problem, host, port, max_tokens, temperature, top_p, top_k, timeout):
    prompt = (
        f"{problem}\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )
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
        return idx, pred, comp, rid, None
    except Exception as e:
        return idx, "", 0, None, str(e)


def lookup_stats(stats_file, rid, timeout=5.0):
    """Poll stats JSONL for the line matching rid. Returns dict or None."""
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


def load_problems(num):
    from datasets import load_dataset

    ds = load_dataset(MATH_HF_DATASET, split="test")
    problems = [(it["problem"], it["answer"]) for it in ds]
    if num:
        problems = problems[: min(num, len(problems))]
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--num-problems", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument(
        "--stats-file",
        default="/tmp/idlm_stats.jsonl",
        help="Path to the per-request JSONL the server writes (algorithm "
        "stats_file / IDLM_STATS_FILE). Truncated on start. Pass empty "
        "string to disable.",
    )
    ap.add_argument(
        "--rounds-file",
        default="/tmp/idlm_rounds.jsonl",
        help="Path to the per-round JSONL the server writes (algorithm "
        "rounds_file / IDLM_ROUNDS_FILE). Truncated on start; "
        "printed as a (pre_conf bin → mean accept_len) table at the end. "
        "Pass empty string to disable.",
    )
    args = ap.parse_args()

    if args.stats_file:
        # Truncate so we don't pick up stale entries from previous runs
        with open(args.stats_file, "w"):
            pass
    if args.rounds_file:
        with open(args.rounds_file, "w"):
            pass

    problems = load_problems(args.num_problems)
    N = len(problems)
    print(
        f"MATH-500: {N} problems → http://{args.host}:{args.port}, "
        f"concurrency={args.concurrency}, max_tokens={args.max_tokens}"
    )

    t0 = time.time()
    results = [None] * N  # (pred, comp, err, ok, verr)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                call_server,
                i,
                prob,
                args.host,
                args.port,
                args.max_tokens,
                args.temperature,
                args.top_p,
                args.top_k,
                args.timeout,
            ): i
            for i, (prob, _) in enumerate(problems)
        }
        done = 0
        for fut in as_completed(futures):
            idx, pred, comp, rid, err = fut.result()
            prob_text, ref = problems[idx]
            ok, verr = try_verify(pred, ref)
            stats = lookup_stats(args.stats_file, rid) if args.stats_file else None
            results[idx] = (pred, comp, err, ok, verr, stats)
            done += 1
            ok_tag = "?" if ok is None else ("✓" if ok else "✗")
            sep = "─" * 78
            print(f"\n{sep}")
            stats_str = ""
            if stats:
                stats_str = (
                    f"  acc={stats['avg_accept_len']:.2f}/{stats['verify_num_specs']}"
                    f"  tok/fwd={stats['tok_per_fwd']:.2f}"
                    f"  fwds={stats['forwards']}"
                )
            elif args.stats_file:
                stats_str = "  acc=?(no stats line)"
            print(f"[{done}/{N}] idx={idx}  {ok_tag}  tokens={comp}  t={time.time() - t0:.0f}s{stats_str}")
            print(f"PROBLEM: {'…' if len(prob_text) > 200 else ''}{prob_text[:200]}")
            if err:
                print(f"ERROR: {err}")
            else:
                print(f"PREDICTION:\n{pred}")
            print(f"REFERENCE: {ref}")
    elapsed = time.time() - t0

    correct = 0
    verifier_available = True
    total_tok = 0
    errors = 0
    details = []
    for i, ((prob, ref), (pred, comp, err, ok, verr, stats)) in enumerate(zip(problems, results)):
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
                "problem": prob,
                "reference": ref,
                "prediction": pred,
                "completion_tokens": comp,
                "correct": ok,
                "error": err,
                "verify_error": verr,
                "stats": stats,
            }
        )

    print("=" * 60)
    if verifier_available:
        print(f"Accuracy:    {correct / N * 100:.1f}%  ({correct}/{N})")
    else:
        print("Accuracy:    [math_verify not installed — skipped]")
    print(f"Total tokens: {total_tok:,}")
    print(f"Wall time:    {elapsed:.1f}s")
    print(f"Throughput:   {total_tok / elapsed:.1f} tok/s" if elapsed > 0 else "")
    print(f"Errors:       {errors}")
    valid_stats = [d["stats"] for d in details if d.get("stats")]
    if valid_stats:
        mean_acc = sum(s["avg_accept_len"] for s in valid_stats) / len(valid_stats)
        mean_tpf = sum(s["tok_per_fwd"] for s in valid_stats) / len(valid_stats)
        denom = valid_stats[0]["verify_num_specs"]
        print(f"Mean acc len: {mean_acc:.2f}/{denom}  (across {len(valid_stats)}/{N} reqs)")
        print(f"Mean tok/fwd: {mean_tpf:.2f}")

    # Step-1: pre_conf → accept_len calibration table
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
        with open(os.path.join(args.output_dir, f"math500_summary{tag}.json"), "w") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(args.output_dir, f"math500_details{tag}.json"), "w") as f:
            json.dump(details, f, indent=2, ensure_ascii=False)
        # Copy/symlink rounds file into output dir for archival
        if args.rounds_file and os.path.exists(args.rounds_file):
            import shutil
            shutil.copy(args.rounds_file,
                        os.path.join(args.output_dir, f"math500_rounds{tag}.jsonl"))
        print(f"Saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
