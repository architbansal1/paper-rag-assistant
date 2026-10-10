"""
Aggregate eval/runs/*.json into a results table.

Run tags follow  <config>  or  <config>__s<seed>, e.g.
  base_closed, base_rag, ft_closed__s1, ft_closed__s2, ft_rag__s1 ...

For every config we report:
  * mean cosine similarity (averaged over seeds, then over questions) with a
    95% bootstrap CI that resamples QUESTIONS
  * spread across seeds (std of the per-seed means) when there are several
  * refusal count (mean per seed)
  * mean LLM-judge score 0-2 and % fully correct, if src/judge.py was run
  * for ft_X configs, the paired difference vs base_X with a bootstrap CI

Usage: python src/report.py [--runs-dir eval/runs] [--boot 5000]
"""
import os
import re
import json
import glob
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_runs(runs_dir):
    groups = {}
    for path in sorted(glob.glob(os.path.join(runs_dir, "*.json"))):
        with open(path) as f:
            run = json.load(f)
        config = re.sub(r"__s\d+$", "", run["tag"])
        groups.setdefault(config, []).append(run)
    return groups


def per_question_matrix(runs, key):
    """Returns (questions, array [n_runs, n_questions]) for a per-result field."""
    questions = [r["question"] for r in runs[0]["results"]]
    rows = []
    for run in runs:
        by_q = {r["question"]: r for r in run["results"]}
        rows.append([by_q[q].get(key) for q in questions])
    arr = np.array(rows, dtype=float)
    return questions, arr


def boot_ci(per_q, n_boot, rng):
    """95% CI of the mean over questions, resampling questions."""
    n = len(per_q)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = per_q[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=os.path.join(HERE, "..", "eval", "runs"))
    ap.add_argument("--boot", type=int, default=5000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    groups = load_runs(args.runs_dir)
    if not groups:
        raise SystemExit(f"No run files found in {args.runs_dir}")

    summary = {}
    print(f"{'config':<14} {'runs':>4} {'n_q':>4} {'mean cos':>9} {'95% CI':>17} {'seed std':>9} {'refusals':>9} {'judge/2':>8} {'%correct':>9}")
    for config in sorted(groups):
        runs = groups[config]
        qs, sims = per_question_matrix(runs, "similarity")
        per_q = sims.mean(axis=0)               # average over seeds
        lo, hi = boot_ci(per_q, args.boot, rng)
        seed_means = sims.mean(axis=1)
        seed_std = float(seed_means.std(ddof=1)) if len(runs) > 1 else float("nan")
        refusals = float(np.mean([r["refusals"] for r in runs]))

        judge_txt, correct_txt = "-", "-"
        if all("judge_score" in r["results"][0] for r in runs):
            _, js = per_question_matrix(runs, "judge_score")
            judge_txt = f"{js.mean():.2f}"
            correct_txt = f"{(js == 2).mean() * 100:.0f}%"

        summary[config] = {"qs": qs, "per_q": per_q, "per_q_judge": None}
        print(f"{config:<14} {len(runs):>4} {len(qs):>4} {per_q.mean():>9.4f} [{lo:.3f}, {hi:.3f}] {seed_std:>9.4f} {refusals:>9.1f} {judge_txt:>8} {correct_txt:>9}")

    print("\nPaired difference vs matching baseline (fine-tuned minus base, per question, 95% bootstrap CI):")
    for config in sorted(summary):
        if not config.startswith("ft_"):
            continue
        base_cfg = "base_" + config[len("ft_"):]
        if base_cfg not in summary:
            print(f"  {config}: no {base_cfg} runs found")
            continue
        a, b = summary[config], summary[base_cfg]
        if a["qs"] != b["qs"]:
            print(f"  {config}: question sets differ from {base_cfg}; skipping")
            continue
        diff = a["per_q"] - b["per_q"]
        lo, hi = boot_ci(diff, args.boot, rng)
        verdict = "CI excludes 0" if (lo > 0 or hi < 0) else "CI includes 0 (not distinguishable from no change)"
        wins = int((diff > 0).sum())
        print(f"  {config} - {base_cfg}: {diff.mean():+.4f}  [{lo:+.3f}, {hi:+.3f}]  wins {wins}/{len(diff)}  -> {verdict}")

    print("\nWorst per-question regressions (fine-tuned - base), first ft config shown:")
    for config in sorted(summary):
        if config.startswith("ft_") and ("base_" + config[3:]) in summary:
            a, b = summary[config], summary["base_" + config[3:]]
            diff = a["per_q"] - b["per_q"]
            for i in np.argsort(diff)[:3]:
                print(f"  {diff[i]:+.3f}  {a['qs'][i][:90]}")
            break


if __name__ == "__main__":
    main()
