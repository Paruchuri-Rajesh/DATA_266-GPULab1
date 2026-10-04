"""
Builds the per-member deliverables the lab's folder spec asks for, from the
metrics and artifacts the training scripts already wrote:

  <task>/<member>/metrics_report.csv      every required metric for the task, one file
  <task>/<member>/failure_analysis.md     pre-filled template (never overwritten once it exists)
  <task>/<member>/outputs/                plots, samples, predictions, error reviews
  task3_gan/<member>/full_metrics_report.csv
  reproducibility/manifests/, reproducibility/raw_logs/   byte-for-byte copies (logs are never edited)

  python tools/export_reports.py --member sivasurya_chandran
  add --ckpt checkpoints_smoketest to export smoke-test outputs instead, --task task3 for one task only
"""
import argparse
import csv
import glob
import json
import os
import shutil

from make_tables import ROOT, TASKS, fmt, load, task1_rows, task2_models, task3_rows

SFX = ""  # "_smoketest" when exporting smoke-test outputs, so real deliverables are never touched


def copy(src, dst_dir):
    for f in glob.glob(src):
        os.makedirs(dst_dir, exist_ok=True)
        if os.path.isdir(f):
            shutil.copytree(f, os.path.join(dst_dir, os.path.basename(f)), dirs_exist_ok=True)
        else:
            shutil.copy2(f, dst_dir)


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows([[fmt(c) if not isinstance(c, str) else c for c in r] for r in rows])
    print("wrote", os.path.relpath(path, ROOT))


def write_once(path, text):
    if os.path.exists(path):
        print("kept existing", os.path.relpath(path, ROOT))
        return
    with open(path, "w") as f:
        f.write(text)
    print("wrote", os.path.relpath(path, ROOT))


def reproducibility(task, member, mdir, ckpt, log_name):
    tag = f"{task}_{member}" + ("_smoketest" if ckpt != "checkpoints" else "")
    man, log = os.path.join(mdir, ckpt, "manifest.json"), os.path.join(mdir, "logs", log_name)
    if os.path.exists(man):
        shutil.copy2(man, os.path.join(ROOT, "reproducibility", "manifests", f"{tag}_manifest.json"))
    if os.path.exists(log):
        shutil.copy2(log, os.path.join(ROOT, "reproducibility", "raw_logs", f"{tag}_train_raw.log"))


def task1(member, ckpt, out_name, log_name):
    mdir = os.path.join(ROOT, TASKS["task1"], member)
    rows = task1_rows(member, ckpt)
    if not rows:
        return print("task1: no metrics yet")
    write_csv(os.path.join(mdir, f"metrics_report{SFX}.csv"), ["metric", "value"], rows)
    out = os.path.join(mdir, out_name)
    for pat in ("loss_curves.png", "samples.txt", "samples.json", "generation_metrics.json", "failure_candidates.json"):
        copy(os.path.join(mdir, ckpt, pat), out)
    cand = load(os.path.join(mdir, ckpt, "failure_candidates.json")) or {}
    cases = "\n".join(
        f"## Case {i}: {label}\n\n**Snippet** (`{s['mode']}`, T={s['temperature']}, rep4={s['repeated_4gram_rate']:.3f}, "
        f"invented-word rate={s['invented_word_rate']:.3f}):\n\n> {s['text'][:500]}\n\n"
        f"**Failure type:** _repetition / broken grammar / loss of coherence / hallucination_\n\n**Observation:** _\n"
        for i, (label, s) in enumerate(cand.items(), 1)
    )
    write_once(os.path.join(mdir, f"failure_analysis{SFX}.md"),
               f"# Task 1 failure analysis — {member}\n\nThree failure cases from `outputs/samples.txt`. The snippets below were "
               f"flagged automatically; replace any of them with a better example and write the observation in your own words.\n\n{cases}")
    reproducibility("task1", member, mdir, ckpt, log_name)


def task2(member, ckpt, out_name, log_name, data_dir):
    mdir = os.path.join(ROOT, TASKS["task2"], member)
    m = task2_models(member, ckpt)
    if not m:
        return print("task2: no metrics yet")
    models, mc = m["models"], m["mcnemar"]
    names = list(models)
    keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "precision_micro", "recall_micro", "f1_micro",
            "precision_weighted", "recall_weighted", "f1_weighted", "confusion_matrix", "roc_auc", "pr_auc", "mcc",
            "brier_score", "ece", "accuracy_ci95", "macro_f1_ci95", "mcc_ci95", "param_count", "training_time_sec",
            "train_examples_per_sec", "inference_examples_per_sec", "peak_memory_mb"]
    rows = [[k] + [json.dumps(models[n].get(k)) if isinstance(models[n].get(k), list) else models[n].get(k) for n in names] for k in keys]
    rows.append(["mcnemar_p_value_vs_baseline"] + [mc[f"baseline_vs_{n}"]["p_value"] if f"baseline_vs_{n}" in mc else "—" for n in names])
    for s in models[names[0]].get("slices", {}):
        rows.append([f"slice_macro_f1:{s}"] + [models[n]["slices"][s]["macro_f1"] for n in names])
        rows.append([f"slice_error_rate:{s}"] + [models[n]["slices"][s]["error_rate"] for n in names])
    rows.append(["hardware"] + [models[n]["hardware"].get("gpu", models[n]["hardware"]["cpu"]) for n in names])
    write_csv(os.path.join(mdir, f"metrics_report{SFX}.csv"), ["metric"] + [f"{n} ({models[n]['type']})" for n in names], rows)
    out = os.path.join(mdir, out_name)
    for pat in ("test_*.png", "error_review_*", "predictions_*.npz", "metrics_task2.json"):
        copy(os.path.join(mdir, ckpt, pat), out)
    copy(os.path.join(mdir, data_dir, "eda"), out)
    copy(os.path.join(mdir, data_dir, "preprocess_stats.json"), os.path.join(out, "eda"))
    reviews = sorted(glob.glob(os.path.join(mdir, ckpt, "error_review_*.md")))
    body = open(reviews[0]).read().split("\n", 1)[1] if reviews else "_run src/error_analysis.py first_\n"
    write_once(os.path.join(mdir, f"failure_analysis{SFX}.md"),
               f"# Task 2 error review — {member}\n\n20 errors: 5 confident false positives, 5 confident false negatives, "
               f"5 near-threshold, 5 from the worst slice. For each, confirm or override the suggested error type and "
               f"write one testable fix.\n\n## Summary of error types and fixes\n\n| Error type | Count | Testable fix | Metric to check |\n|---|---|---|---|\n| | | | |\n\n{body}")
    reproducibility("task2", member, mdir, ckpt, log_name)


def task3(member, ckpt, out_name, log_name):
    mdir = os.path.join(ROOT, TASKS["task3"], member)
    both, single = task3_rows(member, ckpt)
    if not both:
        return print("task3: no metrics yet")
    rows = [[l, a, b] for l, a, b in both] + [[l, v, ""] for l, v in single]
    for name in (f"metrics_report{SFX}.csv", f"full_metrics_report{SFX}.csv"):
        write_csv(os.path.join(mdir, name), ["metric", "photo_to_monet (G_AB)", "monet_to_photo (G_BA)"], rows)
    out = os.path.join(mdir, out_name)
    for pat in ("loss_curves.png", "eval_*.png", "samples", "metrics_eval.json", "metrics_train.json", "audit/audit_results.json"):
        copy(os.path.join(mdir, ckpt, pat), out)
    write_once(os.path.join(mdir, f"failure_analysis{SFX}.md"),
               f"# Task 3 failure analysis — {member}\n\nUse `outputs/pred_A2B/`, `outputs/pred_B2A/`, `outputs/eval_*_input_fake_rec.png` "
               f"and `outputs/samples/`.\n\n| # | Image | Failure type (colour shift / checkerboard / content loss / mode collapse / weak style) | Observation |\n|---|---|---|---|\n| 1 | | | |\n| 2 | | | |\n| 3 | | | |\n")
    reproducibility("task3", member, mdir, ckpt, log_name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", required=True)
    ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--task", choices=["all", "task1", "task2", "task3"], default="all")
    a = ap.parse_args()
    smoke = a.ckpt != "checkpoints"
    SFX = "_smoketest" if smoke else ""
    out_name = "outputs_smoketest" if smoke else "outputs"
    log_name = "train_raw_smoketest.log" if smoke else "train_raw.log"
    for d in ("manifests", "raw_logs"):
        os.makedirs(os.path.join(ROOT, "reproducibility", d), exist_ok=True)
    if a.task in ("all", "task1"):
        task1(a.member, a.ckpt, out_name, log_name)
    if a.task in ("all", "task2"):
        task2(a.member, a.ckpt, out_name, log_name, "data_smoketest" if smoke else "data_processed")
    if a.task in ("all", "task3"):
        task3(a.member, a.ckpt, out_name, log_name)
