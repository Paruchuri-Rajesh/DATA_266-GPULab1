"""
Builds markdown metric tables from the metrics JSON each pipeline writes,
so numbers in results.md and the team report are copied, not retyped.

  python tools/make_tables.py --member sivasurya_chandran        # one member, all tasks
  python tools/make_tables.py --team                             # every member side by side (report tables)
  add --ckpt checkpoints_smoketest to read smoke-test outputs instead of checkpoints/
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = {"task1": "task1_llm", "task2": "task2_sentiment", "task3": "task3_gan"}


def load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.{nd}f}" if abs(v) < 1e5 else f"{v:,.0f}"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, float) for x in v):
        return f"[{v[0]:.4f}, {v[1]:.4f}]"
    return str(v)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(fmt(c) if not isinstance(c, str) else c for c in r) + " |" for r in rows]
    return "\n".join(out)


def members(task_dir):
    d = os.path.join(ROOT, task_dir)
    return sorted(m for m in os.listdir(d) if os.path.isdir(os.path.join(d, m)) and m not in ("data", "__pycache__"))


def task1_rows(member, ckpt):
    base = os.path.join(ROOT, TASKS["task1"], member, ckpt)
    t, g = load(os.path.join(base, "metrics_train.json")), load(os.path.join(base, "generation_metrics.json"))
    if not t:
        return None
    g = g or {}
    return [
        ("Training cross-entropy (nats/char)", t["train_cross_entropy"]),
        ("Validation cross-entropy (nats/char)", t["val_cross_entropy"]),
        ("Perplexity (val)", t["perplexity_val"]),
        ("Bits-per-character (val)", t["bits_per_character_val"]),
        ("Generalization gap (val − train)", t["generalization_gap"]),
        ("Top-1 next-char accuracy (val)", t["top1_next_char_accuracy_val"]),
        ("Distinct-1 / 2 / 3", " / ".join(fmt(g.get(k)) for k in ("distinct_1", "distinct_2", "distinct_3"))),
        ("Repeated 4-gram rate (sampled / greedy)", f"{fmt(g.get('repeated_4gram_rate'))} / {fmt(g.get('repeated_4gram_rate_greedy'))}"),
        ("Grad norm mean / p95 / max", f"{t['grad_norm_mean']:.3f} / {t['grad_norm_p95']:.3f} / {t['grad_norm_max']:.3f}"),
        ("NaN steps / loss spikes", f"{t['nan_steps']} / {t['loss_spikes']}"),
        ("fp16 overflow steps skipped by GradScaler", t.get("amp_overflow_skipped_steps", "—")),
        ("Parameter count", t["param_count"]),
        ("Training tokens/sec", t["training_tokens_per_sec"]),
        ("Generation tokens/sec", g.get("generation_tokens_per_sec")),
        ("Peak memory (MB)", t["peak_memory_mb"]),
        ("Total training time (s)", t["training_time_sec"]),
    ]


def task2_models(member, ckpt):
    m = load(os.path.join(ROOT, TASKS["task2"], member, ckpt, "metrics_task2.json"))
    return m


def task2_table(models, mcnemar):
    names = list(models)
    keys = [
        ("Accuracy", "accuracy"), ("Precision macro", "precision_macro"), ("Recall macro", "recall_macro"), ("F1 macro", "f1_macro"),
        ("Precision micro", "precision_micro"), ("Recall micro", "recall_micro"), ("F1 micro", "f1_micro"),
        ("Precision weighted", "precision_weighted"), ("Recall weighted", "recall_weighted"), ("F1 weighted", "f1_weighted"),
        ("Confusion matrix [[TN,FP],[FN,TP]]", "confusion_matrix"), ("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"), ("MCC", "mcc"),
        ("Brier score", "brier_score"), ("ECE", "ece"), ("Accuracy 95% CI", "accuracy_ci95"), ("Macro-F1 95% CI", "macro_f1_ci95"),
        ("MCC 95% CI", "mcc_ci95"), ("Parameter count", "param_count"), ("Training time (s)", "training_time_sec"),
        ("Train examples/sec", "train_examples_per_sec"), ("Inference examples/sec", "inference_examples_per_sec"), ("Peak memory (MB)", "peak_memory_mb"),
    ]
    rows = [[label] + [models[n].get(k) for n in names] for label, k in keys]
    rows.append(["McNemar vs baseline (b / c / p)"] + [
        "—" if n == "baseline" or f"baseline_vs_{n}" not in mcnemar
        else "{b_A_right_B_wrong} / {c_A_wrong_B_right} / p={p_value:.4g}".format(**mcnemar[f"baseline_vs_{n}"]) for n in names
    ])
    rows.append(["Hardware"] + [models[n]["hardware"].get("gpu", models[n]["hardware"]["cpu"]) for n in names])
    out = table(["Metric"] + [f"{n} ({models[n]['type']})" for n in names], rows)
    slice_names = list(next(iter(models.values())).get("slices", {}))
    if slice_names:
        srows = [[s] + [f"{models[n]['slices'][s]['macro_f1']:.4f} / {models[n]['slices'][s]['error_rate']:.4f}" for n in names] for s in slice_names]
        out += "\n\n**Per-slice macro-F1 / error rate**\n\n" + table(["Slice (n)"] + names, srows)
    return out


def task3_rows(member, ckpt):
    base = os.path.join(ROOT, TASKS["task3"], member, ckpt)
    t, e = load(os.path.join(base, "metrics_train.json")), load(os.path.join(base, "metrics_eval.json"))
    a = load(os.path.join(base, "audit", "audit_results.json"))
    sub = load(os.path.join(ROOT, TASKS["task3"], member, "submission_details.json"))
    course = (sub or {}).get("final_model_course_script", sub)  # instructor's evaluation script on the final model
    if not t or not e:
        return None, None
    p2m, m2p = e["photo_to_monet"], e["monet_to_photo"]
    both = [
        (label, p2m[k], m2p[k])
        for label, k in [
            ("FID ↓", "fid"), ("KID ↓", "kid_mean"), ("Precision ↑", "precision"), ("Recall ↑", "recall"), ("Density", "density"),
            ("Coverage ↑", "coverage"), ("Cycle-reconstruction L1 ↓", "cycle_reconstruction_l1"),
            ("LPIPS input vs translation", "lpips_input_vs_translation"), ("Content cosine similarity ↑", "content_cosine_input_vs_translation"),
            ("Identity L1 ↓", "identity_l1"),
        ]
    ]
    if course and "FID_B2A_photo_to_monet" in course:
        both.append(("FID, instructor's script (first 300) ↓", course["FID_B2A_photo_to_monet"], course["FID_A2B_monet_to_photo"]))
        both.append(("MiFID, instructor's script (first 300) ↓", course["MiFID_B2A_photo_to_monet"], course["MiFID_A2B_monet_to_photo"]))
    single = [
        ("Final cycle / identity loss", f"{fmt(t['final_cycle_loss'])} / {fmt(t['final_identity_loss'])}"),
        ("Final G adversarial / D_A / D_B loss", f"{fmt(t['final_loss_G_adv'])} / {fmt(t['final_loss_D_A'])} / {fmt(t['final_loss_D_B'])}"),
        ("Grad norm G mean/max", f"{t['grad_norm_G']['mean']:.3f} / {t['grad_norm_G']['max']:.3f}"),
        ("Grad norm D mean/max", f"{t['grad_norm_D']['mean']:.3f} / {t['grad_norm_D']['max']:.3f}"),
        ("NaN count", t["nan_count"]),
        ("fp16 overflow steps skipped (G / D)", "{G} / {D}".format(**t["amp_overflow_skipped_steps"]) if "amp_overflow_skipped_steps" in t else "—"),
        ("Human audit score (1–5)", a["overall_human_audit_score_1to5"] if a else "—"),
        ("Inter-rater κ (quadratic) / exact agreement %", f"{fmt(a['overall_cohens_kappa_quadratic'])} / {fmt(a['overall_exact_agreement_pct'], 1)}" if a else "—"),
        ("Parameter count (G_AB+G_BA+D_A+D_B)", t["param_count"]["total"]),
        ("Training time (s)", t["training_time_sec"]),
        ("Images/sec", t["images_per_sec"]),
        ("Peak memory (MB)", t["peak_memory_mb"]),
        ("FID / MiFID, instructor's script (mean of both directions)",
         f"{fmt(course['FID'])} / {fmt(course['MiFID'])}" if course else "—"),
        ("submission.csv (Kaggle entry, " + sub["model"].split(",")[0] + ") FID / MiFID", sub["submission_csv"].split("/ 1,")[-1].replace(",", " / ") +
         " (7,038 photo→Monet vs competition real_stats.npz)") if sub and "model" in sub else ("submission.csv", "—"),
        ("Kaggle public / private score", sub.get("kaggle_public_score", f"{fmt(-(sub['FID'] + sub['MiFID']) / 2)} expected; not submitted yet")
         if sub else "fill in after submission"),
    ]
    return both, single


def member_report(member, ckpt):
    parts = [f"# Metric tables — {member} (from `{ckpt}/`)\n"]
    r1 = task1_rows(member, ckpt)
    parts.append("## Task 1\n\n" + (table(["Metric", "Value"], r1) if r1 else "_no Task 1 metrics yet_"))
    m2 = task2_models(member, ckpt)
    parts.append("## Task 2\n\n" + (task2_table(m2["models"], m2["mcnemar"]) if m2 else "_no Task 2 metrics yet_"))
    both, single = task3_rows(member, ckpt)
    if both:
        parts.append("## Task 3\n\n" + table(["Metric", "photo→Monet", "Monet→photo"], both) + "\n\n" + table(["Metric", "Value"], single))
    else:
        parts.append("## Task 3\n\n_no Task 3 metrics yet_")
    return "\n\n".join(parts) + "\n"


def team_report(ckpt):
    parts = ["# Team comparison tables\n"]
    rows = {}
    for m in members(TASKS["task1"]):
        r = task1_rows(m, ckpt)
        if r:
            cfg = load(os.path.join(ROOT, TASKS["task1"], m, ckpt, "manifest.json"))["config"]
            arch = f"L={cfg['model']['n_layer']} H={cfg['model']['n_head']} d={cfg['model']['n_embd']} T={cfg['data']['block_size']}"
            hp = f"lr={cfg['train']['lr']} bs={cfg['train']['batch_size']} warmup={cfg['train']['warmup_steps']} ep={cfg['train']['epochs']} drop={cfg['model']['dropout']}"
            rows[m] = [("Architecture", arch), ("Hyperparameters", hp)] + r
    if rows:
        names = list(rows)
        labels = [l for l, _ in rows[names[0]]]
        parts.append("## Task 1\n\n" + table(["Metric"] + names, [[l] + [dict(rows[n])[l] for n in names] for l in labels]))
    t2 = {}
    for m in members(TASKS["task2"]):
        d = task2_models(m, ckpt)
        if d:
            for n, rep in d["models"].items():
                t2[f"{m}/{n}"] = rep
    if t2:
        rows2 = [[k, v["type"], json.dumps({x: y for x, y in v["hyperparameters"].items() if x != "type"}), v["accuracy"], v["f1_macro"], v["roc_auc"], v["pr_auc"], v["mcc"], v["brier_score"], v["ece"], v["param_count"], v["training_time_sec"]] for k, v in t2.items()]
        parts.append("## Task 2\n\n" + table(["Member/model", "Arch", "Hyperparameters", "Acc", "Macro-F1", "ROC-AUC", "PR-AUC", "MCC", "Brier", "ECE", "Params", "Train s"], rows2))
    t3 = []
    for m in members(TASKS["task3"]):
        both, single = task3_rows(m, ckpt)
        if both:
            cfg = load(os.path.join(ROOT, TASKS["task3"], m, ckpt, "manifest.json"))["config"]
            arch = f"ngf={cfg['model']['ngf']} ndf={cfg['model']['ndf']} res={cfg['model']['n_resblocks']} img={cfg['data']['image_size']}"
            hp = f"lr={cfg['train']['lr']} λcyc={cfg['train']['lambda_cycle']} λid={cfg['train']['lambda_identity']} ep={cfg['train']['epochs']}"
            d = {l: (a, b) for l, a, b in both}
            t3.append([m, arch, hp] + [f"{fmt(d[k][0])} / {fmt(d[k][1])}" for k in ("FID ↓", "KID ↓", "Cycle-reconstruction L1 ↓", "Content cosine similarity ↑")])
    if t3:
        parts.append("## Task 3 (photo→Monet / Monet→photo)\n\n" + table(["Member", "Architecture", "Hyperparameters", "FID", "KID", "Cycle L1", "Content cos"], t3))
    return "\n\n".join(parts) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--member")
    ap.add_argument("--team", action="store_true")
    ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    md = team_report(a.ckpt) if a.team else member_report(a.member, a.ckpt)
    if a.out:
        with open(a.out, "w") as f:
            f.write(md)
    print(md)
