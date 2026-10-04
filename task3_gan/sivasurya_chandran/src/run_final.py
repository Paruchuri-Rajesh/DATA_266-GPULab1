"""
The graded Task 3 run, end to end, in one command (from the member folder):

    python src/run_final.py --hours 3

1. Trains every variant in config.yaml's `variants:` section at the same time on the one GPU (each process
   sizes its schedule to the time budget from its own measured speed, so all finish together).
2. Each variant keeps its best epoch by the instructor's metric (both directions) on held-out photos
   (raw or EMA generator weights).
3. The variant with the lowest held-out score is promoted: its weights, metrics, manifest, samples and a byte
   copy of its raw log go to checkpoints/ and logs/train_raw.log. All variants' logs and weights stay in
   checkpoints/runs/ and logs/ as the ablation.
4. Post-processing with the promoted model: src/evaluate.py (both directions; writes outputs/pred_A2B for all
   300 Monet) -> src/translate.py --flat (outputs/pred_B2A for all 7,038 photos) -> evaluate_local.py (the
   instructor's script: submission.csv) -> human audit images -> tools/export_reports.py.

Re-running the same command after an interruption resumes every unfinished variant from its train_state.pt
and skips finished ones. Never submits to Kaggle. Progress: logs/final_run_status.log (and the console).
"""
import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MEMBER_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from cfg import load_config, variant_names  # noqa: E402

LOGS = "logs"  # set from the config's log_file in main()


def say(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(LOGS, "final_run_status.log"), "a") as f:
        f.write(line + "\n")


def tail(path, n=3, prefix=None):
    if not os.path.exists(path):
        return []
    with open(path, errors="replace") as f:
        lines = [l.rstrip() for l in f if l.strip()]
    if prefix:
        lines = [l for l in lines if l.startswith(prefix)] or lines
    return lines[-n:]


def run(cmd, label):
    say(f"{label}: {' '.join(cmd)}")
    t = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    with open(os.path.join(LOGS, f"post_{label}.txt"), "w") as f:
        f.write(out)
    if r.returncode != 0:
        say(f"{label} FAILED (exit {r.returncode}, {time.time() - t:.0f}s); last lines:\n" + "\n".join(out.strip().splitlines()[-15:]))
        return False
    say(f"{label} ok ({time.time() - t:.0f}s)")
    return True


def train_all(cfg_path, variants, train_hours, base_out):
    procs = {}
    for v in variants:
        vdir = os.path.join(base_out, "runs", v)
        if os.path.exists(os.path.join(vdir, "metrics_train.json")):
            say(f"{v}: already finished, skipping")
            continue
        cmd = [sys.executable, os.path.join("src", "train.py"), "--config", cfg_path, "--variant", v, "--budget_hours", f"{train_hours:.3f}"]
        if os.path.exists(os.path.join(vdir, "train_state.pt")):
            cmd.append("--resume")
        out = open(os.path.join(LOGS, f"stdout_{v}.txt"), "a")
        procs[v] = (subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT), out)
        say(f"{v}: started (pid {procs[v][0].pid}){' [resume]' if '--resume' in cmd else ''}")
        time.sleep(5)
    t_report = 0
    while procs:
        for v, (p, out) in list(procs.items()):
            if p.poll() is not None:
                out.close()
                del procs[v]
                if p.returncode == 0:
                    say(f"{v}: finished")
                else:
                    say(f"{v}: EXITED WITH ERROR {p.returncode}; last output:\n" + "\n".join(tail(os.path.join(LOGS, f"stdout_{v}.txt"), 15)))
        if procs and time.time() - t_report > 300:
            t_report = time.time()
            for v in procs:
                lines = tail(os.path.join(LOGS, f"stdout_{v}.txt"), 40)
                step = next((l for l in reversed(lines) if " step=" in l), "")
                fid = next((l for l in reversed(lines) if "select_score" in l or "budget " in l), "")
                say(f"{v}: {step[:170]}" + (f"\n    {fid[:200]}" if fid else ""))
        time.sleep(15)


def summarise(variants, base_out, cfg_path):
    rows = []
    for v in variants:
        m_path = os.path.join(base_out, "runs", v, "metrics_train.json")
        if not os.path.exists(m_path):
            rows.append({"variant": v, "status": "no metrics (failed or unfinished)"})
            continue
        m = json.load(open(m_path))
        sel = m.get("checkpoint_selection", {})
        b = sel.get("best", {})
        vc = load_config(cfg_path, v)
        rows.append({
            "variant": v, "status": "ok", "best_score": b.get("score"), "best_fid": b.get("fid"), "best_fid_b2a": b.get("fid_b2a"),
            "best_fid_a2b": b.get("fid_a2b"), "best_epoch": b.get("epoch"), "best_weights": b.get("weights"),
            "epochs": m["epochs"], "steps": m["steps"], "training_time_sec": round(m["training_time_sec"]),
            "sec_per_step": round(m["training_time_sec"] / max(1, m["steps"]), 4),
            "d_scales": vc["model"].get("d_scales", 1),
            "lambda_cycle": (f"{vc['train']['lambda_cycle']}->{vc['train']['lambda_cycle_end']}" if vc["train"].get("lambda_cycle_end", vc["train"]["lambda_cycle"]) != vc["train"]["lambda_cycle"] else vc["train"]["lambda_cycle"]),
            "lambda_identity": vc["train"]["lambda_identity"],
            "diffaugment": vc["train"].get("diffaugment", ""), "ema_decay": vc["train"].get("ema_decay", 0),
            "fid_history": [{k: r[k] for k in ("epoch", "raw", "ema")} for r in sel.get("fid_history", []) if not r.get("calibration")],
        })
    ok = [r for r in rows if r["status"] == "ok" and r["best_score"] is not None and math.isfinite(r["best_score"])]
    winner = min(ok, key=lambda r: r["best_score"])["variant"] if ok else None
    with open(os.path.join(base_out, "variants_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"winner": winner, "selection": "lowest held-out score, instructor's metric, both directions (photos never trained on)", "variants": rows}, f, indent=2)
    head = ("| Variant | D scales | λ_cycle | λ_identity | DiffAugment | Epochs | Steps | s/step (incl. checks) | Best held-out score | FID (photo→Monet / Monet→photo) | Best epoch (weights) |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n")
    body = "".join(
        f"| {'**' + r['variant'] + '**' if r['variant'] == winner else r['variant']} | {r['d_scales']} | {r['lambda_cycle']} | {r['lambda_identity']} | "
        f"{r['diffaugment']} | {r['epochs']} | {r['steps']} | {r['sec_per_step']} | {r['best_score']:.3f} | {r['best_fid']:.2f} ({r['best_fid_b2a']:.2f} / {r['best_fid_a2b']:.2f}) | "
        f"{r['best_epoch']} ({r['best_weights']}) |\n"
        if r["status"] == "ok" else f"| {r['variant']} | {r['status']} ||||||||||\n" for r in rows)
    # encoding="utf-8": the table header contains λ and → ; on Windows the default cp1252 cannot encode
    # them, which raised UnicodeEncodeError here and killed the whole post-processing chain (2026-10-02).
    with open(os.path.join(base_out, "variants_summary.md"), "w", encoding="utf-8") as f:
        f.write(f"# Task 3 final run: variants trained in parallel\n\nWinner (lowest held-out score, instructor's metric): **{winner}**\n\n{head}{body}")
    return winner, rows


def promote(winner, base_out, base_log, variant_log):
    src = os.path.join(base_out, "runs", winner)
    for name in ("G_AB.pt", "G_BA.pt", "D_A.pt", "D_B.pt", "metrics_train.json", "manifest.json", "loss_curves.png", "split.json"):
        shutil.copy2(os.path.join(src, name), os.path.join(base_out, name))
    if os.path.isdir(os.path.join(src, "samples")):
        shutil.copytree(os.path.join(src, "samples"), os.path.join(base_out, "samples"), dirs_exist_ok=True)
    if os.path.exists(base_log):  # never delete a raw log: keep an earlier promoted copy under a new name
        if open(base_log, "rb").read() != open(variant_log, "rb").read():
            os.rename(base_log, base_log.replace(".log", f"_before_{time.strftime('%Y%m%d_%H%M%S')}.log"))
    shutil.copy2(variant_log, base_log)
    with open(os.path.join(base_out, "PROMOTED_FROM.txt"), "w") as f:
        f.write(f"{winner}\nfiles copied from {src}; {base_log} is a byte copy of {variant_log}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--hours", type=float, default=3.0, help="total wall-clock budget, including post-processing")
    ap.add_argument("--reserve_min", type=float, default=20, help="minutes kept for post-processing (eval, translate, score)")
    ap.add_argument("--variants", default=None, help="comma-separated subset of the config's variants (default: all)")
    ap.add_argument("--skip_post", action="store_true")
    ap.add_argument("--no_export", action="store_true", help="skip tools/export_reports.py")
    a = ap.parse_args()

    global LOGS
    os.chdir(MEMBER_DIR)
    cfg = load_config(a.config)
    base_out, base_log = cfg["train"]["out_dir"], cfg["train"]["log_file"]
    LOGS = os.path.dirname(base_log) or "."
    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(base_out, exist_ok=True)
    variants = a.variants.split(",") if a.variants else variant_names(a.config)
    if not variants:
        raise SystemExit(f"{a.config} has no variants: section")
    dcfg = cfg["data"]
    stats = (cfg.get("select") or {}).get("stats")
    missing = [p for p in (dcfg["domain_a_dir"], dcfg["domain_b_dir"]) if not glob.glob(os.path.join(p, "*.jp*g"))]
    if stats and not os.path.exists(stats):
        missing.append(stats)
    if missing:
        raise SystemExit(f"missing data: {missing}\nrun: python ../download_data.py   (or --zip <competition zip>)")

    try:
        import torch
        gpu = f"{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB)" if torch.cuda.is_available() else "no CUDA GPU"
    except Exception as e:  # noqa: BLE001
        gpu = f"unknown ({e})"
    pid_file = os.path.join(LOGS, "run_final.pid")  # lets the notebook see whether a run is already going
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))
    train_hours = a.hours - a.reserve_min / 60
    say(f"=== Task 3 final run | GPU: {gpu} | budget {a.hours:.2f} h (training {train_hours:.2f} h) | variants: {', '.join(variants)} ===")
    if train_hours <= 0.2:
        raise SystemExit("--hours too small")

    train_all(a.config, variants, train_hours, base_out)
    winner, rows = summarise(variants, base_out, a.config)
    for r in rows:
        say(f"{r['variant']}: " + (f"best held-out score {r['best_score']:.3f}, FID {r['best_fid']:.2f} (epoch {r['best_epoch']}, {r['best_weights']}), "
                                   f"{r['steps']} steps in {r['training_time_sec'] / 3600:.2f} h" if r["status"] == "ok" else r["status"]))
    if not winner:
        raise SystemExit("no variant produced a checkpoint; see logs/stdout_*.txt")
    _, wlog = os.path.split(load_config(a.config, winner)["train"]["log_file"])
    promote(winner, base_out, base_log, os.path.join(os.path.dirname(base_log), wlog))
    say(f"WINNER: {winner} -> promoted to {base_out}/ and {base_log}")
    if a.skip_post:
        os.remove(pid_file)
        return

    csv = (cfg.get("submit") or {}).get("csv", "submission.csv")
    pred = cfg["eval"].get("pred_dir", "outputs")
    py = sys.executable
    ok_a2b = run([py, "src/evaluate.py", "--config", a.config], "evaluate")  # writes <pred>/pred_A2B (all 300 Monet)
    ok_b2a = run([py, "src/translate.py", "--config", a.config, "--input", dcfg["domain_a_dir"],
                  "--output", os.path.join(pred, "pred_B2A"), "--flat"], "translate")  # all 7,038 photos
    if ok_a2b and ok_b2a and run([py, "evaluate_local.py", "--real_monet", dcfg["domain_b_dir"], "--real_photo", dcfg["domain_a_dir"],
                                  "--pred_dir", pred, "--out", csv], "score"):
        d = json.load(open(os.path.splitext(csv)[0] + "_details.json"))
        say(f"SUBMISSION {csv} (instructor's script): FID {d['FID']:.3f} (photo->Monet {d['FID_B2A_photo_to_monet']:.2f}, "
            f"Monet->photo {d['FID_A2B_monet_to_photo']:.2f})  MiFID {d['MiFID']:.4f}  -> leaderboard score -{d['score_mean']:.4f} (NOT submitted)")
    run([py, "src/human_audit.py", "prepare", "--config", a.config], "audit_prepare")
    if not a.no_export:
        run([py, os.path.join("..", "..", "tools", "export_reports.py"), "--member", os.path.basename(MEMBER_DIR), "--task", "task3"], "export")
    say("=== ALL DONE ===")
    os.remove(pid_file)


if __name__ == "__main__":
    main()
