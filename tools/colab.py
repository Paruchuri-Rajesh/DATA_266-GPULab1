"""
Helpers for running the task notebooks on a Colab GPU runtime, including a
Colab runtime attached to VS Code through the Google Colab extension.

The repo lives on Google Drive (single source of truth, synced to your laptop by
Google Drive for Desktop). Each runtime session:
  1. mounts Drive,
  2. copies the code to /content/repo (fast local disk),
  3. symlinks the member's checkpoints/, logs/, outputs/, data_processed/ (and
     Task 3 submission/) back to Drive, so results survive a disconnect and
     show up on your laptop,
  4. installs the few packages Colab lacks.
Training is launched as a background process (launch) and watched with
monitor; interrupting the monitor cell or losing the VS Code connection does
not stop training. After a runtime reset, re-run setup and launch: training
continues from the last finished epoch (--resume).
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time

LOCAL = "/content/repo"
PERSIST = ("checkpoints", "logs", "outputs", "data_processed")
TASKS = ("task1_llm", "task2_sentiment", "task3_gan")
# console mirror of the training process (the graded evidence is logs/train_raw.log)
STDOUT = os.path.join("/content" if os.path.isdir("/content") else tempfile.gettempdir(), "train_stdout.txt")


def sh(cmd: str):
    print("$", cmd)
    subprocess.run(cmd, shell=True, check=True)


def mount_drive():
    if os.path.isdir("/content/drive/MyDrive"):
        return
    from google.colab import drive

    try:
        drive.mount("/content/drive")
    except Exception as e:  # the VS Code extension has had mount problems in some versions
        raise RuntimeError(
            "Google Drive did not mount on this runtime. Update the Google Colab VS Code extension, "
            "or open this same notebook at colab.research.google.com (File > Open > Google Drive): it runs unchanged there."
        ) from e


def setup(drive_repo: str, task: str, member: str) -> str:
    mount_drive()
    assert os.path.isdir(drive_repo), f"{drive_repo} not found on Drive"
    excludes = " ".join(f"--exclude '{d}/'" for d in PERSIST + ("submission",))
    excludes += " --exclude '*_smoketest*' --exclude '.venv/' --exclude '.git/' --exclude 'backups/'"
    for t in TASKS:  # only this task's raw data (TinyStories alone is ~2 GB)
        if t != task:
            excludes += f" --exclude '/{t}/data/'"
    if not shutil.which("rsync"):
        sh("apt-get -qq install -y rsync > /dev/null")
    sh(f"rsync -a {excludes} '{drive_repo}/' {LOCAL}/")
    local_member = os.path.join(LOCAL, task, member)
    for d in PERSIST + (("submission",) if task == "task3_gan" else ()):
        target = os.path.join(drive_repo, task, member, d)
        os.makedirs(target, exist_ok=True)
        link = os.path.join(local_member, d)
        if os.path.islink(link) or os.path.exists(link):
            sh(f"rm -rf '{link}'")
        os.symlink(target, link)
    sh(f"pip install -q -r {LOCAL}/requirements-colab.txt")
    os.chdir(local_member)
    import torch

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU: Runtime > Change runtime type > T4 GPU"
    print(f"working in {local_member}\nresults persist in {os.path.join(drive_repo, task, member)}\ntorch {torch.__version__} | {gpu}")
    return local_member


def _running(pattern: str) -> bool:
    # "[s]rc/train.py" matches the training process but not the shell running this pgrep
    return subprocess.run(f"pgrep -f '[{pattern[0]}]{pattern[1:]}'", shell=True, capture_output=True).returncode == 0


def launch(cmd: str, done_file: str, resumable: bool = True, stdout: str = STDOUT):
    """Starts `cmd` in the background unless done_file exists or it is already running."""
    if os.path.exists(done_file):
        return print(f"already finished ({done_file} exists), nothing to do")
    if _running(cmd.split()[1]):
        return print("already running, use monitor()")
    if resumable and os.path.exists(os.path.join(os.path.dirname(done_file), "train_state.pt")):
        cmd += " --resume"
    if cmd.startswith("python "):  # the kernel's own interpreter (a bare "python" may not be the venv's on a local kernel)
        cmd = f"'{sys.executable}' " + cmd[len("python "):]
    subprocess.Popen(f"nohup {cmd} > {stdout} 2>&1 &", shell=True)
    print("started:", cmd)


def monitor(done_file: str, log: str = "logs/train_raw.log", pattern: str = "src/train.py",
            stdout: str = STDOUT, every: int = 60):
    """Shows the tail of the raw log every `every` s until the run ends. Safe to interrupt."""
    from IPython.display import clear_output

    while _running(pattern):
        clear_output(wait=True)
        if os.path.exists(log):
            print("\n".join(open(log).read().splitlines()[-6:]))
        print(f"\n[{time.strftime('%H:%M:%S')}] running; interrupting this cell does not stop training")
        time.sleep(every)
    clear_output(wait=True)
    if os.path.exists(done_file):
        print("finished:", done_file)
    else:
        print("process ended WITHOUT", done_file, "; last console output:")
        if os.path.exists(stdout):
            print("\n".join(open(stdout).read().splitlines()[-30:]))


def sync_back(drive_repo: str, task: str, member: str):
    """Copies files written outside the symlinked folders (metrics_report.csv, failure_analysis.md,
    reproducibility/) to Drive. Never overwrites a failure_analysis.md or results.md already on Drive."""
    src, dst = os.path.join(LOCAL, task, member), os.path.join(drive_repo, task, member)
    sh(f"rsync -a --include='*.md' --exclude='*' --ignore-existing '{src}/' '{dst}/'")
    sh(f"rsync -a --include='*.csv' --exclude='*' '{src}/' '{dst}/'")
    sh(f"rsync -a '{LOCAL}/reproducibility/' '{drive_repo}/reproducibility/'")
    if os.path.exists(f"{LOCAL}/report/team_tables.md"):
        sh(f"cp '{LOCAL}/report/team_tables.md' '{drive_repo}/report/team_tables.md'")
