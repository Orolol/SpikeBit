"""Remaining rung-3 runs on a Colab VM driven by the `colab` CLI (same runs as ladder_rung3.ipynb).

The CLI path skips Drive: the code bundle, the KL teacher checkpoint and the packed data are
uploaded into /content/SpikeBit, and outputs stay on the VM disk until downloaded. The queue
runs inside the kernel so the session stays busy (the CLI keeps a session alive while its
kernel is active):

    colab new -s rung3 --gpu A100
    (upload spikebit_<commit>.tar.gz, extract into /content/SpikeBit, pip install -e . ninja,
     upload checkpoints/ladder/rung1c_lr1e-3_s0/final.pt and data/tinystories_packed_*.npz)
    colab exec -s rung3 --timeout 40000 -f colab/ladder_rung3.py

The image has no ninja: without it the WKV and FAI-LIF CUDA extensions fail to build and the
trainer falls back to the per-step Python reference, ~90x slower. Both are checked below.
A finished run (`=== exit 0` in its pipeline.log) is skipped on re-run.

One run leaves the A100 half idle at this batch size, so a second copy started on the same VM
with the argument `reverse` (e.g. from `colab console`, under nohup) takes the queue from the
other end and skips every run that already has a pipeline.log (started by either copy).
The forward copy only skips finished runs, so it must never reach a run the reverse copy is
still training: reserve the middle of the queue for it by writing a line into that run's
pipeline.log before the reverse copy gets there (done here for rung3l_a4h8_s0, leaving the
reverse copy rung3l_a4h8_s1 and rung3l_a4_s1).
"""
import datetime, hashlib, os, subprocess, sys

import torch

os.chdir("/content/SpikeBit")
COMMIT = open("BUNDLE_COMMIT").read().strip()
GPU = torch.cuda.get_device_name(0).replace(" ", "_")
assert "A100" in GPU, f"expected an A100 runtime, got {GPU}"
KL = "checkpoints/ladder/rung1c_lr1e-3_s0/final.pt"
h = hashlib.sha256()
with open(KL, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
        h.update(chunk)
assert h.hexdigest() == "04ea63897adf6dc7538db63848acd9c6a837883f85047acc3b8e94e9d4354c2c", "KL teacher checkpoint differs from the local one"
ext = subprocess.run(["python", "-c", "from spiking_ternary_lm.layers.wkv_op import _load_extension as a; "
                      "from spiking_ternary_lm.neurons.fai_lif_op import _load_extension as b; "
                      "assert a() is not None and b() is not None"], capture_output=True, text=True)
assert ext.returncode == 0, f"CUDA extensions did not build (ninja missing?):\n{ext.stdout}{ext.stderr}"
print("commit", COMMIT, "| KL teacher OK | CUDA extensions OK |", GPU, flush=True)

KL_ARGS = ["--distill-checkpoint", KL]
TAIL = ["--seq-len", "128", "--batch-size", "16", "--reset-state-every", "1"]


def rung3(act_quant):
    return ["--weight-quant", "ternary", "--act-quant", act_quant, "--ffn-relu-sq", "--act-grid", "faithful",
            "--act-output", "dequant", "--act-ste", "clipped", "--lif-stateless", "--calibrate-batches", "8", *KL_ARGS]


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def ladder_run(name, seed, lr, rung_args):
    run, ck = f"runs/ladder/{name}", f"checkpoints/ladder/{name}"
    os.makedirs(run, exist_ok=True); os.makedirs(ck, exist_ok=True)
    log = f"{run}/pipeline.log"
    if os.path.exists(log) and (REVERSE or "=== exit 0" in open(log).read()):
        print(name, "already started" if REVERSE else "already done", flush=True); return 0
    cmd = ["python", "scripts/train_stage_a.py", "--teacher-id", "RWKV/rwkv-4-169m-pile",
           "--contiguous-rows", "--protocol-eval",
           "--seq-len", "64", "--batch-size", "32", "--grad-accum", "1", "--total-steps", "10000", "--warmup-steps", "300",
           "--lr-bitlinear", lr, "--lr-rwkv", lr, "--lr-embed-head", lr,
           "--log-every", "50", "--eval-every", "2000", "--ckpt-every", "100000", "--seed", str(seed),
           "--out-dir", ck, "--metrics-path", f"{run}/metrics.jsonl", *rung_args, *TAIL]
    with open(log, "a") as f:
        f.write(f"=== commit {COMMIT} host colab-cli {GPU} start {now()}\n{' '.join(cmd)}\n")
    print(name, "start", now(), flush=True)
    with open(f"{run}/train.log", "w") as out:
        rc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT).returncode
    with open(log, "a") as f:
        f.write(f"=== exit {rc} {now()}\n")
    print(name, "exit", rc, now(), "|", subprocess.run(["tail", "-2", f"{run}/train.log"], capture_output=True, text=True).stdout, flush=True)
    return rc


RUNS = [
    ("rung3m_a4h8_s1", 1, rung3("minmax") + ["--bits", "4", "--bits-ffn-hidden", "8"]),
    ("rung3l_a4_s0", 0, rung3("lsq") + ["--bits", "4"]),
    ("rung3l_a4h8_s0", 0, rung3("lsq") + ["--bits", "4", "--bits-ffn-hidden", "8"]),
    ("rung3l_a4_s1", 1, rung3("lsq") + ["--bits", "4"]),
    ("rung3l_a4h8_s1", 1, rung3("lsq") + ["--bits", "4", "--bits-ffn-hidden", "8"]),
]
REVERSE = sys.argv[1:] == ["reverse"]
for run_name, seed, args in RUNS[::-1] if REVERSE else RUNS:
    if ladder_run(run_name, seed, "3e-3", args) != 0:
        raise SystemExit(f"{run_name} failed: see runs/ladder/{run_name}/train.log")
print(subprocess.run(["python", "scripts/ladder_summary.py", "--runs", "runs/ladder", "--prefix", "rung3"],
                     capture_output=True, text=True).stdout, flush=True)
