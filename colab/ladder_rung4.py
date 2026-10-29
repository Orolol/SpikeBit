"""Rung 4b runs (DECISIONS #19) on a Colab VM driven by the `colab` CLI.

Same setup as ladder_rung3.py (code bundle extracted into /content/SpikeBit, pip install -e .
ninja, KL teacher checkpoint and packed data uploaded). Each copy trains the runs named in the
RUNS environment variable (comma-separated), so two copies can share the A100 without
racing for a run; the one in the kernel keeps the session busy:

    colab new -s rung4 --gpu A100
    colab exec -s rung4 --timeout 40000 --env RUNS=rung4b_a4_s0 -f colab/ladder_rung4.py
    (in `colab console`, under nohup) RUNS=rung4b_a4_s1 python /content/ladder_rung4.py

A finished run (`=== exit 0` in its pipeline.log) is skipped on re-run.
"""
import datetime, hashlib, os, subprocess

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
# Rung 3 (DEC-18: LSQ, clipped STE, all-4-bit) without --lif-stateless: beta learnable from
# its 0.9 init, membrane clamped to ±s/2 after the reset (DEC-19).
RUNG4B = ["--weight-quant", "ternary", "--act-quant", "lsq", "--ffn-relu-sq", "--act-grid", "faithful",
          "--act-output", "dequant", "--act-ste", "clipped", "--lif-membrane-clamp", "0.5",
          "--calibrate-batches", "8", "--bits", "4", *KL_ARGS]
RUNS = {"rung4b_a4_s0": (0, RUNG4B), "rung4b_a4_s1": (1, RUNG4B)}


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def ladder_run(name, seed, lr, rung_args):
    run, ck = f"runs/ladder/{name}", f"checkpoints/ladder/{name}"
    os.makedirs(run, exist_ok=True); os.makedirs(ck, exist_ok=True)
    log = f"{run}/pipeline.log"
    if os.path.exists(log) and "=== exit 0" in open(log).read():
        print(name, "already done", flush=True); return 0
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


for run_name in os.environ["RUNS"].split(","):
    seed, args = RUNS[run_name]
    if ladder_run(run_name, seed, "3e-3", args) != 0:
        raise SystemExit(f"{run_name} failed: see runs/ladder/{run_name}/train.log")
