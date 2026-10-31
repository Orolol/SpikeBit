"""General text (DECISIONS #24 tier 3) on a Colab VM driven by the `colab` CLI: rung 1 and rung 5
retrained on WikiText-103 at the protocol budget, rung 5 distilled from rung 1 seed 0.

Same setup as ladder_budget2x.py (code bundle extracted into /content/SpikeBit, pip install -e .
ninja), with the packed WikiText-103 corpus uploaded to data/ instead of TinyStories. Each copy
trains the runs named in the RUNS environment variable (comma-separated) in order, so two copies
share the A100, one per seed; the one in the kernel keeps the session busy:

    colab new -s tier3 --gpu A100
    colab exec -s tier3 --timeout 40000 --env RUNS=wt103_rung1_lr1e-3_s0,wt103_rung5_a4_s0 -f colab/ladder_general.py
    (in `colab console`, under nohup) RUNS=wt103_rung1_lr1e-3_s1,wt103_rung5_a4_s1 python colab/ladder_general.py

A rung-5 run waits for rung 1 seed 0 to finish. A finished run (`=== exit 0` in its
pipeline.log) is skipped on re-run.
"""
import datetime, hashlib, os, subprocess, time

import torch

os.chdir("/content/SpikeBit")
COMMIT = open("BUNDLE_COMMIT").read().strip()
GPU = torch.cuda.get_device_name(0).replace(" ", "_")
assert "A100" in GPU, f"expected an A100 runtime, got {GPU}"
ext = subprocess.run(["python", "-c", "from spiking_ternary_lm.layers.wkv_op import _load_extension as a; "
                      "from spiking_ternary_lm.neurons.fai_lif_op import _load_extension as b; "
                      "assert a() is not None and b() is not None"], capture_output=True, text=True)
assert ext.returncode == 0, f"CUDA extensions did not build (ninja missing?):\n{ext.stdout}{ext.stderr}"
print("commit", COMMIT, "| CUDA extensions OK |", GPU, flush=True)

DATA = "data/wikitext103_packed_RWKV_rwkv-4-169m-pile_train.npz"
assert os.path.exists(DATA), f"{DATA} missing: upload the packed corpus first"
STEPS = "10000"  # the DEC-15 protocol budget
KL_RUN = "wt103_rung1_lr1e-3_s0"
KL = f"checkpoints/ladder/{KL_RUN}/final.pt"
TAIL = ["--seq-len", "128", "--batch-size", "16", "--reset-state-every", "1", "--dataset", "wikitext103"]
RUNG1 = ["--weight-quant", "none", "--act-quant", "none", "--ffn-relu-sq"]
# Rung 5 as in ladder_rung5.py (DEC-22), distilled from rung 1 seed 0 on the same corpus (DEC-24).
RUNG5 = ["--weight-quant", "ternary", "--act-quant", "lsq", "--ffn-relu-sq", "--act-grid", "faithful",
         "--act-output", "dequant", "--act-ste", "clipped", "--lif-stateless", "--act-quant-inputs",
         "--calibrate-batches", "8", "--bits", "4", "--distill-checkpoint", KL]
RUNS = {"wt103_rung1_lr1e-3_s0": (0, "1e-3", RUNG1), "wt103_rung1_lr1e-3_s1": (1, "1e-3", RUNG1),
        "wt103_rung5_a4_s0": (0, "3e-3", RUNG5), "wt103_rung5_a4_s1": (1, "3e-3", RUNG5)}


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def exit_code(name):
    """Exit code of a run's last attempt (from its pipeline.log); None if it never ran or still runs."""
    log = f"runs/ladder/{name}/pipeline.log"
    markers = [line.split() for line in open(log) if line.startswith("===")] if os.path.exists(log) else []
    return int(markers[-1][2]) if markers and markers[-1][1] == "exit" else None


def wait_for_kl_teacher():
    while (rc := exit_code(KL_RUN)) != 0:
        if rc is not None:
            raise SystemExit(f"{KL_RUN} failed (exit {rc}): no KL teacher for rung 5")
        time.sleep(30)
    h = hashlib.sha256()
    with open(KL, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def ladder_run(name, seed, lr, rung_args):
    run, ck = f"runs/ladder/{name}", f"checkpoints/ladder/{name}"
    os.makedirs(run, exist_ok=True); os.makedirs(ck, exist_ok=True)
    log = f"{run}/pipeline.log"
    if exit_code(name) == 0:
        print(name, "already done", flush=True); return 0
    kl_note = f" KL teacher {KL} sha256 {wait_for_kl_teacher()}" if KL in rung_args else ""
    cmd = ["python", "scripts/train_stage_a.py", "--teacher-id", "RWKV/rwkv-4-169m-pile",
           "--contiguous-rows", "--protocol-eval",
           "--seq-len", "64", "--batch-size", "32", "--grad-accum", "1", "--total-steps", STEPS, "--warmup-steps", "300",
           "--lr-bitlinear", lr, "--lr-rwkv", lr, "--lr-embed-head", lr,
           "--log-every", "50", "--eval-every", "2000", "--ckpt-every", "100000", "--seed", str(seed),
           "--out-dir", ck, "--metrics-path", f"{run}/metrics.jsonl", *rung_args, *TAIL]
    with open(log, "a") as f:
        f.write(f"=== commit {COMMIT} host colab-cli {GPU} start {now()}{kl_note}\n{' '.join(cmd)}\n")
    print(name, "start", now(), kl_note, flush=True)
    with open(f"{run}/train.log", "w") as out:
        rc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT).returncode
    with open(log, "a") as f:
        f.write(f"=== exit {rc} {now()}\n")
    print(name, "exit", rc, now(), "|", subprocess.run(["tail", "-2", f"{run}/train.log"], capture_output=True, text=True).stdout, flush=True)
    return rc


for run_name in os.environ["RUNS"].split(","):
    seed, lr, args = RUNS[run_name]
    if ladder_run(run_name, seed, lr, args) != 0:
        raise SystemExit(f"{run_name} failed: see runs/ladder/{run_name}/train.log")
