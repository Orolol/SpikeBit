#!/usr/bin/env bash
# Fetch ladder runs from a colab-cli session (COLAB_SESSION): the small run files (pipeline.log,
# train.log, metrics.jsonl) into runs/ladder/<run>/, and with --ckpt also
# checkpoints/ladder/<run>/final.pt, downloaded in 32 MB parts and checked against the VM's sha256.
# Paths are those of the main checkout (runs/ and checkpoints/ are shared by the worktrees).
#   COLAB_SESSION=tier3 bash colab/colab_get.sh [--ckpt] RUN...
set -eu
S=${COLAB_SESSION:?set COLAB_SESSION}
R=/home/orosius/workspace/SpikeBit
CKPT=0; [ "${1:-}" = "--ckpt" ] && { CKPT=1; shift; }
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
for r in "$@"; do
  mkdir -p "$R/runs/ladder/$r"
  for f in pipeline.log train.log metrics.jsonl; do
    colab download -s "$S" "/content/SpikeBit/runs/ladder/$r/$f" "$R/runs/ladder/$r/$f" > /dev/null 2>&1 || echo "$r/$f: not there"
  done
  [ $CKPT = 1 ] || continue
  printf '%s\n' "(setsid nohup bash -c \"rm -rf /content/dl/$r && mkdir -p /content/dl/$r && cd /content/dl/$r && split -b 32M -d -a 3 /content/SpikeBit/checkpoints/ladder/$r/final.pt p. && sha256sum /content/SpikeBit/checkpoints/ladder/$r/final.pt | cut -c1-64 > sha && ls p.* > list && touch ready\" > /dev/null 2>&1 < /dev/null &)" 'exit' \
    | timeout 120 colab console -s "$S" > /dev/null 2>&1
  L=$TMP/$r; mkdir -p "$L"
  for i in $(seq 1 60); do colab download -s "$S" /content/dl/$r/ready "$L/ready" > /dev/null 2>&1 && break; sleep 5; done
  colab download -s "$S" /content/dl/$r/list "$L/list" > /dev/null
  colab download -s "$S" /content/dl/$r/sha "$L/sha" > /dev/null
  while read -r p; do colab download -s "$S" "/content/dl/$r/$p" "$L/$p" > /dev/null; done < "$L/list"
  D=$R/checkpoints/ladder/$r; mkdir -p "$D"
  cat $(sed "s#^#$L/#" "$L/list") > "$D/final.pt"
  if [ "$(sha256sum "$D/final.pt" | cut -d' ' -f1)" = "$(cat "$L/sha")" ]; then echo "$r ok $(cat "$L/sha")"; else echo "$r SHA MISMATCH"; exit 1; fi
  printf '%s\n' "rm -rf /content/dl/$r" 'exit' | timeout 60 colab console -s "$S" > /dev/null 2>&1
done
