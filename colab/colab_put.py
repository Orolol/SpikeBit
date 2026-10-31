"""Chunked upload to a colab-cli session (Jupyter contents API, chunk 1..n, -1 last).

`colab upload` fails (HTTP 500) above a few tens of MB; this sends CHUNK_MB pieces. Run it with
the colab-cli tool's own python, then check the sha256 on the VM:

    ~/.local/share/uv/tools/google-colab-cli/bin/python colab/colab_put.py SESSION LOCAL REMOTE [CHUNK_MB]
"""
import base64, os, sys, time

from colab_cli.common import state
from colab_cli.contents import ContentsClient

session, local, remote = sys.argv[1:4]
chunk_mb = float(sys.argv[4]) if len(sys.argv) > 4 else 8
c = ContentsClient(state.get_session(state.resolve_session(session)))
size, step = os.path.getsize(local), int(chunk_mb * (1 << 20))
n = max(1, -(-size // step))
t0 = time.time()
with open(local, "rb") as f:
    for i in range(n):
        data = f.read(step)
        idx = -1 if i == n - 1 and n > 1 else i + 1
        payload = {"name": remote.split("/")[-1], "path": remote, "type": "file", "format": "base64",
                   "content": base64.b64encode(data).decode("ascii"), "chunk": idx}
        for attempt in range(4):
            try:
                c._request("PUT", remote, json_data=payload)
                break
            except Exception as e:
                if attempt == 3:
                    raise
                print(f"chunk {i + 1}/{n} retry: {e}", flush=True)
                time.sleep(2 * (attempt + 1))
mib = size / (1 << 20)
print(f"{mib:.0f}/{mib:.0f} MiB  {mib / (time.time() - t0):.1f} MiB/s", flush=True)
