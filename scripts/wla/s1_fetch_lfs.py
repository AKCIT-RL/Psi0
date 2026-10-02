#!/usr/bin/env python3
"""Substitui ponteiros Git LFS pelo conteúdo real num checkout clonado sem git-lfs (stdlib, protocolo LFS batch).

Uso: s1_fetch_lfs.py <repo_dir> <https git url> [--skip-prefix third_party]
Os arquivos ficam alterados no working tree (o índice guarda o ponteiro): marque com `git update-index --skip-worktree`
(feito aqui) para não sujar commits. Rede necessária -> rodar via sbatch.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request

repo, url = sys.argv[1], sys.argv[2].rstrip("/")
skip = [a for a in sys.argv[3:] if not a.startswith("--")]
skip_prefix = tuple(sys.argv[sys.argv.index("--skip-prefix") + 1:][:1]) if "--skip-prefix" in sys.argv else ()
PTR = re.compile(rb"^version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize (\d+)\n?$")

ptrs = {}
for root, dirs, files in os.walk(repo):
    dirs[:] = [d for d in dirs if d != ".git"]
    rel_root = os.path.relpath(root, repo)
    if skip_prefix and (rel_root + "/").startswith(skip_prefix[0] + "/"):
        continue
    for f in files:
        p = os.path.join(root, f)
        if os.path.islink(p) or os.path.getsize(p) > 200:
            continue
        m = PTR.match(open(p, "rb").read())
        if m:
            ptrs.setdefault((m.group(1).decode(), int(m.group(2))), []).append(p)
print(f"{sum(len(v) for v in ptrs.values())} ponteiros, {len(ptrs)} objetos, {sum(s for _, s in ptrs) / 1e6:.1f} MB", flush=True)

def post(objs):
    body = json.dumps({"operation": "download", "transfers": ["basic"], "objects": [{"oid": o, "size": s} for o, s in objs]}).encode()
    req = urllib.request.Request(url + "/info/lfs/objects/batch", data=body, method="POST",
                                 headers={"Accept": "application/vnd.git-lfs+json", "Content-Type": "application/vnd.git-lfs+json"})
    return json.load(urllib.request.urlopen(req, timeout=60))["objects"]

keys = list(ptrs)
ok = bad = 0
for i in range(0, len(keys), 100):
    for ob in post(keys[i:i + 100]):
        key = (ob["oid"], ob["size"])
        act = ob.get("actions", {}).get("download")
        if not act:
            print("sem download:", ob["oid"][:12], ob.get("error"), flush=True); bad += 1; continue
        req = urllib.request.Request(act["href"], headers=act.get("header", {}))
        data = urllib.request.urlopen(req, timeout=300).read()
        if hashlib.sha256(data).hexdigest() != ob["oid"]:
            print("sha256 divergente:", ob["oid"][:12], flush=True); bad += 1; continue
        for p in ptrs[key]:
            open(p, "wb").write(data)
            rel = os.path.relpath(p, repo)
            subprocess.run(["git", "-C", repo, "update-index", "--skip-worktree", rel], check=False)
        ok += 1
print(f"objetos baixados: {ok}, falhas: {bad}")
sys.exit(1 if bad else 0)
