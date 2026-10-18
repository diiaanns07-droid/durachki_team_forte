"""Интеграция (A03, облако): слить ветки агентов в порядке ТЗ, конфликты — по владению папками.

Запуск из корня ветки claude/integration-cloud после `git fetch origin`:  python scripts/integration_merge.py [A04 A05 ...]
"""
import subprocess, sys, json

ORDER = [("A01", "claude/gifted-bardeen-i404eh"), ("A04", "claude/zealous-keller-occtqr"),
         ("A05", "claude/hopeful-hypatia-csegbf"), ("A06", "claude/wizardly-pascal-w4vt3o"),
         ("A07", "claude/nice-gates-fjt926"), ("A08", "claude/practical-cerf-0dm6cc"),
         ("A13", "claude/vigilant-lovelace-ssecv4"), ("A09", "claude/wonderful-pasteur-wbu36i"),
         ("A10", "claude/upbeat-ride-3bylsq"), ("A11", "claude/inspiring-turing-2fm7rq"),
         ("A12", "claude/serene-bell-hycc21"), ("A03", "claude/elegant-knuth-yszqkw"),
         ("A02", "claude/laughing-faraday-xe89m9"), ("A14", "claude/compassionate-wright-9gha0v"),
         ("A15", "claude/busy-pasteur-y15o46"), ("A16", "claude/wizardly-galileo-hg6mhp"),
         ("A17", "claude/pensive-wright-5l5q0v"), ("L1", "data/labels")]
OWN = {"A02": ["backend/api/"], "A03": ["backend/pipeline/"], "A04": ["ml/ingest/"], "A05": ["ml/detect/"],
       "A06": ["ml/track/"], "A07": ["ml/pose/"], "A08": ["ml/scene/"], "A09": ["ml/rules/"],
       "A10": ["ml/temporal/"], "A11": ["ml/vlm/"], "A12": ["ml/fusion/"], "A13": ["norms/"],
       "A14": ["copilot/"], "A15": ["frontend/"], "A16": ["eval/"], "A17": ["deploy/"],
       "A18": ["docs/", "pitch/"], "L1": ["tools/dataprep/", "docs/L1_LOCAL_SETUP.md"],
       "A01": ["contracts/", "testkit/", ".github/", "scripts/"]}


def owner(path):
    for agent, prefixes in OWN.items():
        if any(path == p or path.startswith(p) for p in prefixes):
            return agent
    return "A01" if "/" not in path else None  # корневые файлы — A01


def sh(*args, check=True):
    r = subprocess.run(args, capture_output=True, text=True)
    if check and r.returncode:
        raise SystemExit(f"FAILED {' '.join(args)}\n{r.stdout}\n{r.stderr}")
    return r


log = []
only = sys.argv[1:]  # опционально: подмножество агентов
for agent, branch in ORDER:
    if only and agent not in only:
        continue
    ref = f"origin/{branch}"
    r = sh("git", "merge", "--no-ff", "--allow-unrelated-histories", "--no-edit", "-m",
           f"Merge [{agent}] {branch} into integration", ref, check=False)
    entry = {"agent": agent, "branch": branch, "head": sh("git", "rev-parse", "--short", ref).stdout.strip(),
             "conflicts": []}
    if r.returncode:
        unmerged = sh("git", "diff", "--name-only", "--diff-filter=U").stdout.split()
        if not unmerged and "Already up to date" not in r.stdout:
            print(r.stdout, r.stderr)
            raise SystemExit(f"merge {agent} failed without conflicts")
        for path in unmerged:
            own = owner(path)
            side = "theirs" if own == agent else "ours"
            st = sh("git", "status", "--porcelain", "--", path).stdout[:2]
            # delete/modify: если выбранной стороны нет — удаляем
            res = sh("git", "checkout", f"--{side}", "--", path, check=False)
            if res.returncode:
                sh("git", "rm", "-q", "--", path, check=False)
                how = f"{side} (нет файла → удалён)"
            else:
                sh("git", "add", "--", path)
                how = side
            entry["conflicts"].append({"path": path, "status": st, "owner": own, "taken": how})
        sh("git", "commit", "--no-edit", "-q")
    log.append(entry)
    print(f"{agent:4} {branch:40} {entry['head']}  conflicts={len(entry['conflicts'])}")
    for c in entry["conflicts"]:
        print(f"      {c['status']} {c['path']}  owner={c['owner']} → {c['taken']}")
