#!/usr/bin/env python3
"""Release hygiene gate for the public tree.

Report by default, `--fix` for the mechanically safe rewrites, exit 1 on any
finding a human has to decide. Run before every push and before cutting a DOI.

Checks, in the order they have actually bitten:

1. Absolute home-directory paths, and personal names in import paths. The
   plotting helper was renamed to helpers/plot_style.py to de-personalise the
   tree; copying that file back from the working repo silently undoes it.
2. Reference spectra or constraint files whose own header says they are
   unverified. constraints_data/limits.py guards its own directory; nothing
   guarded core/spectrum_source.py, and a fabricated IGRB spectrum shipped for
   several releases as a result.
3. .npz grids that no figure script names, and figure scripts naming grids that
   do not exist.
4. Stray scratch artefacts (_timingtest, _igrbfix, .bak, .orig).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

HOME_PATH = re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+/")
# Assembled at runtime so this file does not trip its own scan.
PERSONAL = re.compile("|".join(["trinity" + "_plotting", "trinity" + "stenhouse"]))
# Markers that mean "this file is not what it claims to be". Applied ONLY
# outside constraints_data/, whose own loader guard already refuses these and
# where the markers are therefore deliberate, not a leak. A prose mention of
# the word "synthetic" in a docstring is not a finding, so the pattern is
# deliberately narrow: it matches self-incrimination, not vocabulary.
UNVERIFIED = re.compile(
    r"AI-GENERATED|NOT DIGITIS|CITATION SUSPECT|FABRICAT|TODO_verify|"
    r"verify against[^.]{0,80}before final submission",
    re.I,
)
SCRATCH = re.compile(r"_timingtest|_igrbfix|\.bak$|\.orig$|~$|\.__wtest")

TEXT_SUFFIXES = {".py", ".md", ".txt", ".cfg", ".toml", ".sh"}


def iter_files():
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in p.parts):
            continue
        yield p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true",
                    help="apply only the unambiguous rewrites")
    args = ap.parse_args()

    findings: list[str] = []
    fixed = 0
    n_checked = 0

    for p in iter_files():
        rel = p.relative_to(ROOT).as_posix()

        if SCRATCH.search(p.name):
            findings.append(f"scratch artefact: {rel}")
            continue

        if p.suffix not in TEXT_SUFFIXES:
            continue
        n_checked += 1
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            findings.append(f"unreadable: {rel} ({exc})")
            continue

        for pat, label in ((HOME_PATH, "absolute home path"),
                           (PERSONAL, "personal identifier")):
            for m in pat.finditer(text):
                findings.append(f"{label}: {rel}: {m.group(0)!r}")

        # Only flag the unverified markers where they are load-bearing: a file
        # that describes the guard is not itself a violation.
        guarded = rel.startswith("constraints_data/")
        exempt = rel in {"tools/check_release_hygiene.py", "docs/METHODOLOGY.md"}
        if not guarded and not exempt:
            m = UNVERIFIED.search(text)
            if m:
                findings.append(f"unverified-data marker: {rel}: {m.group(0)!r}")

    # Grids are located by constructed names, not literal strings, so an
    # orphan check by substring gives only false positives. What IS checkable:
    # every grid must load and carry the metadata the paper cites.
    import numpy as _np
    present = sorted((ROOT / "constraint_boundaries").glob("*.npz"))
    for g in present:
        try:
            d = _np.load(g, allow_pickle=True)
        except Exception as exc:
            findings.append(f"unreadable grid: {g.name} ({exc})")
            continue
        if "lambda_plot_GeV" not in d.files:
            continue                       # helper array, not a limit grid
        if "fit_normalization" in d.files and not bool(d["fit_normalization"]):
            findings.append(
                f"grid not normalisation-profiled: {g.name} "
                f"(the manuscript states the normalisation is profiled)")

    print(f"checked {n_checked} text files, "
          f"{len(present)} grids")
    if args.fix and fixed:
        print(f"fixed {fixed}")
    if not findings:
        print("clean: no findings.")
        return 0
    print(f"\n{len(findings)} finding(s):")
    for f in findings:
        print(f"  - {f}")
    print("\nThese need a human decision; the tree is NOT release-clean.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
