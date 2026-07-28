"""Unit tests for core.dsph_limits_multi and the Boddy velocity-dependent tables.

Run from the Totani_Scattering directory:
    python -m pytest tests/test_dsph_limits_multi.py -q
or standalone:
    python tests/test_dsph_limits_multi.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.dsph_limits_multi import (  # noqa: E402
    dsph_upper_limit,
    load_dsph_limit_table,
    normalise_limit_source,
    is_velocity_dependent,
    available_dsph_channels,
)


def test_out_of_range_returns_nan_not_inf():
    # Below the table minimum -> nan (the bug that previously gave 0 tension).
    assert math.isnan(dsph_upper_limit(1.0, "bb", "hoof"))
    assert math.isnan(dsph_upper_limit(1e6, "bb", "hoof"))
    # extrapolate=True gives a finite number instead.
    assert np.isfinite(dsph_upper_limit(1e6, "bb", "hoof", extrapolate=True))


def test_pwave_factor_is_exact_ratio():
    # p-wave halo-comparable limit must be the s-wave limit times the velocity
    # factor u_h/u_d ACTUALLY USED to build the table.
    #
    # The publication-grade tables are built from measured J-weighted moments
    # (u_dwarf_pwave = 9.711e-9 from the Boddy 2019 posteriors at 0.5 deg;
    # u_halo_pwave = 2.013e-6 from the rho^2-weighted Jeans solve with
    # baryons), giving u_h/u_d = 207.34 -- and that factor is stamped in the
    # table header ("rescaling s-wave source 'hoof' by velocity factor = ...").
    # Read the factor from the header so the test tracks the provenance
    # rather than a hard-coded number.
    import re
    table = (Path(__file__).resolve().parent.parent / "data"
             / "boddy2019_pwave_bb.txt")
    header = table.read_text().split("\n", 6)
    factor = None
    for line in header:
        mm = re.search(r"velocity factor = ([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)", line)
        if mm:
            factor = float(mm.group(1))
            break
    assert factor is not None, "provenance header missing from p-wave table"
    # sanity-band: the measured moments give O(200), not the 625 placeholder
    assert 100.0 < factor < 400.0, factor
    for m in (200.0, 500.0, 768.0, 1000.0):
        s = dsph_upper_limit(m, "bb", "hoof")
        p = dsph_upper_limit(m, "bb", "boddy_pwave")
        assert math.isclose(p / s, factor, rel_tol=1e-3), (m, p / s)


def test_sommerfeld_strengthens_limit():
    # Coulomb Sommerfeld should make the dwarf limit STRONGER (smaller).
    s = dsph_upper_limit(768.0, "bb", "hoof")
    so = dsph_upper_limit(768.0, "bb", "boddy_somm")
    assert so < s


def test_tension_directions_match_physics():
    # bb best-fit point: s-wave tension ~5x, p-wave << 1, Sommerfeld >> 1.
    m, sv = 768.0, 9.8e-25
    t_s = sv / dsph_upper_limit(m, "bb", "hoof")
    t_p = sv / dsph_upper_limit(m, "bb", "boddy_pwave")
    t_so = sv / dsph_upper_limit(m, "bb", "boddy_somm")
    assert 3.0 < t_s < 7.0
    assert t_p < 0.1
    assert t_so > 10.0


def test_source_normalisation_and_flags():
    assert normalise_limit_source("boddy_p") == "boddy_pwave"
    assert normalise_limit_source("sommerfeld") == "boddy_somm"
    assert is_velocity_dependent("boddy_pwave")
    assert not is_velocity_dependent("hoof")
    assert set(available_dsph_channels("boddy_pwave")) == {"WW", "bb", "tautau"}


def test_tables_are_monotonic_in_mass():
    for src in ("boddy_swave", "boddy_pwave", "boddy_dwave", "boddy_somm"):
        mass, _ = load_dsph_limit_table("bb", src)
        assert np.all(np.diff(mass) > 0)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
