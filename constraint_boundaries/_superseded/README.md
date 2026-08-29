# Superseded CMB elastic boundaries

Moved 2026-08-27.  These three grids were computed with
`CMB_U_MAX[4] = 3.0e-14 cm^2/GeV`, attributed to Boddy & Gluscevic (2018),
arXiv:1801.08609, Table 1.

That paper constrains dark matter-PROTON scattering, parametrised in powers of
the relative velocity, using Planck 2015.  It is not a DM-photon bound.

Wilkinson, Boehm & Lesgourgues (2014), arXiv:1309.7588, publish DM-photon
bounds for a constant cross section and for sigma ~ T^2 only.  There is no
published DM-photon bound for the T^4 scaling these operators require, so
`cmb_constraints.py` now returns an empty boundary for them, as it has always
done for T^6.

Do not restore these files.  If a T^4 DM-photon bound is published, regenerate
from it.
