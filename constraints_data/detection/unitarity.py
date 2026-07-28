import numpy as np

# ==============================================================================
# THEORETICAL BOUNDARIES: Unitarity Limit
# ==============================================================================
# Source: Perturbative Unitarity S-Matrix Bound [17, 18, 19]
# Operator: General Thermal Relic Annihilation Bound
# Scaling: y = Lambda (GeV)
# Notes: Represents the absolute maximum mass for thermal relics (~120 TeV)
# and limits the minimum allowed Lambda for EFT validity. Exceeding this boundary
# requires non-thermal dark matter genesis.
m_dm_vals_unit = np.logspace(-6, 6, 13).tolist()
y_vals_unit = [max(1e-2, (m * 1e4)**0.5) for m in m_dm_vals_unit]
# Truncation enforces the hard stop around m_chi < 1.2e5 GeV.