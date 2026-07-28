# ==============================================================================
# DIRECT DETECTION: XENONnT / LZ / PANDA-X 4T (2024-2026)
# Reference Benchmark: Ibarra, Reichard, & Tomar (2024) [arXiv:2408.15760]
# ==============================================================================

import numpy as np 

# 1. Dirac Magnetic Dipole (O_MD)
# Source: Ibarra, Reichard, & Tomar (2024) [arXiv:2408.15760]; LZ/XENONnT [28]
# Operator: Dirac Magnetic Dipole
# Scaling: y = Lambda / sqrt(C)
# Notes: Includes Migdal effect for low mass (1e-3 to 1e-1 GeV) and NR for high mass.
# Converted from mu_chi constraints to Lambda. Peak sensitivity limited by 8B neutrino fog.
m_dm_vals_DD_MD = np.array(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)
y_vals_DD_MD = np.array(2.1e3, 5.4e4, 1.8e6, 3.2e7, 4.1e8, 2.5e9, 3.8e9, 1.9e9, 4.2e8, 1.1e8)

# 2. Dirac Electric Dipole (O_ED)
# Source: Ibarra, Reichard, & Tomar (2024) [arXiv:2408.15760]; PandaX-4T 
# Operator: Dirac Electric Dipole
# Scaling: y = Lambda / sqrt(C)
# Notes: 1/q^2 enhanced at low energies, benefiting massively from S2-only analyses.
# Translated from e*cm limits.
m_dm_vals_DD_ED = np.array(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)
y_vals_DD_ED = np.array(1.5e5, 2.3e7, 4.5e9, 8.1e10, 5.4e11, 3.2e12, 4.1e12, 1.8e12, 3.5e11, 8.9e10)

# 3. Dirac/Majorana Anapole (O_A)
# Source: LZ 2024 [arXiv:2410.17036]; XENONnT 2025 [arXiv:2502.18005]
# Operator: Dirac/Majorana Anapole
# Scaling: y = Lambda / sqrt(C) (dimension-6)
# Notes: Dimension-6 operator. Momentum and velocity suppressed scattering.
# Combined Migdal (low mass) + Nuclear Recoil (high mass) limits.
# LZ 2024 Migdal component (2e-2 to 10 GeV)
m_dm_vals_DD_A_Migdal = np.array([2.0e-2, 5.0e-2, 1.0e-1, 5.0e-1, 1.0e0, 5.0e0, 1.0e1])
y_vals_DD_A_Migdal = np.array([1.2e0, 8.5e0, 2.2e1, 8.5e1, 1.4e2, 2.1e2, 2.5e2])

# LZ 2024 Nuclear Recoil component (10 GeV to 1 PeV)
m_dm_vals_DD_A_NR = np.array([1.0e1, 2.0e1, 3.0e1, 4.0e1, 5.0e1, 1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6])
y_vals_DD_A_NR = np.array([2.5e2, 1.8e3, 4.2e3, 6.1e3, 5.8e3, 3.2e3, 8.5e2, 2.7e2, 8.5e1, 2.7e1])

# Combined LZ limit (envelope of Migdal + NR)
m_dm_vals_DD_A = np.concatenate([m_dm_vals_DD_A_Migdal, m_dm_vals_DD_A_NR[1:]])
y_vals_DD_A = np.concatenate([y_vals_DD_A_Migdal, y_vals_DD_A_NR[1:]])

# XENONnT 2025 Nuclear Recoil (alternative, slightly weaker)
m_dm_vals_DD_A_XENONnT = np.array([1.0e1, 3.0e1, 4.1e1, 1.0e2, 1.0e3, 1.0e4])
y_vals_DD_A_XENONnT = np.array([2.1e2, 4.5e3, 5.8e3, 3.0e3, 8.0e2, 2.5e2])

# 4. Dirac Charge Radius (O_CR)
# Source: PandaX-4T ; LZ 2025
# Operator: Dirac Charge Radius
# Scaling: y = np.sqrt(Lambda_raw)
# Notes: Dimension-6 operator. Coherent SI. 
# Converted from PandaX-4T limit < 1.9e-10 fm^2.
m_dm_vals_DD_CR = np.array(1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)
Lambda_raw_DD_CR = np.array(2.6e5, 7.0e7, 1.4e10, 1.1e13, 6.5e13, 3.8e13, 2.2e12, 9.6e10)
y_vals_DD_CR = np.sqrt(Lambda_raw_DD_CR)