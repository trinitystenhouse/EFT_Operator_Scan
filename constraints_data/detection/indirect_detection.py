# ==============================================================================
# INDIRECT DETECTION: Fermi-LAT & CTA (2024-2026)
# ==============================================================================
import numpy as np
# 1. Dirac/Majorana Rayleigh (Even/Odd)
# Source: Fermi-LAT 15-yr Pass 8 [7, 29]; CTA Projections [8]
# Operator: Dirac/Majorana Rayleigh
# Scaling: y = (1/C)**(1/3)
# Notes: Dimension-7 operator. Derived from gamma-ray line searches. 
# Applies scaling transformation from raw C coefficient. Transition from Fermi-LAT 
# to CTA dominance occurs around 1 TeV.
m_dm_vals_ID_Rayleigh = np.array(1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 100000.0)
C_raw_Rayleigh = np.array(1.3e-8, 1.8e-9, 2.9e-10, 2.5e-11, 7.1e-12, 7.5e-13, 3.6e-13, 5.6e-14, 3.3e-14, 2.3e-14)
y_vals_ID_Rayleigh = (1 / C_raw_Rayleigh) ** (1/3)

# 2. Scalar Rayleigh
# Source: Scalar Rayleigh DM 
# Operator: Scalar Rayleigh
# Scaling: y = np.sqrt(Lambda_raw)
# Notes: Dimension-6 operator. Monotonically increasing sensitivity from CTA.
m_dm_vals_ID_SR = np.array(1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)
Lambda_raw_ID_SR = np.array(7.2e5, 4.4e6, 4.0e7, 3.2e8, 1.0e9, 1.6e9)
y_vals_ID_SR = np.sqrt(Lambda_raw_ID_SR)