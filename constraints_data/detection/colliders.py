# ==============================================================================
# COLLIDERS: LHC Run 3 (13.6 TeV) & LEP
# ==============================================================================
import numpy as np
# 1. Mono-photon / Mono-jet (Rayleigh)
# Source: LHC Run 3 (13.6 TeV) [31]; LEP 
# Operator: Rayleigh (Dim-7)
# Scaling: y = (1/C)**(1/3)
# Notes: High EFT validity required. Limits plateau at m_dm > sqrt(s)/2 ~ 6.8 TeV.
# LEP provides flat sensitivity down to ultra-low masses. LHC constraints drop off 
# sharply as m_dm approaches kinematic beam thresholds.
m_dm_vals_Coll_Ray = np.array(1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0, 1000.0, 5000.0)
C_raw_Coll_Ray = np.array(3.0e-11, 3.0e-11, 3.0e-11, 3.3e-11, 4.1e-11, 9.3e-11, 1.6e-9, 7.5e-4)
y_vals_Coll_Ray = ( 1 / C_raw_Coll_Ray) ** (1/3)

# 2. Collider Dipole Limits
# Source: LHC Run 3 (13.6 TeV) [31]; LEP 
# Operator: Dirac Magnetic Dipole
# Scaling: y = Lambda / sqrt(C)
# Notes: Validated truncation ensures Q_tr < Lambda.
m_dm_vals_Coll_MD = [1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0, 1000.0, 5000.0]
y_vals_Coll_MD = [1.5e4, 1.5e4, 1.5e4, 1.4e4, 1.3e4, 9.1e3, 2.5e3, 4.2e1]

# 3. Collider Anapole Limits
# Source: LEP Z-decay + Mono-photon (95% CL); LHC ATLAS Run 3 Mono-jet (95% CL)
# Operator: Dirac Anapole
# Scaling: y = Lambda / sqrt(C) (dimension-6)
# Notes: LEP provides flat sensitivity down to ultra-low masses.
# LHC constraints drop off as m_dm approaches kinematic thresholds.

# LEP Z-decay + Mono-photon
m_dm_vals_Coll_A_LEP = np.array([1.0e-6, 1.0e-3, 1.0e-1, 1.0e0, 1.0e1, 4.0e1, 4.5e1, 5.0e1, 9.0e1])
y_vals_Coll_A_LEP = np.array([4.8e2, 4.8e2, 4.8e2, 4.7e2, 4.5e2, 3.9e2, 3.5e2, 1.2e2, 1.0e1])

# LHC ATLAS Run 3 Mono-jet
m_dm_vals_Coll_A_LHC = np.array([1.0e-6, 1.0e-3, 1.0e-1, 1.0e0, 1.0e1, 1.0e2, 5.0e2, 1.0e3, 2.0e3])
y_vals_Coll_A_LHC = np.array([1.2e3, 1.2e3, 1.2e3, 1.15e3, 1.1e3, 9.5e2, 4.0e2, 1.2e2, 1.5e1])

# Combined collider anapole limit (envelope of LEP + LHC)
m_dm_vals_Coll_A = np.array([1.0e-6, 1.0e-3, 1.0e-1, 1.0e0, 1.0e1, 4.0e1, 4.5e1, 5.0e1, 9.0e1, 1.0e2, 5.0e2, 1.0e3, 2.0e3])
y_vals_Coll_A = np.array([1.2e3, 1.2e3, 1.2e3, 1.15e3, 1.1e3, 9.5e2, 9.5e2, 9.5e2, 9.5e2, 9.5e2, 4.0e2, 1.2e2, 1.5e1])