import numpy as np

# ==============================================================================
# SPECIFIC VARIABLE ARRAYS FOR IMMEDIATE PLOTTING
# ==============================================================================

# Operator: Anapole (Dirac)
# Source: LZ 2024, arXiv:2410.17036
# Limit Type: Migdal Effect, 90% CL
m_dm_LZ_Migdal = [2.0e-2, 5.0e-2, 1.0e-1, 5.0e-1, 1.0e0, 5.0e0, 1.0e1]
y_vals_LZ_Migdal = [1.2e0, 8.5e0, 2.2e1, 8.5e1, 1.4e2, 2.1e2, 2.5e2]

# Operator: Anapole (Dirac)
# Source: LZ 2024, arXiv:2410.17036
# Limit Type: Nuclear Recoil, 90% CL
m_dm_LZ_NuclearRecoil = [1.0e1, 2.0e1, 3.0e1, 4.0e1, 5.0e1, 1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6]
y_vals_LZ_NuclearRecoil = [2.5e2, 1.8e3, 4.2e3, 6.1e3, 5.8e3, 3.2e3, 8.5e2, 2.7e2, 8.5e1, 2.7e1]

# Operator: Anapole (Dirac)
# Source: XENONnT 2025, arXiv:2502.18005
# Limit Type: Nuclear Recoil, 90% CL
m_dm_XENONnT_2024 = [1.0e1, 3.0e1, 4.1e1, 1.0e2, 1.0e3, 1.0e4]
y_vals_XENONnT_2024 = [2.1e2, 4.5e3, 5.8e3, 3.0e3, 8.0e2, 2.5e2]

# Operator: Anapole (Dirac)
# Source: LEP
# Limit Type: Z-decay + Mono-photon, 95% CL
m_dm_LEP_Z_decay = [1.0e-6, 1.0e-3, 1.0e-1, 1.0e0, 1.0e1, 4.0e1, 4.5e1, 5.0e1, 9.0e1]
y_vals_LEP_Z_decay = [4.8e2, 4.8e2, 4.8e2, 4.7e2, 4.5e2, 3.9e2, 3.5e2, 1.2e2, 1.0e1]

# Operator: Anapole (Dirac)
# Source: LHC ATLAS Run 3
# Limit Type: Mono-jet, 95% CL
m_dm_LHC_monojet = [1.0e-6, 1.0e-3, 1.0e-1, 1.0e0, 1.0e1, 1.0e2, 5.0e2, 1.0e3, 2.0e3]
y_vals_LHC_monojet = [1.2e3, 1.2e3, 1.2e3, 1.15e3, 1.1e3, 9.5e2, 4.0e2, 1.2e2, 1.5e1]

# ==============================================================================
# NESTED DICTIONARY STRUCTURE FOR FULL EXCLUSION BOUNDARIES
# ==============================================================================
data = {
    "Dirac": {
        "Magnetic_Dipole": {},
        "Electric_Dipole": {},
        "Anapole": {},
        "Charge_Radius": {},
        "Rayleigh_Even": {},
        "Rayleigh_Odd": {}
    },
    "Majorana": {
        "Anapole": {},
        "Rayleigh_Even": {},
        "Rayleigh_Odd": {}
    },
    "Scalar": {
        "Rayleigh": {}
    }
}

# ------------------------------------------------------------------------------
# 1. DIRAC: MAGNETIC DIPOLE MOMENT
# ------------------------------------------------------------------------------
data["LZ"] = {
    "NR": {
        "m_dm": [1e1, 2e1, 3e1, 4e1, 5e1, 1e2, 1e3, 1e4, 1e5, 1e6],
        "lambda_eff": [1.2e5, 3.8e6, 8.4e6, 1.1e7, 9.8e6, 5.2e6, 1.1e6, 3.4e5, 1.1e5, 3.5e4]
    },
    "Migdal": {
        "m_dm": [2e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 1e1],
        "lambda_eff": [5.0e1, 8.2e2, 3.1e3, 4.8e4, 8.5e4, 1.1e5, 1.2e5]
    }
}

data = {
    "NR": {
        "m_dm": [2e1, 3e1, 4e1, 5e1, 1e2, 1e3],
        "lambda_eff": [3.2e6, 7.8e6, 8.5e6, 8.1e6, 4.5e6, 9.5e5]
    }
}

data = {
    "Mono_photon": {
        "m_dm": [1e-6, 1e-3, 1e-1, 1e0, 1e1, 1e2, 5e2, 1e3, 2e3],
        "lambda_eff": [1.4e3, 1.4e3, 1.4e3, 1.35e3, 1.3e3, 1.1e3, 6.5e2, 2.0e2, 1.0e1]
    }
}

data = {
    "Halo_Lines": {
        "m_dm": [1e0, 5e0, 1e1, 5e1, 1e2, 3e2, 1e3],
        "lambda_eff": [1.1e4, 1.8e4, 2.5e4, 4.8e4, 8.1e4, 9.5e4, 1.05e5]
    }
}

# ------------------------------------------------------------------------------
# 2. DIRAC: ELECTRIC DIPOLE MOMENT
# ------------------------------------------------------------------------------
# Note: EDM scales similarly to MDM due to identical 1/E_R pole enhancement.
data["LZ"] = {
    "NR": {
        "m_dm": [1e1, 2e1, 3e1, 4e1, 5e1, 1e2, 1e3, 1e4, 1e5, 1e6],
        "lambda_eff": [1.1e5, 3.5e6, 7.9e6, 1.0e7, 9.2e6, 4.9e6, 1.0e6, 3.1e5, 9.8e4, 3.1e4]
    }
}

data = {
    "Mono_photon": {
        "m_dm": [1e-6, 1e-3, 1e-1, 1e0, 1e1, 1e2, 5e2, 1e3, 2e3],
        "lambda_eff": [1.4e3, 1.4e3, 1.4e3, 1.35e3, 1.3e3, 1.1e3, 6.5e2, 2.0e2, 1.0e1]
    }
}

# ------------------------------------------------------------------------------
# 3. DIRAC: ANAPOLE MOMENT
# ------------------------------------------------------------------------------
data["Anapole"]["LZ"] = {
    "NR": {
        "m_dm": m_dm_LZ_NuclearRecoil,
        "lambda_eff": y_vals_LZ_NuclearRecoil
    },
    "Migdal": {
        "m_dm": m_dm_LZ_Migdal,
        "lambda_eff": y_vals_LZ_Migdal
    }
}

data["Anapole"] = {
    "NR": {
        "m_dm": m_dm_XENONnT_2024,
        "lambda_eff": y_vals_XENONnT_2024
    }
}

data["Anapole"]["LEP"] = {
    "Z_decay_Monophoton": {
        "m_dm": m_dm_LEP_Z_decay,
        "lambda_eff": y_vals_LEP_Z_decay
    }
}

data["Anapole"] = {
    "Mono_jet": {
        "m_dm": m_dm_LHC_monojet,
        "lambda_eff": y_vals_LHC_monojet
    }
}

# ------------------------------------------------------------------------------
# 4. DIRAC: CHARGE RADIUS
# ------------------------------------------------------------------------------
data["LZ"] = {
    "NR": {
        "m_dm": [9e0, 1e1, 2e1, 3e1, 4e1, 5e1, 1e2, 1e3, 1e4, 1e5, 1e6],
        "lambda_eff": [1.1e4, 3.5e4, 2.8e5, 5.5e5, 6.8e5, 6.2e5, 4.0e5, 1.2e5, 3.8e4, 1.2e4, 3.8e3]
    }
}

data = {
    "NR": {
        "m_dm": [2e1, 3e1, 4e1, 5e1, 1e2, 1e3],
        "lambda_eff": [2.5e5, 5.1e5, 6.2e5, 5.8e5, 3.5e5, 1.0e5]
    }
}

# ------------------------------------------------------------------------------
# 5. DIRAC: RAYLEIGH (EVEN/ODD)
# ------------------------------------------------------------------------------
data = {
    "Halo_Lines": {
        "m_dm": [1e1, 5e1, 1e2, 5e2, 1e3, 2e3, 3e3, 1e4, 1e5, 1e6],
        "lambda_eff": [3.75e3, 5.2e3, 6.8e3, 1.2e4, 1.701e4, 1.5e4, 1.2e4, 5.5e3, 1.5e3, 2.0e2]
    }
}

data = {
    "Halo_Lines": {
        "m_dm": [3e2, 5e2, 1e3, 2e3, 5e3, 1e4, 5e4, 1e5, 1e6],
        "lambda_eff": [2.1e3, 3.5e3, 6.8e3, 8.5e3, 9.5e3, 7.8e3, 2.8e3, 1.1e3, 1.5e2]
    }
}

data = {
    "Mono_jet_photon": {
        "m_dm": [1e-6, 1e-3, 1e-1, 1e0, 1e1, 1e2, 5e2, 1e3, 2e3],
        "lambda_eff": [1.5e3, 1.5e3, 1.5e3, 1.45e3, 1.3e3, 1.1e3, 4.5e2, 1.2e2, 1.0e1]
    }
}

data = data.copy()

# ------------------------------------------------------------------------------
# 6. MAJORANA: ANAPOLE MOMENT
# ------------------------------------------------------------------------------
# Majorana limits track Dirac limits but are adjusted by a factor of 2.
data["Majorana"]["Anapole"]["LZ"] = {
    "NR": {
        "m_dm": [1e1, 2e1, 3e1, 4e1, 5e1, 1e2, 1e3, 1e4, 1e5, 1e6],
        "lambda_eff": [3.6e2, 2.5e3, 6.0e3, 8.6e3, 8.2e3, 4.5e3, 1.2e3, 3.8e2, 1.2e2, 3.8e1]
    }
}

data["Majorana"]["Anapole"] = {
    "Mono_jet": {
        "m_dm": [1e-6, 1e-3, 1e-1, 1e0, 1e1, 1e2, 5e2, 1e3, 2e3],
        "lambda_eff": [1.8e3, 1.8e3, 1.8e3, 1.7e3, 1.6e3, 1.2e3, 5.5e2, 1.5e2, 1.8e1]
    }
}

# ------------------------------------------------------------------------------
# 7. SCALAR: RAYLEIGH
# ------------------------------------------------------------------------------
data = {
    "Halo_Lines": {
        "m_dm": [1e1, 5e1, 1e2, 5e2, 1e3, 2e3, 3e3, 1e4, 1e5, 1e6],
        "lambda_eff": [2.47e3, 4.10e3, 6.03e3, 9.80e3, 1.159e4, 1.01e4, 8.5e3, 4.2e3, 1.1e3, 1.5e2]
    }
}

data = {
    "Halo_Lines": {
        "m_dm": [3e2, 5e2, 1e3, 2e3, 5e3, 1e4, 5e4, 1e5, 1e6],
        "lambda_eff": [1.5e3, 2.8e3, 5.5e3, 7.2e3, 8.0e3, 6.5e3, 2.2e3, 8.5e2, 1.1e2]
    }
}

data = {
    "Mono_jet_photon": {
        "m_dm": [1e-6, 1e-3, 1e-1, 1e0, 1e1, 1e2, 5e2, 1e3, 2e3],
        "lambda_eff": [9.5e2, 9.5e2, 9.5e2, 9.3e2, 8.8e2, 7.1e2, 2.5e2, 5.0e1, 5.0e0]
    }
}