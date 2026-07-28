# ==============================================================================
# THEORETICAL BOUNDARIES: Deconvolution Spectral-Modification Limit
# ==============================================================================
# Source: Deconvolution Spectral-Modification Limit 
# Operator: Affects all Indirect Detection photon channels
# Scaling: y = Lambda (GeV)
# Notes: Affects ID for m_chi > 1e4 GeV due to EBL/CMB cascading. The limits 
# artificially flatten without cascade modeling.
m_dm_vals_deconv = np.logspace(3, 6, 10).tolist()
y_vals_deconv = [5.0e3] * 10  # Artificial ceiling above which ID limits flatten