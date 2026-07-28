BASE="python scan_deconvolved_scattering_fit.py \
  --best-fit-type dsph_cl \
  --ann-mass-mode scan \
  --ann-mass-min 100 \
  --ann-mass-max 5000 \
  --n-ann-mass 80 \
  --scatter-mass-min 1e-6 \
  --scatter-mass-max 1 \
  --n-scatter-mass 100 \
  --lambda-min 1e-3 \
  --lambda-max 1e4 \
  --n-lambda 100 \
  --n-ext-bins 30 \
  --ext-energy-max 10000 \
  --min-tau 1e-5 --max-tau 0.3 --cl 0.95"

# dipole_magnetic
$BASE --operator dipole_magnetic --ann-channel WW
$BASE --operator dipole_magnetic --ann-channel bb
$BASE --operator dipole_magnetic --ann-channel tautau

# dipole_electric
$BASE --operator dipole_electric --ann-channel WW
$BASE --operator dipole_electric --ann-channel bb
$BASE --operator dipole_electric --ann-channel tautau

# charge_radius
$BASE --operator charge_radius --ann-channel WW
$BASE --operator charge_radius --ann-channel bb
$BASE --operator charge_radius --ann-channel tautau

# anapole (Majorana-allowed — add --majorana for Majorana DM)
$BASE --operator anapole --ann-channel WW
$BASE --operator anapole --ann-channel bb
$BASE --operator anapole --ann-channel tautau

# rayleigh_even / rayleigh_odd / rayleigh_full (Majorana-allowed)
$BASE --operator rayleigh_even --ann-channel WW
$BASE --operator rayleigh_even --ann-channel bb
$BASE --operator rayleigh_even --ann-channel tautau

$BASE --operator rayleigh_odd --ann-channel WW
$BASE --operator rayleigh_odd --ann-channel bb
$BASE --operator rayleigh_odd --ann-channel tautau

$BASE --operator rayleigh_full --ann-channel WW
$BASE --operator rayleigh_full --ann-channel bb
$BASE --operator rayleigh_full --ann-channel tautau

$BASE --operator scalar_rayleigh --dm-type scalar --ann-channel WW
$BASE --operator scalar_rayleigh --dm-type scalar --ann-channel bb
$BASE --operator scalar_rayleigh --dm-type scalar --ann-channel tautau


# charge_radius
$BASE --operator charge_radius --majorana --ann-channel WW
$BASE --operator charge_radius --majorana --ann-channel bb
$BASE --operator charge_radius --majorana --ann-channel tautau

# anapole (Majorana-allowed — add --majorana for Majorana DM)
$BASE --operator anapole --majorana --ann-channel WW
$BASE --operator anapole --majorana --ann-channel bb
$BASE --operator anapole --majorana --ann-channel tautau

# rayleigh_even / rayleigh_odd / rayleigh_full (Majorana-allowed)
$BASE --operator rayleigh_even --majorana --ann-channel WW
$BASE --operator rayleigh_even --majorana --ann-channel bb
$BASE --operator rayleigh_even --majorana --ann-channel tautau

$BASE --operator rayleigh_odd --majorana --ann-channel WW
$BASE --operator rayleigh_odd --majorana --ann-channel bb
$BASE --operator rayleigh_odd --majorana --ann-channel tautau

$BASE --operator rayleigh_full --majorana --ann-channel WW
$BASE --operator rayleigh_full --majorana --ann-channel bb
$BASE --operator rayleigh_full --majorana --ann-channel tautau
