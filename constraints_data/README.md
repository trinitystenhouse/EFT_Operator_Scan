Place digitized literature constraints here as plain text files with two columns:

`mchi_GeV  lambda_plot_GeV`

The central Python registry is `limits.py`.  It exposes `LIMITS_BY_OPERATOR`,
where each operator has one dictionary containing `operator_type`,
`fermion_type`, `dm_type`, and a list of limit dictionaries.  Each limit entry
contains `constraint_type`, `color`, `linestyle`, `path`, and loaded `data`
arrays.  Use `get_operator_limits(...)` or `find_limits(...)` to filter by
operator type, fermion type, or constraint type.

The plotting script expects filenames such as:

- `lux2016_dipole_magnetic.txt`
- `xenon1t2018_dipole_magnetic.txt`
- `lz_projected_dipole_magnetic.txt`
- `fermi_dsphs2016_dipole_electric.txt`
- `hess_gc2013_dipole_electric.txt`
- `lux2016_charge_radius.txt`
- `xenon1t2018_charge_radius.txt`
- `lz_projected_charge_radius.txt`
- `fermi_dsphs2016_anapole.txt`
- `fermi_gh2013_anapole.txt`
- `fermi_5p8yr2013_rayleigh_even.txt`
- `hess_gc2013_rayleigh_even.txt`
- `fermi_5p8yr2013_rayleigh_odd.txt`
- `hess_gc2013_rayleigh_odd.txt`

These y-values should already be in the paper-style convention used on the plots:

- dim-5 / dim-6 operators: `Lambda / C^(1/2)`
- Rayleigh-type operators: `Lambda / C^(1/3)`

Example file contents:

```text
# mchi_GeV lambda_plot_GeV
1.0e1  3.2e1
3.0e1  4.1e1
1.0e2  5.0e1
3.0e2  4.8e1
1.0e3  3.9e1
```
