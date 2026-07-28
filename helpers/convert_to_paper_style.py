#!/usr/bin/env python3
"""
Helper script to convert plotting scripts from dark_transparent to paper style.

This script can be used to:
1. Show what changes would be made (dry-run mode)
2. Actually apply the changes to files
3. Create backup copies before modifying

Usage:
    python convert_to_paper_style.py --dry-run  # Show what would change
    python convert_to_paper_style.py --backup   # Create backups and apply changes
    python convert_to_paper_style.py            # Apply changes directly
"""

import argparse
import re
import shutil
from pathlib import Path
from typing import List, Tuple


# Mapping of conference font sizes to paper font sizes
FONTSIZE_MAP = {
    14: 9,
    15: 9,
    16: 10,
    17: 10,
    18: 10,
}

# Mapping of conference linewidths to paper linewidths
LINEWIDTH_MAP = {
    2.0: 1.5,
    2.2: 1.6,
    2.4: 1.6,
    2.6: 1.8,
}


def find_plot_style_calls(content: str) -> List[Tuple[str, str]]:
    """
    Find all set_plot_style calls with dark_transparent.
    
    Returns list of (full_match, indentation) tuples.
    """
    # Pattern to match set_plot_style calls with dark_transparent
    pattern = r'(\s*)set_plot_style\s*\(\s*style\s*=\s*["\']dark_transparent["\'][^)]*\)'
    matches = []
    for match in re.finditer(pattern, content, re.MULTILINE | re.DOTALL):
        indentation = match.group(1)
        full_match = match.group(0)
        matches.append((full_match, indentation))
    return matches


def convert_to_paper_style(old_call: str, indentation: str = "") -> str:
    """
    Convert a set_plot_style(style="dark_transparent", ...) call to set_paper_style(...).
    
    Parameters
    ----------
    old_call : str
        The original set_plot_style call
    indentation : str
        Leading whitespace to preserve
        
    Returns
    -------
    str
        The converted set_paper_style call
    """
    # Extract parameters
    base_fontsize = None
    linewidth = None
    n_colors = None
    cmap_name = None
    min_cycle = None
    
    # Extract base_fontsize
    fs_match = re.search(r'base_fontsize\s*=\s*(\d+(?:\.\d+)?)', old_call)
    if fs_match:
        old_fs = float(fs_match.group(1))
        base_fontsize = FONTSIZE_MAP.get(int(old_fs), 10)
    
    # Extract linewidth
    lw_match = re.search(r'linewidth\s*=\s*(\d+(?:\.\d+)?)', old_call)
    if lw_match:
        old_lw = float(lw_match.group(1))
        linewidth = LINEWIDTH_MAP.get(old_lw, 1.5)
    
    # Extract n_colors
    nc_match = re.search(r'n_colors\s*=\s*(\d+)', old_call)
    if nc_match:
        n_colors = int(nc_match.group(1))
    
    # Extract cmap_name
    cmap_match = re.search(r'cmap_name\s*=\s*["\']([^"\']+)["\']', old_call)
    if cmap_match:
        cmap_name = cmap_match.group(1)
    
    # Extract min_cycle
    mc_match = re.search(r'min_cycle\s*=\s*(\d+)', old_call)
    if mc_match:
        min_cycle = int(mc_match.group(1))
    
    # Build new call
    params = []
    if base_fontsize is not None:
        params.append(f"base_fontsize={base_fontsize}")
    if linewidth is not None:
        params.append(f"linewidth={linewidth}")
    if n_colors is not None:
        params.append(f"n_colors={n_colors}")
    if cmap_name is not None:
        params.append(f'cmap_name="{cmap_name}"')
    if min_cycle is not None:
        params.append(f"min_cycle={min_cycle}")
    
    # Default values if nothing was extracted
    if not params:
        params = ["base_fontsize=10", "linewidth=1.5"]
    
    param_str = ", ".join(params)
    new_call = f"{indentation}set_paper_style({param_str})"
    
    return new_call


def update_imports(content: str) -> str:
    """
    Update imports to include set_paper_style if needed.
    """
    # Check if set_paper_style is already imported
    if "set_paper_style" in content:
        return content
    
    # Find import statement for set_plot_style
    import_pattern = r'from\s+helpers\.trinity_plotting\s+import\s+([^;\n]+)'
    match = re.search(import_pattern, content)
    
    if match:
        imports = match.group(1)
        if "set_plot_style" in imports and "set_paper_style" not in imports:
            # Add set_paper_style to the import
            new_imports = imports.replace("set_plot_style", "set_plot_style, set_paper_style")
            content = content.replace(match.group(0), f"from helpers.trinity_plotting import {new_imports}")
    
    return content


def process_file(filepath: Path, dry_run: bool = False, backup: bool = False) -> bool:
    """
    Process a single file to convert dark_transparent to paper style.
    
    Returns True if changes were made/would be made.
    """
    try:
        content = filepath.read_text()
        original_content = content
        
        # Find all set_plot_style calls with dark_transparent
        matches = find_plot_style_calls(content)
        
        if not matches:
            return False
        
        # Convert each match
        for old_call, indentation in matches:
            new_call = convert_to_paper_style(old_call, indentation)
            content = content.replace(old_call, new_call)
        
        # Update imports
        content = update_imports(content)
        
        if dry_run:
            print(f"\n{'='*80}")
            print(f"File: {filepath}")
            print(f"{'='*80}")
            print("\nChanges that would be made:")
            for old_call, indentation in matches:
                new_call = convert_to_paper_style(old_call, indentation)
                print(f"\n  OLD:\n{old_call}")
                print(f"\n  NEW:\n{new_call}")
            return True
        
        # Create backup if requested
        if backup:
            backup_path = filepath.with_suffix(filepath.suffix + ".backup")
            shutil.copy2(filepath, backup_path)
            print(f"Created backup: {backup_path}")
        
        # Write the modified content
        filepath.write_text(content)
        print(f"Updated: {filepath}")
        return True
        
    except Exception as e:
        print(f"✗ Error processing {filepath}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert plotting scripts from dark_transparent to paper style",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show what would change without modifying files
  python convert_to_paper_style.py --dry-run
  
  # Apply changes with backups
  python convert_to_paper_style.py --backup
  
  # Apply changes to specific files
  python convert_to_paper_style.py --files script1.py script2.py
        """
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create .backup files before modifying"
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Specific files to process (otherwise processes all from list)"
    )
    
    args = parser.parse_args()
    
    # Get repository root
    repo_root = Path(__file__).parent.parent
    
    # List of all files to process (from the regeneration list)
    all_files = [
        "York_paper_check/fig6/Fermi-LAT_analysis_coupling.py",
        "Totani_paper_check/check_templates/make_template_grid_21gev.py",
        "Totani_paper_check/figures/make_fig11.py",
        "Totani_paper_check/figures/make_totani_fig8.py",
        "Totani_paper_check/figures/make_totani_fig9_likelihood.py",
        "Totani_Scattering/deconvolve_totani_spectrum.py",
        "Totani_Scattering/make_combined_fermion_scalar_tau_grid.py",
        "Totani_Scattering/make_higgs_portal_y_eff_grid.py",
        "Totani_Scattering/make_inscatter_mass_grid.py",
        "Totani_Scattering/make_paper_style_operator_overlays.py",
        "Totani_Scattering/plot_deconv_diagnostics.py",
        "Totani_Scattering/plot_tension_summary.py",
        "Fermionic_DM_Eff_Operator/Fermi-LAT_analysis_eff_coupling_fermionic.py",
        "Scalar_DM_Eff_Operator/Fermi-LAT_analysis_eff_coupling_scalar.py",
        "PBH_as_DM/conference_pbh_plot.py",
        "Scattering_in_Schwarzchild_Background/plots/dsigma_dOmega_all_cases_overlay.py",
        "Scattering_in_Schwarzchild_Background/plots/plot_a_potentials.py",
        "Scattering_in_Schwarzchild_Background/plots/plot_d_optical_depth_vs_energy_gc.py",
        "Scattering_in_Schwarzchild_Background/plots/plot_f_polarisation_vs_angle_all_cases.py",
        "Scattering_in_Schwarzchild_Background/plots/polarisation_vs_omegars_resonances.py",
        "Scattering_in_Schwarzchild_Background/plots/resonance_tower_sigma_vs_omegars.py",
        "Scattering_in_Schwarzchild_Background/cmb/pbh_cmb_lensing.py",
        "Scattering_in_Schwarzchild_Background/cmb/tau_vs_mchi_cmb.py",
        "Scattering_in_Schwarzchild_Background/validation_scripts/higgs_pole_scan.py",
        "Scattering_in_Schwarzchild_Background/validation_scripts/tau_vs_mchi.py",
    ]
    
    # Determine which files to process
    if args.files:
        files_to_process = [Path(f) for f in args.files]
    else:
        files_to_process = [repo_root / f for f in all_files]
    
    # Process files
    processed_count = 0
    for filepath in files_to_process:
        if not filepath.exists():
            print(f"⚠ File not found: {filepath}")
            continue
        
        if process_file(filepath, dry_run=args.dry_run, backup=args.backup):
            processed_count += 1
    
    # Summary
    print(f"\n{'='*80}")
    if args.dry_run:
        print(f"Dry run complete. {processed_count} files would be modified.")
        print("Run without --dry-run to apply changes.")
    else:
        print(f"Complete! {processed_count} files processed.")
        if args.backup:
            print("Backup files created with .backup extension.")


if __name__ == "__main__":
    main()
