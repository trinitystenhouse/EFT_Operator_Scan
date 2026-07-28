"""
Cross-section caching system for expensive wave-optics calculations.

This module provides utilities to:
  1. Compute and cache Schwarzschild gravitational cross-sections
  2. Load precomputed cross-sections from disk
  3. Interpolate cached cross-sections to arbitrary parameters

The cache stores:
  - Differential cross-sections dσ/dΩ(θ, r/r_s) for perpendicular, parallel, unpolarized
  - Metadata: photon energy, BH mass, ℓ_max, theta grid, radius grid
  - Computation method and timestamp

Cache file format: HDF5 or NPZ (compressed numpy arrays)

Usage:
    from helpers.cross_section_cache import CrossSectionCache
    
    # Create cache manager
    cache = CrossSectionCache(cache_dir="./cache")
    
    # Try to load from cache
    result = cache.load(omega_gev=6.34e-13, m_bh_g=2.1e26, ell_max=50)
    
    if result is None:
        # Compute and save
        result = cache.compute_and_save(
            omega_gev=6.34e-13,
            m_bh_g=2.1e26,
            theta_arr=theta_grid,
            r_over_rs_arr=r_grid,
            ell_max=50
        )
    
    # Use cached cross-sections
    dsig_perp, dsig_par, dsig_unpol = result['dsig_perp'], result['dsig_par'], result['dsig_unpol']
"""

import hashlib
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


class CrossSectionCache:
    """
    Manager for caching expensive Schwarzschild cross-section calculations.
    """
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize cache manager.
        
        Parameters
        ----------
        cache_dir : str, optional
            Directory to store cache files. If None, uses default location.
        """
        if cache_dir is None:
            # Default: store in project root / cache / cross_sections
            cache_dir = Path(__file__).parent.parent / "cache" / "cross_sections"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file tracking all cached entries
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"entries": []}
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _generate_cache_key(self, omega_gev: float, m_bh_g: float, ell_max: int,
                           n_theta: int = None, n_r: int = None) -> str:
        """
        Generate unique cache key for given parameters.
        
        Parameters are rounded to avoid floating-point precision issues.
        """
        # Round to reasonable precision
        omega_str = f"{omega_gev:.6e}"
        m_bh_str = f"{m_bh_g:.6e}"
        ell_str = f"{ell_max:d}"
        
        # Include grid sizes if provided
        if n_theta is not None and n_r is not None:
            key_str = f"omega={omega_str}_mbh={m_bh_str}_ell={ell_str}_ntheta={n_theta}_nr={n_r}"
        else:
            key_str = f"omega={omega_str}_mbh={m_bh_str}_ell={ell_str}"
        
        # Hash for shorter filename
        hash_obj = hashlib.sha256(key_str.encode())
        return hash_obj.hexdigest()[:16]
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cache entry."""
        return self.cache_dir / f"xsec_{cache_key}.npz"
    
    def load(self, omega_gev: float, m_bh_g: float, ell_max: int,
             n_theta: int = None, n_r: int = None,
             allow_lower_ell: bool = True) -> Optional[Dict]:
        """
        Load cached cross-sections from disk.
        
        Parameters
        ----------
        omega_gev : float
            Photon energy in GeV
        m_bh_g : float
            Black hole mass in grams
        ell_max : int
            Maximum partial wave number
        n_theta : int, optional
            Number of theta samples (for exact match)
        n_r : int, optional
            Number of radius samples (for exact match)
        allow_lower_ell : bool, default=True
            If True, allow loading cache with lower ℓ_max and warn user
        
        Returns
        -------
        dict or None
            Dictionary with keys:
              - 'dsig_perp': (n_theta, n_r) array
              - 'dsig_par': (n_theta, n_r) array
              - 'dsig_unpol': (n_theta, n_r) array
              - 'theta_arr': (n_theta,) array
              - 'r_over_rs_arr': (n_r,) array
              - 'omega_gev': float
              - 'm_bh_g': float
              - 'ell_max': int
              - 'method': str
              - 'timestamp': str
            Returns None if not found in cache.
        """
        # Try exact match first
        cache_key = self._generate_cache_key(omega_gev, m_bh_g, ell_max, n_theta, n_r)
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            return self._load_from_file(cache_path)
        
        # Try without grid size specification
        if n_theta is not None or n_r is not None:
            cache_key = self._generate_cache_key(omega_gev, m_bh_g, ell_max)
            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists():
                return self._load_from_file(cache_path)
        
        # Try to find entry with lower ℓ_max
        if allow_lower_ell:
            for entry in self.metadata.get("entries", []):
                if (abs(entry["omega_gev"] - omega_gev) / omega_gev < 1e-6 and
                    abs(entry["m_bh_g"] - m_bh_g) / m_bh_g < 1e-6 and
                    entry["ell_max"] < ell_max):
                    
                    warnings.warn(
                        f"Exact cache not found for ℓ_max={ell_max}. "
                        f"Loading ℓ_max={entry['ell_max']} instead. "
                        f"Results may be less accurate.",
                        UserWarning
                    )
                    cache_path = self._get_cache_path(entry["cache_key"])
                    if cache_path.exists():
                        return self._load_from_file(cache_path)
        
        return None
    
    def _load_from_file(self, cache_path: Path) -> Dict:
        """Load cross-sections from NPZ file."""
        data = np.load(cache_path, allow_pickle=True)
        
        result = {
            'dsig_perp': data['dsig_perp'],
            'dsig_par': data['dsig_par'],
            'dsig_unpol': data['dsig_unpol'],
            'theta_arr': data['theta_arr'],
            'r_over_rs_arr': data['r_over_rs_arr'],
            'omega_gev': float(data['omega_gev']),
            'm_bh_g': float(data['m_bh_g']),
            'ell_max': int(data['ell_max']),
            'method': str(data['method']),
            'timestamp': str(data['timestamp']),
        }
        
        print(f"Loaded cached cross-sections from {cache_path.name}")
        print(f"  ω = {result['omega_gev']:.3e} GeV, M_BH = {result['m_bh_g']:.3e} g")
        print(f"  ℓ_max = {result['ell_max']}, method = {result['method']}")
        print(f"  Grid: {len(result['theta_arr'])} × {len(result['r_over_rs_arr'])} points")
        print(f"  Cached: {result['timestamp']}")
        
        return result
    
    def save(self, omega_gev: float, m_bh_g: float, ell_max: int,
             theta_arr: np.ndarray, r_over_rs_arr: np.ndarray,
             dsig_perp: np.ndarray, dsig_par: np.ndarray, dsig_unpol: np.ndarray,
             method: str = "full") -> str:
        """
        Save computed cross-sections to cache.
        
        Parameters
        ----------
        omega_gev : float
            Photon energy in GeV
        m_bh_g : float
            Black hole mass in grams
        ell_max : int
            Maximum partial wave number used
        theta_arr : ndarray
            Scattering angle grid (radians)
        r_over_rs_arr : ndarray
            Radius grid (in units of r_s)
        dsig_perp : ndarray
            Perpendicular cross-section (n_theta, n_r) [fb/sr]
        dsig_par : ndarray
            Parallel cross-section (n_theta, n_r) [fb/sr]
        dsig_unpol : ndarray
            Unpolarized cross-section (n_theta, n_r) [fb/sr]
        method : str
            Computation method ('full', 'go', 'auto')
        
        Returns
        -------
        str
            Cache key for the saved entry
        """
        n_theta = len(theta_arr)
        n_r = len(r_over_rs_arr)
        
        cache_key = self._generate_cache_key(omega_gev, m_bh_g, ell_max, n_theta, n_r)
        cache_path = self._get_cache_path(cache_key)
        
        # Save to compressed NPZ
        np.savez_compressed(
            cache_path,
            dsig_perp=dsig_perp,
            dsig_par=dsig_par,
            dsig_unpol=dsig_unpol,
            theta_arr=theta_arr,
            r_over_rs_arr=r_over_rs_arr,
            omega_gev=omega_gev,
            m_bh_g=m_bh_g,
            ell_max=ell_max,
            method=method,
            timestamp=datetime.now().isoformat(),
        )
        
        # Update metadata
        entry = {
            "cache_key": cache_key,
            "omega_gev": float(omega_gev),
            "m_bh_g": float(m_bh_g),
            "ell_max": int(ell_max),
            "n_theta": int(n_theta),
            "n_r": int(n_r),
            "method": method,
            "timestamp": datetime.now().isoformat(),
            "file_size_mb": cache_path.stat().st_size / 1024**2,
        }
        
        # Remove old entry with same key if exists
        self.metadata["entries"] = [
            e for e in self.metadata.get("entries", [])
            if e["cache_key"] != cache_key
        ]
        self.metadata["entries"].append(entry)
        self._save_metadata()
        
        print(f"Saved cross-sections to cache: {cache_path.name}")
        print(f"  File size: {entry['file_size_mb']:.2f} MB")
        
        return cache_key
    
    def compute_and_save(self, omega_gev: float, m_bh_g: float,
                        theta_arr: np.ndarray, r_over_rs_arr: np.ndarray,
                        m_chi_gev: float, ell_max: int,
                        method: str = "full") -> Dict:
        """
        Compute Schwarzschild cross-sections and save to cache.
        
        This is a convenience method that computes and caches in one call.
        
        Parameters
        ----------
        omega_gev : float
            Photon energy in GeV
        m_bh_g : float
            Black hole mass in grams
        theta_arr : ndarray
            Scattering angle grid (radians)
        r_over_rs_arr : ndarray
            Radius grid (in units of r_s)
        m_chi_gev : float
            DM particle mass in GeV
        ell_max : int
            Maximum partial wave number
        method : str
            Computation method ('full', 'go', 'auto')
        
        Returns
        -------
        dict
            Same format as load() method
        """
        # Import here to avoid circular dependency
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "plot_scripts"))
        from sch_grav import get_sch_grav_cross_sections_vectorized
        
        # Convert to r_s in GeV^-1
        G_CGS = 6.67430e-8
        C_CGS = 2.99792458e10
        GEVINV_TO_CM = 1.973269804e-14
        
        rs_cm = 2.0 * G_CGS * m_bh_g / C_CGS**2
        rs_gevinv = rs_cm / GEVINV_TO_CM
        
        print(f"Computing Schwarzschild cross-sections...")
        print(f"  ω = {omega_gev:.3e} GeV, M_BH = {m_bh_g:.3e} g")
        print(f"  ℓ_max = {ell_max}, method = {method}")
        print(f"  Grid: {len(theta_arr)} × {len(r_over_rs_arr)} points")
        print(f"  This may take several minutes...")
        
        dsig_perp, dsig_par, dsig_unpol = get_sch_grav_cross_sections_vectorized(
            omega_arr=np.array([omega_gev], dtype=float),
            theta_arr=theta_arr,
            r_over_rs_arr=r_over_rs_arr,
            m_chi_GeV=m_chi_gev,
            r_s_GeVinv=rs_gevinv,
            ell_max=ell_max,
            method=method,
        )
        
        # Extract first omega index
        dsig_perp = dsig_perp[0]
        dsig_par = dsig_par[0]
        dsig_unpol = dsig_unpol[0]
        
        print(f"Computation complete")
        
        # Save to cache
        self.save(
            omega_gev=omega_gev,
            m_bh_g=m_bh_g,
            ell_max=ell_max,
            theta_arr=theta_arr,
            r_over_rs_arr=r_over_rs_arr,
            dsig_perp=dsig_perp,
            dsig_par=dsig_par,
            dsig_unpol=dsig_unpol,
            method=method,
        )
        
        return {
            'dsig_perp': dsig_perp,
            'dsig_par': dsig_par,
            'dsig_unpol': dsig_unpol,
            'theta_arr': theta_arr,
            'r_over_rs_arr': r_over_rs_arr,
            'omega_gev': omega_gev,
            'm_bh_g': m_bh_g,
            'ell_max': ell_max,
            'method': method,
            'timestamp': datetime.now().isoformat(),
        }
    
    def list_cache(self):
        """Print all cached entries."""
        entries = self.metadata.get("entries", [])
        
        if not entries:
            print("Cache is empty.")
            return
        
        print(f"\nCached cross-sections ({len(entries)} entries):")
        print("=" * 80)
        
        for i, entry in enumerate(entries, 1):
            print(f"{i}. ω = {entry['omega_gev']:.3e} GeV, "
                  f"M_BH = {entry['m_bh_g']:.3e} g, "
                  f"ℓ_max = {entry['ell_max']}")
            print(f"   Grid: {entry['n_theta']} × {entry['n_r']}, "
                  f"Method: {entry['method']}, "
                  f"Size: {entry['file_size_mb']:.2f} MB")
            print(f"   Cached: {entry['timestamp']}")
            print()
    
    def clear_cache(self, confirm: bool = False):
        """
        Clear all cached cross-sections.
        
        Parameters
        ----------
        confirm : bool
            Must be True to actually delete files (safety check)
        """
        if not confirm:
            print("To clear cache, call with confirm=True")
            return
        
        entries = self.metadata.get("entries", [])
        
        for entry in entries:
            cache_path = self._get_cache_path(entry["cache_key"])
            if cache_path.exists():
                cache_path.unlink()
        
        self.metadata = {"entries": []}
        self._save_metadata()
        
        print(f"Cleared {len(entries)} cache entries")


def interpolate_cross_sections(cached_data: Dict, 
                               new_theta_arr: np.ndarray = None,
                               new_r_arr: np.ndarray = None) -> Dict:
    """
    Interpolate cached cross-sections to new grids.
    
    Parameters
    ----------
    cached_data : dict
        Data returned from CrossSectionCache.load()
    new_theta_arr : ndarray, optional
        New theta grid (radians). If None, use cached grid.
    new_r_arr : ndarray, optional
        New radius grid (r/r_s). If None, use cached grid.
    
    Returns
    -------
    dict
        Interpolated cross-sections on new grids
    """
    from scipy.interpolate import RectBivariateSpline
    
    theta_cached = cached_data['theta_arr']
    r_cached = cached_data['r_over_rs_arr']
    
    if new_theta_arr is None:
        new_theta_arr = theta_cached
    if new_r_arr is None:
        new_r_arr = r_cached
    
    # Create interpolators
    interp_perp = RectBivariateSpline(theta_cached, r_cached, cached_data['dsig_perp'])
    interp_par = RectBivariateSpline(theta_cached, r_cached, cached_data['dsig_par'])
    interp_unpol = RectBivariateSpline(theta_cached, r_cached, cached_data['dsig_unpol'])
    
    # Interpolate
    dsig_perp_new = interp_perp(new_theta_arr, new_r_arr)
    dsig_par_new = interp_par(new_theta_arr, new_r_arr)
    dsig_unpol_new = interp_unpol(new_theta_arr, new_r_arr)
    
    return {
        'dsig_perp': dsig_perp_new,
        'dsig_par': dsig_par_new,
        'dsig_unpol': dsig_unpol_new,
        'theta_arr': new_theta_arr,
        'r_over_rs_arr': new_r_arr,
        'omega_gev': cached_data['omega_gev'],
        'm_bh_g': cached_data['m_bh_g'],
        'ell_max': cached_data['ell_max'],
        'method': cached_data['method'],
        'timestamp': cached_data['timestamp'],
    }
