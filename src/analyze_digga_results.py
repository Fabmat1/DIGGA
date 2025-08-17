#!/usr/bin/env python3
"""
Analyze DIGGA fit results by comparing with true parameters from metadata.
"""

import json
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from pathlib import Path
import argparse
from collections import defaultdict

# Define parameter display names and units
PARAM_INFO = {
    'teff': {'name': 'T_eff', 'unit': 'K', 'label': r'$T_{\rm eff}$ [K]', 'color': 'tab:blue'},
    'logg': {'name': 'log g', 'unit': 'dex', 'label': r'$\log g$ [dex]', 'color': 'tab:orange'},
    'vsini': {'name': 'v sin i', 'unit': 'km/s', 'label': r'$v \sin i$ [km/s]', 'color': 'tab:green'},
    'xi': {'name': 'ξ', 'unit': 'km/s', 'label': r'$\xi$ [km/s]', 'color': 'tab:red'},
    'z': {'name': '[M/H]', 'unit': 'dex', 'label': '[M/H] [dex]', 'color': 'tab:purple'},
    'zeta': {'name': 'ζ', 'unit': 'km/s', 'label': r'$\zeta$ [km/s]', 'color': 'tab:brown'},
    'he': {'name': 'log(He/H)', 'unit': 'dex', 'label': r'$\log$(He/H) [dex]', 'color': 'tab:pink'},
    'vrad': {'name': 'v_rad', 'unit': 'km/s', 'label': r'$v_{\rm rad}$ [km/s]', 'color': 'tab:gray'},
}

class DIGGAAnalyzer:
    def __init__(self, data_dir='.', results_dir='fit_results', exclude_params=None):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.data = defaultdict(list)
        self.exclude_params = set(exclude_params or [])
        
    def load_single_dataset(self, folder_idx):
        """Load metadata and fit results for a single folder."""
        folder_name = f"{folder_idx:04d}"
        
        # Load metadata
        metadata_path = self.data_dir / folder_name / "metadata.json"
        if not metadata_path.exists():
            return None
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load fit results
        fit_path = self.results_dir / folder_name / "fit_parameters.csv"
        if not fit_path.exists():
            return None
            
        fit_params = {}
        fit_errors = {}
        
        with open(fit_path, 'r') as f:
            reader = csv.reader(f)
            # Skip header row if it exists
            first_row = next(reader, None)
            if first_row and (first_row[0].lower() in ['parameter', 'param', 'name'] or 
                             first_row[1].lower() in ['value', 'val']):
                # This looks like a header row, skip it
                pass
            else:
                # This is data, process it
                if first_row and len(first_row) >= 3:
                    param_name = first_row[0]
                    try:
                        value = float(first_row[1])
                        error = float(first_row[2])
                        fit_params[param_name] = value
                        fit_errors[param_name] = error
                    except ValueError:
                        # Skip this row if conversion fails
                        pass
            
            # Process remaining rows
            for row in reader:
                if len(row) >= 3:
                    param_name = row[0]
                    try:
                        value = float(row[1])
                        error = float(row[2])
                        fit_params[param_name] = value
                        fit_errors[param_name] = error
                    except ValueError:
                        # Skip rows where value/error can't be converted to float
                        continue
        
        return {
            'metadata': metadata,
            'fit_params': fit_params,
            'fit_errors': fit_errors,
            'folder': folder_name
        }
    
    def extract_comparisons(self, dataset):
        """Extract parameter comparisons from a single dataset."""
        metadata = dataset['metadata']
        fit_params = dataset['fit_params']
        fit_errors = dataset['fit_errors']
        
        true_params = metadata['true_parameters']
        comparisons = {}
        
        # Compare global parameters
        for param in ['teff', 'logg', 'vsini', 'xi', 'z', 'zeta', 'he']:
            if param in self.exclude_params:
                continue
                
            fit_key = f'c1_{param}'
            if fit_key in fit_params and param in true_params:
                comparisons[param] = {
                    'true': true_params[param],
                    'fit': fit_params[fit_key],
                    'error': fit_errors[fit_key],
                    'diff': fit_params[fit_key] - true_params[param]
                }
        
        # Compare individual vrad values
        if 'vrad' not in self.exclude_params:
            vrad_comparisons = []
            for i, spectrum in enumerate(metadata['spectra'], 1):
                fit_key = f'c1_vrad_d{i}'
                if fit_key in fit_params:
                    vrad_comparisons.append({
                        'true': spectrum['vrad_actual'],
                        'fit': fit_params[fit_key],
                        'error': fit_errors[fit_key],
                        'diff': fit_params[fit_key] - spectrum['vrad_actual'],
                        'spectrum_idx': i
                    })
            
            if vrad_comparisons:
                comparisons['vrad'] = vrad_comparisons
            
        return comparisons
    
    def collect_all_data(self, start_idx, end_idx):
        """Collect data from all folders."""
        print("Loading data...")
        
        if self.exclude_params:
            print(f"Excluding parameters: {', '.join(self.exclude_params)}")
        
        successful_loads = 0
        failed_loads = 0
        
        for idx in range(start_idx, end_idx + 1):
            try:
                dataset = self.load_single_dataset(idx)
                if dataset:
                    comparisons = self.extract_comparisons(dataset)
                    
                    # Store global parameters
                    for param, comp in comparisons.items():
                        if param != 'vrad':
                            self.data[param].append(comp)
                        else:
                            # Store vrad comparisons
                            self.data['vrad'].extend(comp)
                    successful_loads += 1
                else:
                    failed_loads += 1
            except Exception as e:
                print(f"Error loading dataset {idx:04d}: {e}")
                failed_loads += 1
        
        print(f"Successfully loaded {successful_loads} datasets")
        print(f"Failed to load {failed_loads} datasets")
        
        # Print summary of loaded parameters
        for param, data_list in self.data.items():
            print(f"  {param}: {len(data_list)} measurements")
    
    def plot_scatter_histograms(self):
        """Create scatter plots and histograms for all parameters."""
        # Determine layout - only plot parameters that are not excluded
        params = [p for p in PARAM_INFO.keys() 
                 if p in self.data and self.data[p] and p not in self.exclude_params]
        n_params = len(params)
        
        if n_params == 0:
            print("No data to plot!")
            return
        
        print(f"Plotting {n_params} parameters: {', '.join(params)}")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 4 * n_params))
        gs = GridSpec(n_params, 3, width_ratios=[2, 2, 1], hspace=0.3, wspace=0.3)
        
        for i, param in enumerate(params):
            param_data = self.data[param]
            color = PARAM_INFO[param]['color']
            
            if not param_data:
                continue
                
            # Extract values
            if param == 'vrad':
                true_vals = np.array([d['true'] for d in param_data])
                fit_vals = np.array([d['fit'] for d in param_data])
                errors = np.array([d['error'] for d in param_data])
                diffs = np.array([d['diff'] for d in param_data])
            else:
                true_vals = np.array([d['true'] for d in param_data])
                fit_vals = np.array([d['fit'] for d in param_data])
                errors = np.array([d['error'] for d in param_data])
                diffs = np.array([d['diff'] for d in param_data])
            
            # 1. True vs Fitted scatter plot
            ax1 = fig.add_subplot(gs[i, 0])
            ax1.errorbar(true_vals, fit_vals, yerr=errors, fmt='o', alpha=0.6, 
                        markersize=4, capsize=3, capthick=1, color=color)
            
            # Add diagonal line
            val_range = [min(true_vals.min(), fit_vals.min()), 
                        max(true_vals.max(), fit_vals.max())]
            ax1.plot(val_range, val_range, 'k--', alpha=0.5, label='1:1 line')
            
            ax1.set_xlabel(f'True {PARAM_INFO[param]["label"]}')
            ax1.set_ylabel(f'Fitted {PARAM_INFO[param]["label"]}')
            ax1.set_title(f'{PARAM_INFO[param]["name"]} Comparison')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 2. Residuals scatter plot
            ax2 = fig.add_subplot(gs[i, 1])
            ax2.errorbar(true_vals, diffs, yerr=errors, fmt='o', alpha=0.6,
                        markersize=4, capsize=3, capthick=1, color=color)
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            
            ax2.set_xlabel(f'True {PARAM_INFO[param]["label"]}')
            ax2.set_ylabel(f'Residual (Fit - True) [{PARAM_INFO[param]["unit"]}]')
            ax2.set_title(f'{PARAM_INFO[param]["name"]} Residuals')
            ax2.grid(True, alpha=0.3)
            
            # Add RMS text
            rms = np.sqrt(np.mean(diffs**2))
            ax2.text(0.02, 0.98, f'RMS = {rms:.3f} {PARAM_INFO[param]["unit"]}',
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # 3. Histogram of residuals
            ax3 = fig.add_subplot(gs[i, 2])
            
            # Determine sensible bin limits
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs)
            hist_range = (mean_diff - 4*std_diff, mean_diff + 4*std_diff)
            
            n, bins, patches = ax3.hist(diffs, bins=30, range=hist_range, 
                                       orientation='horizontal', alpha=0.7, 
                                       color=color, edgecolor='black')
            
            # Add normal distribution overlay
            x_hist = np.linspace(hist_range[0], hist_range[1], 100)
            y_hist = len(diffs) * (bins[1] - bins[0]) * \
                     np.exp(-(x_hist - mean_diff)**2 / (2 * std_diff**2)) / \
                     (std_diff * np.sqrt(2 * np.pi))
            ax3.plot(y_hist, x_hist, 'r-', linewidth=2, label='Normal dist.')
            
            ax3.set_xlabel('Count')
            ax3.set_ylabel(f'Residual [{PARAM_INFO[param]["unit"]}]')
            ax3.set_title('Distribution')
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.legend()
            
            # Add statistics text
            ax3.text(0.98, 0.98, f'μ = {mean_diff:.3f}\nσ = {std_diff:.3f}',
                    transform=ax3.transAxes, verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('DIGGA Fit Results Analysis', fontsize=16, y=0.995)
        plt.tight_layout()
        
        # Save figure
        output_file = 'digga_analysis_results.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved analysis plots to {output_file}")
        
        # Also create a summary statistics file
        self.save_statistics_summary()
        
        plt.show()
    
    def save_statistics_summary(self):
        """Save summary statistics to a text file."""
        with open('digga_analysis_summary.txt', 'w') as f:
            f.write("DIGGA Fit Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            
            if self.exclude_params:
                f.write(f"Excluded parameters: {', '.join(self.exclude_params)}\n\n")
            
            for param in PARAM_INFO.keys():
                if param not in self.data or not self.data[param] or param in self.exclude_params:
                    continue
                    
                param_data = self.data[param]
                
                if param == 'vrad':
                    diffs = np.array([d['diff'] for d in param_data])
                    errors = np.array([d['error'] for d in param_data])
                else:
                    diffs = np.array([d['diff'] for d in param_data])
                    errors = np.array([d['error'] for d in param_data])
                
                f.write(f"{PARAM_INFO[param]['name']} ({param}):\n")
                f.write(f"  Number of measurements: {len(diffs)}\n")
                f.write(f"  Mean residual: {np.mean(diffs):.4f} {PARAM_INFO[param]['unit']}\n")
                f.write(f"  Std deviation: {np.std(diffs):.4f} {PARAM_INFO[param]['unit']}\n")
                f.write(f"  RMS: {np.sqrt(np.mean(diffs**2)):.4f} {PARAM_INFO[param]['unit']}\n")
                f.write(f"  Median residual: {np.median(diffs):.4f} {PARAM_INFO[param]['unit']}\n")
                f.write(f"  Mean fit error: {np.mean(errors):.4f} {PARAM_INFO[param]['unit']}\n")
                f.write(f"  Min residual: {np.min(diffs):.4f} {PARAM_INFO[param]['unit']}\n")
                f.write(f"  Max residual: {np.max(diffs):.4f} {PARAM_INFO[param]['unit']}\n")
                f.write("\n")
            
        print("Saved summary statistics to digga_analysis_summary.txt")

def main():
    parser = argparse.ArgumentParser(description='Analyze DIGGA fit results')
    parser.add_argument('--start', type=int, default=1, help='Starting folder index (default: 1)')
    parser.add_argument('--end', type=int, required=True, help='Ending folder index')
    parser.add_argument('--data-dir', type=str, default='.', help='Directory containing mock data folders (default: .)')
    parser.add_argument('--results-dir', type=str, default='fit_results', help='Directory containing fit results (default: fit_results)')
    parser.add_argument('--exclude', nargs='*', default=[], 
                       help='Parameters to exclude from plots (e.g., --exclude teff logg)')
    
    args = parser.parse_args()
    
    # Validate exclude parameters
    invalid_params = [p for p in args.exclude if p not in PARAM_INFO]
    if invalid_params:
        print(f"Warning: Invalid parameter names to exclude: {invalid_params}")
        print(f"Valid parameters are: {list(PARAM_INFO.keys())}")
        args.exclude = [p for p in args.exclude if p in PARAM_INFO]
    
    analyzer = DIGGAAnalyzer(args.data_dir, args.results_dir, args.exclude)
    analyzer.collect_all_data(args.start, args.end)
    analyzer.plot_scatter_histograms()

if __name__ == "__main__":
    main()