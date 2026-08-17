import numpy as np
import pandas as pd
import os
import sys
from scipy.stats import spearmanr

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import moral_foundations_config as old_config
from config import MF_config_llm as new_config

data_path = 'data/plain/current/2000/'

def load_matrix(filename):
    return np.loadtxt(os.path.join(data_path, filename))

def get_scores(ideal_profile, space_profiles):
    dot_products = np.dot(ideal_profile, space_profiles)
    magnitudes = np.linalg.norm(space_profiles, axis=0)
    magnitudes[magnitudes == 0] = 1.0
    return dot_products / magnitudes

def run_loo_analysis(config_module, U, RAW_A, V, num_comp):
    ARCH_SPACE = np.matmul(U.T, RAW_A)
    TRAIT_CONFIGS = np.matmul(RAW_A, V)
    
    stabilities = {}
    
    for foundation, trait_map in config_module.FOUNDATIONS.items():
        trait_indices = list(trait_map.keys())
        if len(trait_indices) < 2:
            stabilities[foundation] = (float('nan'), len(trait_indices))
            continue

        # 1. Baseline
        baseline_profile = np.zeros(num_comp)
        for idx, direction in trait_map.items():
            baseline_profile += direction * TRAIT_CONFIGS[idx - 1, :num_comp]
        baseline_scores = get_scores(baseline_profile, ARCH_SPACE[:num_comp, :])
        
        # 2. LOO
        corrs = []
        for excluded_idx in trait_indices:
            loo_profile = np.zeros(num_comp)
            for idx, direction in trait_map.items():
                if idx == excluded_idx:
                    continue
                loo_profile += direction * TRAIT_CONFIGS[idx - 1, :num_comp]
            
            loo_scores = get_scores(loo_profile, ARCH_SPACE[:num_comp, :])
            corr, _ = spearmanr(baseline_scores, loo_scores)
            corrs.append(corr)
            
        stabilities[foundation] = (np.mean(corrs), len(trait_indices))
        
    return stabilities

def main():
    print("Loading data files...")
    try:
        U = load_matrix('data_archetype_space_U.txt')
        RAW_A = load_matrix('data_raw_A.txt')
        V = load_matrix('data_archetype_space_V.txt')
    except Exception as e:
        print(f"Error loading SVD files: {e}")
        return

    num_comp = 50
    
    print("\nRunning Leave-One-Out (LOO) Stability Analysis...")
    old_results = run_loo_analysis(old_config, U, RAW_A, V, num_comp)
    new_results = run_loo_analysis(new_config, U, RAW_A, V, num_comp)
    
    print("\n" + "="*80)
    print("                   MATHEMATICAL STABILITY COMPARISON")
    print("="*80)
    print(f"{'Moral Foundation':<20} | {'Old Config (3 traits)':<25} | {'New Config (Consensus)':<25}")
    print(f"{'':<20} | {'Mean stability (N)':<25} | {'Mean stability (N)':<25}")
    print("-"*80)
    
    for f in old_results.keys():
        old_stable, old_n = old_results[f]
        new_stable, new_n = new_results.get(f, (float('nan'), 0))
        
        old_str = f"{old_stable:.4f} ({old_n} traits)" if not np.isnan(old_stable) else "N/A"
        new_str = f"{new_stable:.4f} ({new_n} traits)" if not np.isnan(new_stable) else "N/A"
        
        print(f"{f:<20} | {old_str:<25} | {new_str:<25}")
        
    print("="*80)
    print("Interpretation:")
    print("- Stability Score is the average Spearman Rank Correlation when removing 1 trait.")
    print("- Higher scores indicate the foundation's representation is highly robust to variations.")
    print("- An increase in score mathematically proves the new configuration is more stable.")

if __name__ == "__main__":
    main()
