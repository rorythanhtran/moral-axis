import numpy as np
import pandas as pd
import os
import sys
from scipy.stats import spearmanr

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import moral_foundations_config as config

data_path = 'data/plain/current/2000/'

def load_matrix(filename):
    return np.loadtxt(os.path.join(data_path, filename))

def get_scores(ideal_profile, space_profiles):
    dot_products = np.dot(ideal_profile, space_profiles)
    magnitudes = np.linalg.norm(space_profiles, axis=0)
    magnitudes[magnitudes == 0] = 1.0
    return dot_products / magnitudes

def main():
    print("--- Sensitivity Analysis: Leave-One-Out (LOO) Stability ---")
    
    try:
        U = load_matrix('data_archetype_space_U.txt')
        RAW_A = load_matrix('data_raw_A.txt')
        V = load_matrix('data_archetype_space_V.txt')
        traits_df = pd.read_csv(os.path.join(data_path, 'data_traits.tsv'), sep='\t')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    ARCH_SPACE = np.matmul(U.T, RAW_A)
    TRAIT_CONFIGS = np.matmul(RAW_A, V)
    num_comp = config.NUM_COMPONENTS
    
    results = []

    for foundation, trait_map in config.FOUNDATIONS.items():
        print(f"\nTesting Foundation: {foundation}")
        print("-" * 50)
        
        trait_indices = list(trait_map.keys())
        if len(trait_indices) < 2:
            print(f"Skipping {foundation}: Need at least 2 traits for LOO analysis.")
            continue

        # 1. Calculate Baseline
        baseline_profile = np.zeros(num_comp)
        for idx, direction in trait_map.items():
            baseline_profile += direction * TRAIT_CONFIGS[idx - 1, :num_comp]
        baseline_scores = get_scores(baseline_profile, ARCH_SPACE[:num_comp, :])
        
        # 2. Leave-One-Out Iteration
        for excluded_idx in trait_indices:
            loo_profile = np.zeros(num_comp)
            for idx, direction in trait_map.items():
                if idx == excluded_idx:
                    continue
                loo_profile += direction * TRAIT_CONFIGS[idx - 1, :num_comp]
            
            loo_scores = get_scores(loo_profile, ARCH_SPACE[:num_comp, :])
            
            # Calculate Rank Correlation
            corr, _ = spearmanr(baseline_scores, loo_scores)
            
            trait_name = traits_df[traits_df['index'] == excluded_idx].iloc[0]['differential']
            
            results.append({
                'Foundation': foundation,
                'Excluded_Trait': trait_name,
                'Rank_Correlation': corr
            })
            print(f"  Excluded '{trait_name:<30}' | Stability: {corr:.4f}")

    # Summary
    summary_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print(" STABILITY SUMMARY ")
    print("="*60)
    print(summary_df.sort_values('Rank_Correlation').to_string(index=False))
    print("\nInterpretation:")
    print("- High Stability (>0.95): The foundation is robust; no single trait dominates.")
    print("- Low Stability (<0.80): The excluded trait was 'carrying' the definition.")

if __name__ == "__main__":
    main()
