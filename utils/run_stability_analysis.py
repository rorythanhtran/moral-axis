import numpy as np
import pandas as pd
import os
import sys

# Add project root to path so we can import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import moral_foundations_config as config

# Set paths
data_path = 'data/plain/current/2000/'
# Sweep through different resolutions to find the most robust characters
COMPONENTS_TO_TEST = [6, 10, 20, 50, 100, 232]

def load_matrix(filename):
    return np.loadtxt(os.path.join(data_path, filename))

def calculate_cosine_similarity(ideal_profile, space_profiles):
    """Efficiently calculate cosine similarity for all characters at once"""
    # ideal_profile: (k,)
    # space_profiles: (k, 2000)
    dot_products = np.dot(ideal_profile, space_profiles)
    magnitudes = np.linalg.norm(space_profiles, axis=0)
    
    # Handle zero magnitudes safely
    magnitudes[magnitudes == 0] = 1.0
    return dot_products / magnitudes

def main():
    print("--- Starting Moral Stability Analysis (Consistency Check) ---")
    
    try:
        characters_df = pd.read_csv(os.path.join(data_path, 'data_characters.tsv'), sep='\t')
        U = load_matrix('data_archetype_space_U.txt')
        RAW_A = load_matrix('data_raw_A.txt')
        V = load_matrix('data_archetype_space_V.txt')
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Track appearances in Top 10 across ALL foundations and ALL resolutions
    # This finds characters who are "Consistently Moral"
    stability_tracker = {char: 0 for char in characters_df['character']}
    story_map = {row['character']: row['character/story'] for _, row in characters_df.iterrows()}
    
    # Pre-calculate full spaces
    ARCHETYPE_SPACE_FULL = np.matmul(U.T, RAW_A)
    TRAIT_CONFIGS_FULL = np.matmul(RAW_A, V)

    for k in COMPONENTS_TO_TEST:
        print(f"Analyzing at {k:>3} components...")
        
        for foundation, trait_map in config.FOUNDATIONS.items():
            # Build ideal profile for this resolution
            ideal_profile = np.zeros(k)
            for trait_idx, direction in trait_map.items():
                # trait_idx - 1 because matrix is 0-indexed
                ideal_profile += direction * TRAIT_CONFIGS_FULL[trait_idx - 1, :k]
            
            # Normalize
            ideal_norm = np.linalg.norm(ideal_profile)
            if ideal_norm > 0:
                ideal_profile /= ideal_norm
            
            # Get scores for all 2000 characters
            scores = calculate_cosine_similarity(ideal_profile, ARCHETYPE_SPACE_FULL[:k, :])
            
            # Find indices of Top 10 characters
            top_indices = np.argsort(scores)[-10:]
            for idx in top_indices:
                char_name = characters_df.iloc[idx]['character']
                stability_tracker[char_name] += 1

    # Format Results
    results = pd.DataFrame([
        {
            'Character': name, 
            'Story': story_map[name],
            'Stability_Count': count
        } 
        for name, count in stability_tracker.items()
        if count > 0 # Only show characters who placed at least once
    ])
    
    print("\n" + "="*70)
    print(" MORAL STABILITY HALL OF FAME ")
    print(" Frequency of appearing in the Top 10 across 6 resolutions & 6 Foundations ")
    print("="*70)
    
    # Sort by how many times they appeared in a Top 10 list
    hall_of_fame = results.sort_values('Stability_Count', ascending=False).head(20)
    
    print(f"{'Character':<25} | {'Story/Show':<35} | {'Appearances':<12}")
    print("-" * 75)
    for _, row in hall_of_fame.iterrows():
        print(f"{str(row['Character']):<25} | {str(row['Story']):<35} | {row['Stability_Count']:<12}")
    print("-" * 75)
    
    print("\nInterpretation: A high 'Appearance' count means a character is consistently")
    print("moral regardless of how much 'detail' (SVD components) we look at.")

if __name__ == "__main__":
    main()
