import numpy as np
import pandas as pd
import os
import sys

# Import from config/MF_config_llm.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import MF_config_llm as config

data_path = 'data/plain/current/2000/'

def load_matrix(filename):
    return np.loadtxt(os.path.join(data_path, filename))

def calculate_cosine_similarity(ideal_profile, space_profiles):
    alignment_scores = []
    for i in range(space_profiles.shape[1]):
        profile = space_profiles[:, i]
        char_magnitude = np.linalg.norm(profile)
        if char_magnitude == 0:
            alignment_scores.append(0)
        else:
            alignment_scores.append(np.dot(ideal_profile, profile) / char_magnitude)
    return alignment_scores

def main():
    print("Loading SVD data using MF_config_llm.py...")
    try:
        traits_df = pd.read_csv(os.path.join(data_path, 'data_traits.tsv'), sep='\t')
        characters_df = pd.read_csv(os.path.join(data_path, 'data_characters.tsv'), sep='\t')
        U = load_matrix('data_archetype_space_U.txt')
        V = load_matrix('data_archetype_space_V.txt')
        RAW_A = load_matrix('data_raw_A.txt')
        
        with open(os.path.join(data_path, 'data_archetype_space_archetype_classes.txt'), 'r') as f:
            archetype_names = [line.strip() for line in f.readlines()]
        
        with open(os.path.join(data_path, 'data_archetype_space_character_archetype_indices.txt'), 'r') as f:
            char_archetype_indices = [int(line.strip()) - 1 for line in f.readlines()]
            
        char_archetype_labels = [archetype_names[i] if i < len(archetype_names) else "Unknown" for i in char_archetype_indices]
        characters_df['Archetype'] = char_archetype_labels
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    ARCHETYPE_SPACE = np.matmul(U.T, RAW_A)
    TRAIT_CONFIGS = np.matmul(RAW_A, V)
    num_comp = config.NUM_COMPONENTS

    foundation_scores = {
        'character': characters_df['character'],
        'story': characters_df['character/story'],
        'archetype': characters_df['Archetype']
    }

    for component, trait_map in config.FOUNDATIONS.items():
        print(f"Processing component: {component}...")
        ideal_profile = np.zeros(num_comp)
        for trait_idx, direction in trait_map.items():
            ideal_profile += direction * TRAIT_CONFIGS[trait_idx - 1, :num_comp]
        
        ideal_norm = np.linalg.norm(ideal_profile)
        if ideal_norm > 0:
            ideal_profile /= ideal_norm
        
        foundation_scores[component] = calculate_cosine_similarity(ideal_profile, ARCHETYPE_SPACE[:num_comp, :])

    results_df = pd.DataFrame(foundation_scores)
    results_df['overall_morality'] = results_df[list(config.FOUNDATIONS.keys())].mean(axis=1)

    foundations_to_show = list(config.FOUNDATIONS.keys()) + ['overall_morality']
    print(f"\nDisplaying results for {len(foundations_to_show)} foundations...")

    for foundation in foundations_to_show:
        print("\n" + "#" * 120)
        print(f" Component: {foundation} ")
        print("#" * 120)
        
        top_20 = results_df.sort_values(foundation, ascending=False).head(20)
        
        header = f"{'Character':<25} | {'Story/Show':<35} | {'Archetype':<30} | {'Score':<10}"
        print(header)
        print("-" * 120)
        for _, row in top_20.iterrows():
            print(f"{str(row['character']):<25} | {str(row['story']):<35} | {str(row['archetype']):<30} | {row[foundation]:.4f}")
        print("-" * 120)

    print("\nDone!")

if __name__ == "__main__":
    main()
