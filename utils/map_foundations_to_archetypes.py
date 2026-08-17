import numpy as np
import os
import sys

# Add project root to path so we can import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import moral_foundations_config as config

# Set paths
data_path = 'data/plain/current/2000/'
ARCHETYPE_LABELS = ["Hero", "Fool", "Angel", "Demon", "Traditionalist", "Adventurer"]

def load_matrix(filename):
    return np.loadtxt(os.path.join(data_path, filename))

def main():
    print("--- Mapping Moral Foundations to Archetype Labels ---")
    
    # Load matrices
    try:
        V = load_matrix('data_archetype_space_V.txt')
        RAW_A = load_matrix('data_raw_A.txt')
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    # Traits x Archetypes (using V as the character-basis)
    TRAIT_ARCHETYPE_PROFILES = np.matmul(RAW_A, V)
    
    num_comp = 6 # We only care about the top 6

    for foundation, trait_map in config.FOUNDATIONS.items():
        foundation_profile = np.zeros(num_comp)
        for trait_idx, direction in trait_map.items():
            foundation_profile += direction * TRAIT_ARCHETYPE_PROFILES[trait_idx - 1, :num_comp]
        
        # Normalize for easier comparison
        norm = np.linalg.norm(foundation_profile)
        if norm > 0:
            foundation_profile /= norm
        
        print(f"\n[ FOUNDATION: {foundation.upper()} ]")
        print("-" * 65)
        print(f"{'Archetype Label':<15} | {'Weight':<8} | {'Influence Graph'}")
        print("-" * 65)
        
        for i in range(num_comp):
            weight = foundation_profile[i]
            label = ARCHETYPE_LABELS[i]
            bar_len = int(abs(weight) * 30)
            bar = ("#" * bar_len).ljust(30)
            
            # Show if it's a positive or negative pull
            indicator = "(+)" if weight > 0 else "(-)"
            print(f"{label:<15} | {weight:+.3f} | {bar} {indicator}")

if __name__ == "__main__":
    main()
