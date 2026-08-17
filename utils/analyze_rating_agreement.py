import pandas as pd
import numpy as np
import os
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
HUMAN_DIR = ROOT / "config" / "human_rating" / "responses"
LLM_DIR = ROOT / "config" / "llm_rating" / "outputs"
RESULTS_DIR = ROOT / "results"

FOUNDATIONS = ["care", "fairness", "loyalty", "authority", "purity", "liberty", "general"]

def cohen_kappa(y1, y2):
    """Calculate Cohen's Kappa for binary ratings (0 or 1) manually to avoid dependencies."""
    y1 = np.array(y1, dtype=int)
    y2 = np.array(y2, dtype=int)
    
    if len(y1) != len(y2) or len(y1) == 0:
        return np.nan
        
    total = len(y1)
    # Observed agreement
    po = np.sum(y1 == y2) / total
    
    # Expected agreement by chance
    p1_rate = np.sum(y1 == 1) / total
    p0_rate = 1.0 - p1_rate
    q1_rate = np.sum(y2 == 1) / total
    q0_rate = 1.0 - q1_rate
    
    pe = (p1_rate * q1_rate) + (p0_rate * q0_rate)
    
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
        
    return (po - pe) / (1.0 - pe)

def load_human_ratings():
    # Fehr: CSV format, columns: trait_index, differential, ..., purity, general, notes
    fehr_path = HUMAN_DIR / "trait_foundation_rating_fehr29june2026.csv"
    # Tran: TSV format, columns: trait_index, differential, ..., sanctity, general, notes
    tran_path = HUMAN_DIR / "trait_foundation_rating_tran_1july2026.tsv"
    
    if not fehr_path.exists() or not tran_path.exists():
        print("Missing one or both human rating files.")
        return None, None
        
    fehr_df = pd.read_csv(fehr_path)
    tran_df = pd.read_csv(tran_path, sep='\t')
    
    # Standardize columns
    fehr_df = fehr_df.rename(columns={'purity': 'purity'})
    tran_df = tran_df.rename(columns={'sanctity': 'purity'}) # Tran uses 'sanctity'
    
    # Clean up column types (ensure binary numeric)
    for f in FOUNDATIONS:
        fehr_df[f] = pd.to_numeric(fehr_df[f], errors='coerce').fillna(0).astype(int)
        tran_df[f] = pd.to_numeric(tran_df[f], errors='coerce').fillna(0).astype(int)
        
    return fehr_df, tran_df

def load_llm_ratings():
    llms = {
        'claude': LLM_DIR / "claude_claude-sonnet-4-6" / "combined_ratings.tsv",
        'gemini': LLM_DIR / "gemini_gemini-2.5-flash" / "combined_ratings.tsv",
        'deepseek': LLM_DIR / "deepseek_deepseek-v4-flash" / "combined_ratings.tsv"
    }
    
    llm_dfs = {}
    for name, path in llms.items():
        if path.exists():
            df = pd.read_csv(path, sep='\t')
            for f in FOUNDATIONS:
                df[f] = pd.to_numeric(df[f], errors='coerce').fillna(0).astype(int)
            llm_dfs[name] = df
        else:
            print(f"Warning: LLM ratings for {name} not found at {path}")
            
    return llm_dfs

def main():
    print("==================================================")
    print("      Moral Foundations Rating Agreement Analysis ")
    print("==================================================")
    
    fehr_df, tran_df = load_human_ratings()
    llm_dfs = load_llm_ratings()
    
    if fehr_df is None or tran_df is None or not llm_dfs:
        print("Required files not loaded. Exiting.")
        return

    # Align on common indices
    # We want indices that exist in both humans and all loaded LLMs
    common_indices = set(fehr_df['trait_index']).intersection(set(tran_df['trait_index']))
    for name, df in llm_dfs.items():
        common_indices = common_indices.intersection(set(df['trait_index']))
        
    common_indices = sorted(list(common_indices))
    print(f"Analyzing {len(common_indices)} common traits across all raters.")
    
    # Filter datasets to only common indices
    fehr_clean = fehr_df[fehr_df['trait_index'].isin(common_indices)].set_index('trait_index').sort_index()
    tran_clean = tran_df[tran_df['trait_index'].isin(common_indices)].set_index('trait_index').sort_index()
    
    llm_cleans = {}
    for name, df in llm_dfs.items():
        llm_cleans[name] = df[df['trait_index'].isin(common_indices)].set_index('trait_index').sort_index()

    # Build human consensus (agreed ratings where both say 1, or majority etc. - let's do strict agreement and union)
    # Human Strict Agreement (+1 when both say 1)
    human_consensus = {}
    # LLM Consensus (+1 when at least 2 models say 1)
    llm_consensus = {}
    
    # Analysis results container
    results = []
    
    for f in FOUNDATIONS:
        # Values for each rater
        h1_vals = fehr_clean[f].values
        h2_vals = tran_clean[f].values
        
        c_vals = llm_cleans['claude'][f].values if 'claude' in llm_cleans else None
        g_vals = llm_cleans['gemini'][f].values if 'gemini' in llm_cleans else None
        d_vals = llm_cleans['deepseek'][f].values if 'deepseek' in llm_cleans else None
        
        # Calculate Human Consensus (both say 1)
        h_agree_1 = (h1_vals == 1) & (h2_vals == 1)
        h_agree_1_int = h_agree_1.astype(int)
        
        # Calculate LLM Consensus (majority vote: at least 2 out of 3 say 1)
        llm_votes = np.zeros_like(h1_vals)
        if c_vals is not None: llm_votes += (c_vals == 1)
        if g_vals is not None: llm_votes += (g_vals == 1)
        if d_vals is not None: llm_votes += (d_vals == 1)
        
        llm_agree_1 = (llm_votes >= 2).astype(int)
        
        # 1. Human-Human Agreement
        hh_pct = np.mean(h1_vals == h2_vals) * 100
        hh_kappa = cohen_kappa(h1_vals, h2_vals)
        
        # 2. Human Consensus vs. LLM Consensus Agreement
        hl_pct = np.mean(h_agree_1_int == llm_agree_1) * 100
        hl_kappa = cohen_kappa(h_agree_1_int, llm_agree_1)
        
        # 3. Individual Human-LLM Agreements (average)
        pct_list = []
        kappa_list = []
        for h_vals in [h1_vals, h2_vals]:
            for model_name, model_clean in llm_cleans.items():
                m_vals = model_clean[f].values
                pct_list.append(np.mean(h_vals == m_vals) * 100)
                kappa_list.append(cohen_kappa(h_vals, m_vals))
        
        avg_hl_pct = np.mean(pct_list)
        avg_hl_kappa = np.mean(kappa_list)
        
        # 4. LLM Inter-model Agreement (Average of C-G, G-D, C-D)
        llm_pct_list = []
        llm_kappa_list = []
        model_names = list(llm_cleans.keys())
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                m1_vals = llm_cleans[model_names[i]][f].values
                m2_vals = llm_cleans[model_names[j]][f].values
                llm_pct_list.append(np.mean(m1_vals == m2_vals) * 100)
                llm_kappa_list.append(cohen_kappa(m1_vals, m2_vals))
        
        avg_llm_pct = np.mean(llm_pct_list) if llm_pct_list else 100.0
        avg_llm_kappa = np.mean(llm_kappa_list) if llm_kappa_list else 1.0

        results.append({
            'Foundation': f,
            'Human_Human_Agreement_%': hh_pct,
            'Human_Human_Kappa': hh_kappa,
            'LLM_LLM_Agreement_%': avg_llm_pct,
            'LLM_LLM_Kappa': avg_llm_kappa,
            'HumanConsensus_LLMConsensus_Agreement_%': hl_pct,
            'HumanConsensus_LLMConsensus_Kappa': hl_kappa,
            'Avg_Individual_Human_LLM_Agreement_%': avg_hl_pct,
            'Avg_Individual_Human_LLM_Kappa': avg_hl_kappa,
            'Human_1s_Count': np.sum(h_agree_1_int),
            'LLM_1s_Count': np.sum(llm_agree_1)
        })

    summary_df = pd.DataFrame(results)
    
    print("\n" + "="*95)
    print(f"{'FOUNDATION AGREEMENT SUMMARY':^95}")
    print("="*95)
    header = f"{'Foundation':<12} | {'Human-Human':<18} | {'LLM-LLM':<18} | {'Human-LLM Avg':<18} | {'Consensus-Consensus':<20}"
    print(header)
    print(f"{'':<12} | {'Pct (Kappa)':<18} | {'Pct (Kappa)':<18} | {'Pct (Kappa)':<18} | {'Pct (Kappa)':<20}")
    print("-" * 95)
    
    for _, row in summary_df.iterrows():
        hh_str = f"{row['Human_Human_Agreement_%']:.1f}% ({row['Human_Human_Kappa']:.2f})"
        ll_str = f"{row['LLM_LLM_Agreement_%']:.1f}% ({row['LLM_LLM_Kappa']:.2f})"
        hl_str = f"{row['Avg_Individual_Human_LLM_Agreement_%']:.1f}% ({row['Avg_Individual_Human_LLM_Kappa']:.2f})"
        cc_str = f"{row['HumanConsensus_LLMConsensus_Agreement_%']:.1f}% ({row['HumanConsensus_LLMConsensus_Kappa']:.2f})"
        
        print(f"{row['Foundation'].capitalize():<12} | {hh_str:<18} | {ll_str:<18} | {hl_str:<18} | {cc_str:<20}")
    print("="*95)
    print("Kappa Interpretation: <0 Poor, 0-0.20 Slight, 0.21-0.40 Fair, 0.41-0.60 Moderate, 0.61-0.80 Substantial, 0.81-1.00 Almost Perfect")

    # ----------------------------------------------------
    # Identify Specific Disagreement / Alignment Patterns
    # ----------------------------------------------------
    print("\n" + "="*95)
    print(f"{'INTERESTING RATING DISCREPANCIES (TOP EXAMPLES)':^95}")
    print("="*95)
    
    discrepancies = []
    
    for idx in common_indices:
        trait_diff = fehr_clean.loc[idx, 'differential']
        
        for f in FOUNDATIONS:
            h1 = fehr_clean.loc[idx, f]
            h2 = tran_clean.loc[idx, f]
            
            c = llm_cleans['claude'].loc[idx, f] if 'claude' in llm_cleans else 0
            g = llm_cleans['gemini'].loc[idx, f] if 'gemini' in llm_cleans else 0
            d = llm_cleans['deepseek'].loc[idx, f] if 'deepseek' in llm_cleans else 0
            
            # Humans agree it is 1, but LLMs all agree it is 0
            if h1 == 1 and h2 == 1 and c == 0 and g == 0 and d == 0:
                discrepancies.append({
                    'trait_index': idx,
                    'differential': trait_diff,
                    'foundation': f,
                    'type': 'Human=1 / LLM=0 (Human-only consensus)',
                    'detail': 'Fehr=1, Tran=1 | Claude=0, Gemini=0, Deepseek=0'
                })
                
            # Humans agree it is 0, but LLMs all agree it is 1
            elif h1 == 0 and h2 == 0 and c == 1 and g == 1 and d == 1:
                discrepancies.append({
                    'trait_index': idx,
                    'differential': trait_diff,
                    'foundation': f,
                    'type': 'Human=0 / LLM=1 (LLM-only consensus)',
                    'detail': 'Fehr=0, Tran=0 | Claude=1, Gemini=1, Deepseek=1'
                })
                
            # Humans disagree with each other, but LLMs are unanimous
            elif h1 != h2 and (c == g == d):
                llm_val = c
                discrepancies.append({
                    'trait_index': idx,
                    'differential': trait_diff,
                    'foundation': f,
                    'type': 'Human Disagreement / LLM Unanimous',
                    'detail': f'Fehr={h1}, Tran={h2} | LLMs Unanimous={llm_val}'
                })

    disc_df = pd.DataFrame(discrepancies)
    if not disc_df.empty:
        # Show a few examples of each type
        for dtype in disc_df['type'].unique():
            print(f"\n--- Category: {dtype} ---")
            subset = disc_df[disc_df['type'] == dtype].head(5)
            print(f"{'Index':<5} | {'Trait':<30} | {'Foundation':<12} | {'Details':<45}")
            print("-" * 95)
            for _, row in subset.iterrows():
                print(f"{row['trait_index']:<5} | {row['differential']:<30} | {row['foundation']:<12} | {row['detail']:<45}")
    else:
        print("No extreme discrepancies found.")
        
    # Save statistics and discrepancy logs to files
    summary_out_path = RESULTS_DIR / "rating_agreement_summary.csv"
    summary_df.to_csv(summary_out_path, index=False)
    
    disc_out_path = RESULTS_DIR / "rating_discrepancies.csv"
    disc_df.to_csv(disc_out_path, index=False)
    
    print("\n" + "="*95)
    print(f"Outputs successfully saved to:")
    print(f"- Summary table: {summary_out_path.name}")
    print(f"- Discrepancies log: {disc_out_path.name}")
    print("="*95)

if __name__ == "__main__":
    main()
