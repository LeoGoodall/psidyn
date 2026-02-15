"""
Compute semantic similarity between premise pairs in AAE samples.

Uses sentence-transformers to embed premises and compute cosine similarity.
This runs locally (no GPU required) and merges with existing PID results.

Input: aae_samples.json, pid_aae_results.csv
Output: pid_aae_results_with_similarity.csv
"""

import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# File paths
SAMPLES_FILE = "aae_samples.json"
PID_RESULTS_FILE = "pid_aae_results.csv"
OUTPUT_FILE = "pid_aae_results_with_similarity.csv"

# Model for embeddings (small, fast, runs on CPU)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_samples():
    """Load AAE samples."""
    print("Loading samples...")
    with open(SAMPLES_FILE, 'r') as f:
        data = json.load(f)
    samples = data.get('samples', [])
    print(f"  Loaded {len(samples)} samples")
    return samples


def compute_similarities(samples, model):
    """Compute cosine similarity between premise pairs."""
    print("\nComputing premise similarities...")

    results = []
    for sample in tqdm(samples, desc="Embedding premises"):
        p1_text = sample['premise1_text']
        p2_text = sample['premise2_text']
        claim_text = sample['claim_text']

        # Embed all three texts
        embeddings = model.encode([p1_text, p2_text, claim_text])

        # Cosine similarities
        p1_p2_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0, 0]
        p1_claim_sim = cosine_similarity([embeddings[0]], [embeddings[2]])[0, 0]
        p2_claim_sim = cosine_similarity([embeddings[1]], [embeddings[2]])[0, 0]

        results.append({
            'essay_id': sample.get('essay_id', ''),
            'claim_id': sample.get('claim_id', ''),
            'premise_similarity': float(p1_p2_sim),
            'p1_claim_similarity': float(p1_claim_sim),
            'p2_claim_similarity': float(p2_claim_sim),
        })

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("COMPUTING PREMISE SIMILARITY FOR AAE SAMPLES")
    print("=" * 70)

    # Load samples
    samples = load_samples()

    # Load embedding model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("  Model loaded")

    # Compute similarities
    sim_df = compute_similarities(samples, model)

    # Load existing PID results
    print(f"\nLoading PID results from {PID_RESULTS_FILE}...")
    pid_df = pd.read_csv(PID_RESULTS_FILE)
    print(f"  Loaded {len(pid_df)} rows")

    # Merge on essay_id and claim_id
    print("\nMerging similarity with PID results...")
    merged_df = pid_df.merge(sim_df, on=['essay_id', 'claim_id'], how='left')

    # Quick correlation check
    valid = merged_df.dropna(subset=['synergy', 'premise_similarity'])
    if len(valid) > 0:
        corr = valid['synergy'].corr(valid['premise_similarity'])
        print(f"\n  Quick check - Synergy ~ Premise similarity: r = {corr:.4f}")

        corr_red = valid['redundancy'].corr(valid['premise_similarity'])
        print(f"  Quick check - Redundancy ~ Premise similarity: r = {corr_red:.4f}")

    # Save
    print(f"\nSaving to {OUTPUT_FILE}...")
    merged_df.to_csv(OUTPUT_FILE, index=False)
    print(f"  Saved {len(merged_df)} rows")

    # Summary stats
    print("\n" + "=" * 70)
    print("SIMILARITY STATISTICS")
    print("=" * 70)
    print(f"\nPremise-Premise similarity:")
    print(f"  Mean: {sim_df['premise_similarity'].mean():.4f}")
    print(f"  SD:   {sim_df['premise_similarity'].std():.4f}")
    print(f"  Min:  {sim_df['premise_similarity'].min():.4f}")
    print(f"  Max:  {sim_df['premise_similarity'].max():.4f}")

    print(f"\nPremise1-Claim similarity:")
    print(f"  Mean: {sim_df['p1_claim_similarity'].mean():.4f}")
    print(f"  SD:   {sim_df['p1_claim_similarity'].std():.4f}")

    print(f"\nPremise2-Claim similarity:")
    print(f"  Mean: {sim_df['p2_claim_similarity'].mean():.4f}")
    print(f"  SD:   {sim_df['p2_claim_similarity'].std():.4f}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"\nOutput: {OUTPUT_FILE}")
    print("Next: Run updated aae_analysis.R")


if __name__ == "__main__":
    main()
