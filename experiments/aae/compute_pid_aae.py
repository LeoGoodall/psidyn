"""
Compute PID atoms for AAE corpus samples.

For each sample:
- Source 1: Premise 1
- Source 2: Premise 2
- Target: Claim

Computes both MMI and CCS redundancy decompositions from the same
LLM forward passes (the raw information terms i1, i2, i12 are
identical; only the redundancy functional differs).

After computing real samples, generates a permutation null model
by shuffling premise-claim pairings across essays, breaking the
argumentative relationship while preserving marginal text statistics.

Input: aae_samples.json
Output: pid_aae_results.csv, pid_aae_null_results.csv
"""

import json
import csv
import random
import numpy as np
from tqdm import tqdm
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pid import PID
from trident import Droplet

# File paths
SAMPLES_FILE = "aae_samples.json"
OUTPUT_CSV = "pid_aae_results.csv"
NULL_OUTPUT_CSV = "pid_aae_null_results.csv"

# PID model config
MODEL_NAME = "meta-llama/Llama-3.2-3B"
MARGINALIZATION_METHOD = "omit"  # "omit" (true marginals) or "mask" (attention masking)

# Null model config
N_NULL_PERMUTATIONS = 1  # Number of shuffled datasets to generate
RANDOM_SEED = 42


def load_samples():
    """Load AAE samples."""
    print("Loading samples...")
    with open(SAMPLES_FILE, 'r') as f:
        data = json.load(f)
    samples = data.get('samples', [])
    print(f"  Loaded {len(samples)} samples")
    return samples


def compute_pid_for_sample(pid_model: PID, sample: dict, method: str = MARGINALIZATION_METHOD) -> dict:
    """
    Compute PID atoms for a single sample using both MMI and CCS.

    Returns a dict with the raw information terms, MMI atoms, and CCS atoms.
    CCS is derived from the same i1/i2/i12 values (no extra LLM calls).
    """
    droplets = [
        Droplet(
            user_id="premise1",
            timestamp=0,
            content=sample['premise1_text'],
            post_id=sample.get('premise1_id', 'p1')
        ),
        Droplet(
            user_id="premise2",
            timestamp=1,
            content=sample['premise2_text'],
            post_id=sample.get('premise2_id', 'p2')
        ),
        Droplet(
            user_id="claim",
            timestamp=2,
            content=sample['claim_text'],
            post_id=sample.get('claim_id', 'c')
        ),
    ]

    # Compute MMI PID (includes raw i1, i2, i12)
    result = pid_model.compute_pointwise_pid(
        posts=droplets,
        source_user_1="premise1",
        source_user_2="premise2",
        target_user="claim",
        target_post_idx=2,
        lag_window=2,
        redundancy="mmi",
        method=method,
    )

    # Derive CCS atoms from the same raw information terms
    i1 = result.get('i_y_x1_bits_per_token', 0.0)
    i2 = result.get('i_y_x2_bits_per_token', 0.0)
    i12 = result.get('i_y_x1x2_bits_per_token', 0.0)

    ccs_red = pid_model.redundancy_ccs_pointwise(i1, i2, i12)
    ccs_unq1 = i1 - ccs_red
    ccs_unq2 = i2 - ccs_red
    ccs_syn = i12 - ccs_unq1 - ccs_unq2 - ccs_red

    result['ccs_red_bits_per_token'] = ccs_red
    result['ccs_unq_x1_bits_per_token'] = ccs_unq1
    result['ccs_unq_x2_bits_per_token'] = ccs_unq2
    result['ccs_syn_bits_per_token'] = ccs_syn

    return result


def build_result_row(sample: dict, pid_result: dict) -> dict:
    """Build a CSV row from a sample and its PID result."""
    return {
        'essay_id': sample.get('essay_id', ''),
        'claim_id': sample.get('claim_id', ''),
        'premise1_id': sample.get('premise1_id', ''),
        'premise2_id': sample.get('premise2_id', ''),
        'claim_len': len(sample['claim_text']),
        'premise1_len': len(sample['premise1_text']),
        'premise2_len': len(sample['premise2_text']),
        'total_premise_len': len(sample['premise1_text']) + len(sample['premise2_text']),
        # Raw information terms
        'i_y_x1': pid_result.get('i_y_x1_bits_per_token', np.nan),
        'i_y_x2': pid_result.get('i_y_x2_bits_per_token', np.nan),
        'i_y_x1x2': pid_result.get('i_y_x1x2_bits_per_token', np.nan),
        # MMI decomposition
        'redundancy': pid_result.get('red_bits_per_token', np.nan),
        'unique_x1': pid_result.get('unq_x1_bits_per_token', np.nan),
        'unique_x2': pid_result.get('unq_x2_bits_per_token', np.nan),
        'synergy': pid_result.get('syn_bits_per_token', np.nan),
        # CCS decomposition
        'ccs_redundancy': pid_result.get('ccs_red_bits_per_token', np.nan),
        'ccs_unique_x1': pid_result.get('ccs_unq_x1_bits_per_token', np.nan),
        'ccs_unique_x2': pid_result.get('ccs_unq_x2_bits_per_token', np.nan),
        'ccs_synergy': pid_result.get('ccs_syn_bits_per_token', np.nan),
        'token_count': pid_result.get('scored_token_count', 0),
    }


def build_nan_row(sample: dict) -> dict:
    """Build a CSV row with NaN values for a failed sample."""
    return {
        'essay_id': sample.get('essay_id', ''),
        'claim_id': sample.get('claim_id', ''),
        'premise1_id': sample.get('premise1_id', ''),
        'premise2_id': sample.get('premise2_id', ''),
        'claim_len': len(sample['claim_text']),
        'premise1_len': len(sample['premise1_text']),
        'premise2_len': len(sample['premise2_text']),
        'total_premise_len': len(sample['premise1_text']) + len(sample['premise2_text']),
        'i_y_x1': np.nan, 'i_y_x2': np.nan, 'i_y_x1x2': np.nan,
        'redundancy': np.nan, 'unique_x1': np.nan, 'unique_x2': np.nan,
        'synergy': np.nan,
        'ccs_redundancy': np.nan, 'ccs_unique_x1': np.nan,
        'ccs_unique_x2': np.nan, 'ccs_synergy': np.nan,
        'token_count': 0,
    }


def save_results(results, output_path):
    """Save results list to CSV."""
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def generate_null_samples(samples, n_permutations, rng):
    """
    Generate shuffled premise-claim pairings for the null model.

    For each real sample, randomly assigns premises from OTHER essays,
    breaking the argumentative relationship while preserving marginal
    text statistics.
    """
    # Pool all premises by essay
    premises_by_essay = {}
    for s in samples:
        eid = s['essay_id']
        if eid not in premises_by_essay:
            premises_by_essay[eid] = []
        premises_by_essay[eid].append({
            'id': s['premise1_id'],
            'text': s['premise1_text'],
        })
        premises_by_essay[eid].append({
            'id': s['premise2_id'],
            'text': s['premise2_text'],
        })

    # Deduplicate premises within each essay
    for eid in premises_by_essay:
        seen = set()
        unique = []
        for p in premises_by_essay[eid]:
            if p['id'] not in seen:
                seen.add(p['id'])
                unique.append(p)
        premises_by_essay[eid] = unique

    # Build pool of all premises from other essays (for each essay)
    all_essays = list(premises_by_essay.keys())

    null_samples = []
    for perm_id in range(n_permutations):
        for s in samples:
            # Collect premises from all OTHER essays
            other_premises = []
            for eid in all_essays:
                if eid != s['essay_id']:
                    other_premises.extend(premises_by_essay[eid])

            # Sample 2 premises without replacement
            chosen = rng.sample(other_premises, 2)

            null_samples.append({
                'permutation_id': perm_id,
                'essay_id': s['essay_id'],
                'claim_id': s['claim_id'],
                'claim_text': s['claim_text'],
                'premise1_id': chosen[0]['id'],
                'premise1_text': chosen[0]['text'],
                'premise2_id': chosen[1]['id'],
                'premise2_text': chosen[1]['text'],
            })

    return null_samples


def print_summary(results, label=""):
    """Print summary statistics for a set of results."""
    valid = [r for r in results if not np.isnan(r['synergy']) and r['token_count'] > 0]
    print(f"\n{label} Samples with valid PID: {len(valid)}/{len(results)}")

    if not valid:
        return

    synergy_vals = [r['synergy'] for r in valid]
    print(f"  MMI Synergy: mean={np.mean(synergy_vals):.4f}, std={np.std(synergy_vals):.4f}")

    red_vals = [r['redundancy'] for r in valid]
    print(f"  MMI Redundancy: mean={np.mean(red_vals):.4f}, std={np.std(red_vals):.4f}")

    unq1_vals = [r['unique_x1'] for r in valid]
    unq2_vals = [r['unique_x2'] for r in valid]
    print(f"  MMI Unique: P1={np.mean(unq1_vals):.4f}, P2={np.mean(unq2_vals):.4f}")

    ccs_syn_vals = [r['ccs_synergy'] for r in valid]
    print(f"  CCS Synergy: mean={np.mean(ccs_syn_vals):.4f}, std={np.std(ccs_syn_vals):.4f}")

    ccs_red_vals = [r['ccs_redundancy'] for r in valid]
    print(f"  CCS Redundancy: mean={np.mean(ccs_red_vals):.4f}, std={np.std(ccs_red_vals):.4f}")


def main():
    print("=" * 70)
    print("COMPUTING PID ATOMS FOR AAE SAMPLES")
    print("=" * 70)

    # Load samples
    samples = load_samples()

    # Initialize PID model
    print("\nInitializing PID model...")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Marginalization: {MARGINALIZATION_METHOD}")
    print(f"  Redundancy: MMI + CCS (from same forward passes)")
    pid_model = PID(model_name=MODEL_NAME)
    print("  Model loaded")

    # ── Phase 1: Compute PID for real samples ────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 1: REAL SAMPLES")
    print("=" * 70)

    results = []
    for sample in tqdm(samples, desc="Real samples"):
        try:
            pid_result = compute_pid_for_sample(pid_model, sample)
            results.append(build_result_row(sample, pid_result))
        except Exception as e:
            print(f"\n  Error processing {sample['essay_id']}/{sample['claim_id']}: {e}")
            results.append(build_nan_row(sample))

    print(f"\nSaving real results to {OUTPUT_CSV}...")
    save_results(results, OUTPUT_CSV)
    print(f"  Saved {len(results)} rows")
    print_summary(results, label="REAL:")

    # ── Phase 2: Permutation null model ──────────────────────────────
    print("\n" + "=" * 70)
    print(f"PHASE 2: PERMUTATION NULL MODEL ({N_NULL_PERMUTATIONS} permutation(s))")
    print("=" * 70)

    rng = random.Random(RANDOM_SEED)
    null_samples = generate_null_samples(samples, N_NULL_PERMUTATIONS, rng)
    print(f"  Generated {len(null_samples)} null samples")

    null_results = []
    for ns in tqdm(null_samples, desc="Null samples"):
        try:
            pid_result = compute_pid_for_sample(pid_model, ns)
            row = build_result_row(ns, pid_result)
            row['permutation_id'] = ns['permutation_id']
            null_results.append(row)
        except Exception as e:
            print(f"\n  Error processing null sample: {e}")
            row = build_nan_row(ns)
            row['permutation_id'] = ns['permutation_id']
            null_results.append(row)

    print(f"\nSaving null results to {NULL_OUTPUT_CSV}...")
    save_results(null_results, NULL_OUTPUT_CSV)
    print(f"  Saved {len(null_results)} rows")
    print_summary(null_results, label="NULL:")

    # ── Done ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"  Real results: {OUTPUT_CSV}")
    print(f"  Null results: {NULL_OUTPUT_CSV}")
    print("\nNext step: Rscript aae_analysis.R")


if __name__ == "__main__":
    main()
