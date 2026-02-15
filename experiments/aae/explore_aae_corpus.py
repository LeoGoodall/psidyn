"""
Explore the Argument Annotated Essays (AAE) corpus.
Find claims with exactly 2 premises for PID analysis.

Structure:
- S1: Premise 1
- S2: Premise 2
- T: Claim
"""

import os
import re
from collections import defaultdict, Counter
import json

# Path to corpus
CORPUS_DIR = "/Volumes/One Touch/PSIDyn/ArgumentAnnotatedEssays-2.0/brat-project-final"
OUTPUT_FILE = "/Volumes/One Touch/PSIDyn/aae_samples.json"


def parse_ann_file(ann_path, txt_path):
    """Parse a brat .ann file and corresponding .txt file."""

    # Read essay text
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        essay_text = f.read()

    # Read annotations
    with open(ann_path, 'r', encoding='utf-8', errors='ignore') as f:
        ann_lines = f.readlines()

    # Parse entities (T lines)
    entities = {}
    for line in ann_lines:
        line = line.strip()
        if line.startswith('T'):
            # Format: T1\tMajorClaim 503 575\ttext content
            parts = line.split('\t')
            if len(parts) >= 3:
                entity_id = parts[0]
                type_span = parts[1].split()
                entity_type = type_span[0]
                start = int(type_span[1])
                end = int(type_span[2])
                text = parts[2]

                entities[entity_id] = {
                    'type': entity_type,
                    'start': start,
                    'end': end,
                    'text': text
                }

    # Parse relations (R lines)
    relations = []
    for line in ann_lines:
        line = line.strip()
        if line.startswith('R'):
            # Format: R1\tsupports Arg1:T4 Arg2:T3
            parts = line.split('\t')
            if len(parts) >= 2:
                rel_parts = parts[1].split()
                rel_type = rel_parts[0]

                arg1_match = re.search(r'Arg1:(\w+)', parts[1])
                arg2_match = re.search(r'Arg2:(\w+)', parts[1])

                if arg1_match and arg2_match:
                    relations.append({
                        'type': rel_type,
                        'source': arg1_match.group(1),  # Premise
                        'target': arg2_match.group(1)   # Claim
                    })

    return entities, relations


def find_claims_with_premises(entities, relations):
    """Find claims and their supporting premises."""

    # Build mapping: claim -> list of premises
    claim_premises = defaultdict(list)

    for rel in relations:
        if rel['type'] == 'supports':
            source_id = rel['source']
            target_id = rel['target']

            # Check if source is a premise and target is a claim
            if source_id in entities and target_id in entities:
                source_type = entities[source_id]['type']
                target_type = entities[target_id]['type']

                if source_type == 'Premise' and target_type == 'Claim':
                    claim_premises[target_id].append(source_id)

    return claim_premises


def main():
    print("=" * 70)
    print("EXPLORING AAE CORPUS")
    print("=" * 70)

    # Find all essay files
    ann_files = sorted([f for f in os.listdir(CORPUS_DIR) if f.endswith('.ann')])
    print(f"\nFound {len(ann_files)} essays")

    # Statistics
    premise_counts = Counter()
    total_claims = 0
    claims_with_2_premises = []

    for ann_file in ann_files:
        essay_id = ann_file.replace('.ann', '')
        ann_path = os.path.join(CORPUS_DIR, ann_file)
        txt_path = os.path.join(CORPUS_DIR, essay_id + '.txt')

        if not os.path.exists(txt_path):
            continue

        entities, relations = parse_ann_file(ann_path, txt_path)
        claim_premises = find_claims_with_premises(entities, relations)

        for claim_id, premise_ids in claim_premises.items():
            total_claims += 1
            num_premises = len(premise_ids)
            premise_counts[num_premises] += 1

            # Collect claims with exactly 2 premises
            if num_premises == 2:
                claim_text = entities[claim_id]['text']
                premise1_text = entities[premise_ids[0]]['text']
                premise2_text = entities[premise_ids[1]]['text']

                claims_with_2_premises.append({
                    'essay_id': essay_id,
                    'claim_id': claim_id,
                    'claim_text': claim_text,
                    'premise1_id': premise_ids[0],
                    'premise1_text': premise1_text,
                    'premise2_id': premise_ids[1],
                    'premise2_text': premise2_text,
                })

    # Print statistics
    print(f"\nTotal claims: {total_claims}")
    print(f"\nDistribution of premises per claim:")
    for num_premises in sorted(premise_counts.keys()):
        count = premise_counts[num_premises]
        pct = 100 * count / total_claims
        print(f"  {num_premises} premises: {count} ({pct:.1f}%)")

    print(f"\n{'=' * 70}")
    print(f"CLAIMS WITH EXACTLY 2 PREMISES: {len(claims_with_2_premises)}")
    print(f"{'=' * 70}")

    # Show examples
    print("\nExamples:")
    for sample in claims_with_2_premises[:3]:
        print(f"\n--- {sample['essay_id']} ---")
        print(f"CLAIM: {sample['claim_text']}")
        print(f"PREMISE 1: {sample['premise1_text']}")
        print(f"PREMISE 2: {sample['premise2_text']}")

    # Save samples
    print(f"\nSaving to {OUTPUT_FILE}...")
    output = {
        'samples': claims_with_2_premises,
        'metadata': {
            'total_claims': total_claims,
            'claims_with_2_premises': len(claims_with_2_premises),
            'premise_distribution': dict(premise_counts)
        }
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(claims_with_2_premises)} samples")

    # Text length analysis
    if claims_with_2_premises:
        print(f"\n{'=' * 70}")
        print("TEXT LENGTH ANALYSIS")
        print(f"{'=' * 70}")

        claim_lens = [len(s['claim_text']) for s in claims_with_2_premises]
        p1_lens = [len(s['premise1_text']) for s in claims_with_2_premises]
        p2_lens = [len(s['premise2_text']) for s in claims_with_2_premises]

        print(f"\nClaim length: mean={sum(claim_lens)/len(claim_lens):.0f}, min={min(claim_lens)}, max={max(claim_lens)}")
        print(f"Premise 1 length: mean={sum(p1_lens)/len(p1_lens):.0f}, min={min(p1_lens)}, max={max(p1_lens)}")
        print(f"Premise 2 length: mean={sum(p2_lens)/len(p2_lens):.0f}, min={min(p2_lens)}, max={max(p2_lens)}")


if __name__ == "__main__":
    main()
