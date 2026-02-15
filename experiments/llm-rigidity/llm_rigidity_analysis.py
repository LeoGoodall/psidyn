"""
Semantic Transfer Entropy Analysis for LLM-LLM Conversations

Analyses semantic information dynamics across three conditions:
- rigid-rigid
- rigid-flexible  
- flexible-flexible

Outputs full timeseries data for R visualisation with SEM shading.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
import pandas as pd

from te import TransferEntropy
from trident import Droplet

# Configuration
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-3B"
DEFAULT_LOAD_IN_4BIT = True
DEFAULT_LOAD_IN_8BIT = False
DEFAULT_LAG_WINDOW = 5

# Paths
DATA_PATH = Path("llm-rigidity") / "conversations.json"
RESULTS_DIR = Path("llm-rigidity") / "results"

# Output files
TIMESERIES_RESULTS_PATH = RESULTS_DIR / "te_timeseries_full.csv"
CONVERSATION_SUMMARY_PATH = RESULTS_DIR / "te_conversation_summary.csv"
DYAD_METRICS_PATH = RESULTS_DIR / "te_dyad_metrics.csv"


def normalise_condition(condition: str) -> str:
    """Normalise condition names (fix typos like 'rig-flexible')."""
    condition = condition.lower().strip()
    if condition == "rig-flexible":
        return "rigid-flexible"
    return condition


def load_conversations(data_path: Path) -> List[Dict[str, Any]]:
    """Load conversations from JSON file."""
    with open(data_path, "r") as f:
        return json.load(f)


def create_posts_from_turns(turns: List[Dict[str, Any]]) -> List[Droplet]:
    """Convert conversation turns to Droplet objects for TE computation."""
    posts = []
    for turn in turns:
        posts.append(
            Droplet(
                user_id=turn["speaker"],
                timestamp=turn["turn"],
                content=turn["text"],
                post_id=f"turn_{turn['turn']}",
            )
        )
    return posts


def process_conversation(
    conversation: Dict[str, Any],
    estimator: TransferEntropy,
    lag_window: int,
) -> Dict[str, Any]:
    """Process a single conversation and compute transfer entropy metrics."""
    
    conv_id = conversation["conversation_id"]
    condition = normalise_condition(conversation["condition"])
    topic = conversation["topic"]
    turns = conversation["turns"]
    
    # Create posts from turns
    posts = create_posts_from_turns(turns)
    
    if len(posts) < 2:
        return None
    
    # Compute transfer entropy for all dyads (A→B and B→A)
    te_dict, te_timeseries = estimator.compute_all_dyadic_transfer_entropies(
        posts,
        lag_window=lag_window,
        save_te=False,
        save_timeseries=False,
        te_csv_path=None,
        timeseries_csv_path=None,
    )
    
    # Add metadata to timeseries entries
    for row in te_timeseries:
        row["conversation_id"] = conv_id
        row["condition"] = condition
        row["topic"] = topic
        # Determine direction label
        if row["source_user"] == "A" and row["target_user"] == "B":
            row["direction"] = "A_to_B"
        elif row["source_user"] == "B" and row["target_user"] == "A":
            row["direction"] = "B_to_A"
        else:
            row["direction"] = f"{row['source_user']}_to_{row['target_user']}"
    
    # Get dyad-level TE values
    te_a_to_b = float(te_dict.get(("A", "B"), 0.0))
    te_b_to_a = float(te_dict.get(("B", "A"), 0.0))
    
    # Create conversation summary
    conversation_summary = {
        "conversation_id": conv_id,
        "condition": condition,
        "topic": topic,
        "num_turns": len(turns),
        "te_A_to_B": te_a_to_b,
        "te_B_to_A": te_b_to_a,
        "te_bidirectional_mean": (te_a_to_b + te_b_to_a) / 2,
        "te_asymmetry": te_a_to_b - te_b_to_a,
    }
    
    # Create dyad metrics rows
    dyad_metrics = []
    for (source, target), te_value in te_dict.items():
        dyad_metrics.append({
            "conversation_id": conv_id,
            "condition": condition,
            "topic": topic,
            "source_user": source,
            "target_user": target,
            "te_value": float(te_value),
            "direction": f"{source}_to_{target}",
        })
    
    return {
        "timeseries": te_timeseries,
        "summary": conversation_summary,
        "dyad_metrics": dyad_metrics,
    }


def analyse_llm_rigidity_dataset(
    data_path: Path = DATA_PATH,
    output_dir: Path = RESULTS_DIR,
    lag_window: int = DEFAULT_LAG_WINDOW,
    model_name: str = DEFAULT_MODEL_NAME,
    load_in_4bit: bool = DEFAULT_LOAD_IN_4BIT,
    load_in_8bit: bool = DEFAULT_LOAD_IN_8BIT,
) -> Dict[str, pd.DataFrame]:
    """
    Analyse the LLM rigidity conversation dataset.
    
    Args:
        data_path: Path to conversations.json
        output_dir: Directory to save results
        lag_window: Lag window for TE computation
        model_name: HuggingFace model name
        load_in_4bit: Use 4-bit quantisation
        load_in_8bit: Use 8-bit quantisation
        
    Returns:
        Dictionary containing DataFrames for timeseries, summary, and dyad metrics
    """
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load conversations
    print(f"Loading conversations from {data_path}...")
    conversations = load_conversations(data_path)
    print(f"Loaded {len(conversations)} conversations")
    
    # Count by condition
    condition_counts = {}
    for conv in conversations:
        cond = normalise_condition(conv["condition"])
        condition_counts[cond] = condition_counts.get(cond, 0) + 1
    print(f"Conditions: {condition_counts}")
    
    # Initialise estimator
    print(f"Initialising TransferEntropy with {model_name}...")
    estimator = TransferEntropy(
        model_name=model_name,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )
    
    # Process all conversations
    all_timeseries = []
    all_summaries = []
    all_dyad_metrics = []
    
    for conversation in tqdm(conversations, desc="Processing conversations"):
        try:
            result = process_conversation(conversation, estimator, lag_window)
            if result is not None:
                all_timeseries.extend(result["timeseries"])
                all_summaries.append(result["summary"])
                all_dyad_metrics.extend(result["dyad_metrics"])
        except Exception as e:
            print(f"Error processing {conversation.get('conversation_id', 'unknown')}: {e}")
            continue
    
    # Convert to DataFrames
    timeseries_df = pd.DataFrame(all_timeseries)
    summary_df = pd.DataFrame(all_summaries)
    dyad_df = pd.DataFrame(all_dyad_metrics)
    
    # Add turn number to timeseries (extract from target_timestamp)
    if not timeseries_df.empty:
        timeseries_df["turn"] = timeseries_df["target_timestamp"].astype(int)
    
    # Save results
    timeseries_path = output_dir / TIMESERIES_RESULTS_PATH.name
    summary_path = output_dir / CONVERSATION_SUMMARY_PATH.name
    dyad_path = output_dir / DYAD_METRICS_PATH.name
    
    timeseries_df.to_csv(timeseries_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    dyad_df.to_csv(dyad_path, index=False)
    
    print(f"\nResults saved:")
    print(f"  Timeseries: {timeseries_path} ({len(timeseries_df)} rows)")
    print(f"  Summary: {summary_path} ({len(summary_df)} rows)")
    print(f"  Dyad metrics: {dyad_path} ({len(dyad_df)} rows)")
    
    # Print summary statistics
    if not summary_df.empty:
        print("\nSummary statistics by condition:")
        summary_stats = summary_df.groupby("condition").agg({
            "te_A_to_B": ["mean", "std"],
            "te_B_to_A": ["mean", "std"],
            "te_bidirectional_mean": ["mean", "std"],
        }).round(4)
        print(summary_stats)
    
    return {
        "timeseries": timeseries_df,
        "summary": summary_df,
        "dyad_metrics": dyad_df,
        "paths": {
            "timeseries": str(timeseries_path),
            "summary": str(summary_path),
            "dyad_metrics": str(dyad_path),
        },
    }


if __name__ == "__main__":
    print(f"Running LLM Rigidity STE analysis with lag window = {DEFAULT_LAG_WINDOW}")
    results = analyse_llm_rigidity_dataset(
        data_path=DATA_PATH,
        output_dir=RESULTS_DIR,
        lag_window=DEFAULT_LAG_WINDOW,
        model_name=DEFAULT_MODEL_NAME,
        load_in_4bit=DEFAULT_LOAD_IN_4BIT,
        load_in_8bit=DEFAULT_LOAD_IN_8BIT,
    )
