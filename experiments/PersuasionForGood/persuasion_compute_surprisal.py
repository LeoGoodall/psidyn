import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from psidyn import Trident, Post


def prepare_conversations(dialog_df: pd.DataFrame):
    grouped = dialog_df.sort_values(["B2", "Turn", "post_id"]).groupby("B2", sort=False)
    conversations = []
    for conv_id, conv_dialog in grouped:
        posts = []
        for idx, row in enumerate(conv_dialog.itertuples(index=False)):
            content = row.Unit if isinstance(row.Unit, str) else ""
            if not content.strip():
                continue
            posts.append(
                Post(
                    user_id=str(row.B4),
                    timestamp=idx,
                    content=content,
                    post_id=str(row.post_id),
                )
            )
        if len(posts) < 2:
            continue
        user_ids = {p.user_id for p in posts}
        if user_ids != {"0", "1"}:
            continue
        conversations.append((conv_id, posts))
    return conversations


def main():
    data_dir = Path("PersuasionForGood/data")
    dialog_path = data_dir / "dialog.csv"
    results_dir = Path("PersuasionForGood/results/surprisal")
    results_dir.mkdir(parents=True, exist_ok=True)

    dialog_df = pd.read_csv(dialog_path)
    conversations = prepare_conversations(dialog_df)
    if not conversations:
        print("No valid conversations found.", file=sys.stderr)
        return

    lag_window = 16
    estimator = Trident(
        model_name="meta-llama/Llama-3.2-3B",
        device="cuda" if torch.cuda.is_available() else "cpu",  # type: ignore[name-defined]
        load_in_4bit=True,
        load_in_8bit=False,
        normalise_te_with_mi=False,
    )

    timeseries_rows = []
    summary_rows = []

    for conv_id, posts in tqdm(conversations, desc="Surprisal lag 16", leave=False):
        convo_records = []
        for idx, post in enumerate(posts):
            surprisal = estimator.compute_post_surprisal(posts, idx, lag_window)
            if surprisal is None:
                continue
            record = {
                "conversation_id": conv_id,
                "target_post_id": post.post_id,
                "target_user": post.user_id,
                "turn_index": idx,
                "lag": lag_window,
                "surprisal_bits_per_token": surprisal,
            }
            timeseries_rows.append(record)
            convo_records.append(record)

        if convo_records:
            convo_df = pd.DataFrame(convo_records)
            grouped = (
                convo_df.groupby("target_user")["surprisal_bits_per_token"]
                .agg(["median", "mean", "count"])
                .reset_index()
            )
            for _, row in grouped.iterrows():
                summary_rows.append(
                    {
                        "conversation_id": conv_id,
                        "target_user": row["target_user"],
                        "lag": lag_window,
                        "median_surprisal_bits_per_token": row["median"],
                        "mean_surprisal_bits_per_token": row["mean"],
                        "n_posts": int(row["count"]),
                    }
                )

    if timeseries_rows:
        pd.DataFrame(timeseries_rows).to_csv(
            results_dir / "surprisal_timeseries.csv", index=False
        )
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            results_dir / "surprisal_conversation_summary.csv", index=False
        )


if __name__ == "__main__":
    main()
