import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from psidyn import Trident, Post


def prepare_conversations(dialog_df):
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
    results_dir = Path("PersuasionForGood/results/prechecks")
    results_dir.mkdir(parents=True, exist_ok=True)

    dialog_df = pd.read_csv(dialog_path)
    conversations = prepare_conversations(dialog_df)
    if not conversations:
        return

    candidate_lags = list(range(1, 26))
    model_name = "meta-llama/Llama-3.2-3B"
    device = "cuda"
    load_in_4bit = True
    surrogate_count = 3
    surrogate_rng = np.random.default_rng(42)

    estimator = Trident(
        model_name=model_name,
        device=device,
        load_in_4bit=load_in_4bit
    )

    per_lag_totals = {}
    per_lag_details = {}
    per_lag_surrogate_totals = {}
    conversation_summary_rows = []
    all_dyad_metric_rows = []
    all_timeseries_rows = []

    for lag in candidate_lags:
        totals = []
        details = []
        dyad_metrics = []
        timeseries_records = []

        for conv_id, posts in tqdm(
            conversations,
            desc=f"Lag {lag}",
            leave=False,
            file=sys.stdout,
        ):
            te_results, ts_rows, info_summary = estimator.compute_all_dyadic_transfer_entropies(
                posts,
                lag_window=lag,
                save_te=False,
                save_timeseries=False,
                te_csv_path=None,
                timeseries_csv_path=None,
                return_information=True,
            )

            forward = float(te_results.get(("0", "1"), 0.0))
            backward = float(te_results.get(("1", "0"), 0.0))
            total_te = forward + backward
            totals.append(total_te)

            detail_row = {
                "conversation_id": conv_id,
                "lag": lag,
                "total_te": total_te,
                "te_0_to_1": forward,
                "te_1_to_0": backward,
                "num_posts": len(posts),
            }
            surrogate_values = []
            if surrogate_count > 0:
                for s_idx in range(surrogate_count):
                    shuffled_indices = np.arange(len(posts))
                    surrogate_rng.shuffle(shuffled_indices)
                    shuffled_posts = [
                        Post(
                            user_id=posts[idx].user_id,
                            timestamp=pos,
                            content=posts[idx].content,
                            post_id=f"{posts[idx].post_id}_shuf{s_idx}",
                        )
                        for pos, idx in enumerate(shuffled_indices)
                    ]
                    surrogate_results, _ = estimator.compute_all_dyadic_transfer_entropies(
                        shuffled_posts,
                        lag_window=lag,
                        save_te=False,
                        save_timeseries=False,
                        te_csv_path=None,
                        timeseries_csv_path=None,
                    )
                    s_forward = float(surrogate_results.get(("0", "1"), 0.0))
                    s_backward = float(surrogate_results.get(("1", "0"), 0.0))
                    surrogate_total = s_forward + s_backward
                    surrogate_values.append(surrogate_total)
                per_lag_surrogate_totals.setdefault(lag, []).extend(surrogate_values)
                if surrogate_values:
                    detail_row["surrogate_mean_total_te"] = float(np.mean(surrogate_values))
                    detail_row["surrogate_std_total_te"] = float(np.std(surrogate_values))
                    detail_row["surrogate_min_total_te"] = float(np.min(surrogate_values))
                    detail_row["surrogate_max_total_te"] = float(np.max(surrogate_values))

            details.append(detail_row)
            conversation_summary_rows.append(detail_row.copy())

            for (src, tgt), metrics in info_summary.items():
                metrics_row = {
                    "conversation_id": conv_id,
                    "lag": lag,
                    "source_user": src,
                    "target_user": tgt,
                    "te_value": float(te_results.get((src, tgt), 0.0)),
                }
                metrics_row.update(metrics)
                dyad_metrics.append(metrics_row)

            for row in ts_rows:
                record = dict(row)
                record["conversation_id"] = conv_id
                record["lag"] = lag
                timeseries_records.append(record)

        per_lag_totals[lag] = totals
        per_lag_details[lag] = details
        all_dyad_metric_rows.extend(dyad_metrics)
        all_timeseries_rows.extend(timeseries_records)

    lag_values = []
    median_totals = []
    surrogate_medians = []
    total_std = []
    surrogate_std = []
    for lag in candidate_lags:
        lag_values.append(lag)
        values = per_lag_totals.get(lag, [])
        if values:
            median_totals.append(float(np.median(values)))
            total_std.append(float(np.std(values)))
        else:
            median_totals.append(float("nan"))
            total_std.append(float("nan"))

        surrogate_vals = per_lag_surrogate_totals.get(lag, [])
        if surrogate_vals:
            surrogate_medians.append(float(np.median(surrogate_vals)))
            surrogate_std.append(float(np.std(surrogate_vals)))
        else:
            surrogate_medians.append(float("nan"))
            surrogate_std.append(float("nan"))

    # Compute median_total_te_minus_surrogate for all lags
    median_te_minus_surrogate = []
    for i in range(len(lag_values)):
        if not np.isnan(median_totals[i]) and not np.isnan(surrogate_medians[i]):
            median_te_minus_surrogate.append(median_totals[i] - surrogate_medians[i])
        else:
            median_te_minus_surrogate.append(float("nan"))

    # Find the maximum value of median_te_minus_surrogate
    valid_values = [v for v in median_te_minus_surrogate if not np.isnan(v)]
    max_te_minus_surrogate = max(valid_values) if valid_values else 0.0
    
    # Find the first lag where:
    # 1. median_total_te_minus_surrogate >= 90% of maximum
    # 2. incremental gain over previous lag < 0.05 bits
    optimal_lag = lag_values[-1]
    threshold_90pct = 0.90 * max_te_minus_surrogate
    incremental_gain_threshold = 0.05
    
    for i in range(1, len(lag_values)):
        curr = median_te_minus_surrogate[i]
        prev = median_te_minus_surrogate[i - 1]
        
        if np.isnan(curr) or np.isnan(prev):
            continue
        
        # Check if current value is >= 90% of maximum
        if curr >= threshold_90pct:
            # Check if incremental gain is < 0.05 bits
            incremental_gain = curr - prev
            if incremental_gain < incremental_gain_threshold:
                optimal_lag = lag_values[i]
                break

    lag_summary = pd.DataFrame({
        "lag": lag_values,
        "median_total_te": median_totals,
    })
    lag_summary["median_diff"] = lag_summary["median_total_te"].diff()
    lag_summary["plateau_start"] = lag_summary["lag"] == optimal_lag
    lag_summary["surrogate_median_total_te"] = surrogate_medians
    lag_summary["median_total_te_minus_surrogate"] = lag_summary["median_total_te"] - lag_summary["surrogate_median_total_te"]
    lag_summary["std_total_te"] = total_std
    lag_summary["std_total_te_surrogate"] = surrogate_std

    conversation_df = None
    if conversation_summary_rows:
        conversation_df = pd.DataFrame(conversation_summary_rows)
        conversation_df.to_csv(results_dir / "te_conversation_summaries.csv", index=False)

    dyad_df = None
    if all_dyad_metric_rows:
        dyad_df = pd.DataFrame(all_dyad_metric_rows)
        dyad_df.to_csv(results_dir / "te_dyad_metrics.csv", index=False)

    if all_timeseries_rows:
        timeseries_df = pd.DataFrame(all_timeseries_rows)
        timeseries_df.to_csv(results_dir / "te_timeseries_records.csv", index=False)

    if dyad_df is not None:
        agg_columns = [
            "median_te_bits_per_token",
            "median_te_normalized_bits_per_token",
            "median_mi_source_bits_per_token",
            "median_mi_target_history_bits_per_token",
            "median_mi_joint_bits_per_token",
            "median_te_normalized_per_post_bits_per_token",
        ]
        available_cols = [col for col in agg_columns if col in dyad_df.columns]
        if available_cols:
            lag_agg = dyad_df.groupby("lag")[available_cols].median().reset_index()
            lag_summary = lag_summary.merge(lag_agg, on="lag", how="left")

    lag_summary.to_csv(results_dir / "te_lag_scan.csv", index=False)
        


if __name__ == "__main__":
    main()
