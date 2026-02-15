import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, get_context
from pathlib import Path

from psidyn import Trident, Post

DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-3B"
DEFAULT_LOAD_IN_4BIT = True
DEFAULT_LOAD_IN_8BIT = False
DEFAULT_LAG_WINDOW = 16
RESULTS_DIR = Path("PersuasionForGood") / "results"
CONVERSATION_RESULTS_PATH = RESULTS_DIR / "persuasion_te_analysis_results.csv"
STRATEGY_RESULTS_PATH = RESULTS_DIR / "persuasion_strategy_te_analysis.csv"
TIMESERIES_RESULTS_PATH = RESULTS_DIR / "persuasion_te_timeseries.csv"


_ESTIMATOR = None
_VALID_STRATEGIES = None

TARGET_STRATEGIES = frozenset({
    'logical-appeal', 'emotion-appeal', 'credibility-appeal',
    'foot-in-the-door', 'self-modeling', 'personal-story',
    'donation-information', 'source-related-inquiry',
    'task-related-inquiry', 'personal-related-inquiry',
})


def _pool_initializer(model_name: str, valid_strategies, load_in_4bit: bool, load_in_8bit: bool):
    global _ESTIMATOR, _VALID_STRATEGIES
    _ESTIMATOR = Trident(
        model_name=model_name,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit
    )
    _VALID_STRATEGIES = frozenset(valid_strategies)


def _process_conversation(payload):
    global _ESTIMATOR, _VALID_STRATEGIES
    if _ESTIMATOR is None:
        raise RuntimeError("TransferEntropyEstimator not initialised in worker process.")

    conv_id, conv_dialog, meta_rows, lag_window = payload

    persuader_row = meta_rows[meta_rows['B4'] == 0]
    persuadee_row = meta_rows[meta_rows['B4'] == 1]
    if persuader_row.empty or persuadee_row.empty:
        return None

    intended = persuader_row['B5'].iloc[0]
    actual = persuader_row['B6'].iloc[0]
    intended = float(intended) if pd.notna(intended) else 0.0
    actual = float(actual) if pd.notna(actual) else 0.0
    persuasion_success = actual - intended

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

    if len(posts) < 2 or set(p.user_id for p in posts) != {'0', '1'}:
        return None

    te_dict, te_timeseries = _ESTIMATOR.compute_all_dyadic_transfer_entropies(
        posts,
        lag_window=lag_window,
        save_te=False,
        save_timeseries=False,
        te_csv_path=None,
        timeseries_csv_path=None,
    )
    for row in te_timeseries:
        row['conversation_id'] = conv_id

    
    te_0_to_1 = float(te_dict.get(('0', '1'), 0.0))
    te_1_to_0 = float(te_dict.get(('1', '0'), 0.0))

    # Strategy extraction uses persuader (B4 == 0) annotations.
    persuader_mask = conv_dialog['B4'] == 0
    strategy_series = conv_dialog.loc[persuader_mask, 'er_label_1'].dropna()
    if _VALID_STRATEGIES:
        strategy_series = strategy_series[strategy_series.isin(_VALID_STRATEGIES)]
    strategy_counts = strategy_series.value_counts().to_dict()
    unique_strategies = list(strategy_counts.keys())


    conversation_row = {
        'conversation_id': conv_id,
        'te_persuader_to_persuadee': te_0_to_1,
        'te_persuadee_to_persuader': te_1_to_0,
        'persuasion_success': persuasion_success,
        'actual_donation': actual,
        'intended_donation': intended,
        'num_posts': len(posts),
        'persuader_posts': sum(p.user_id == '0' for p in posts),
        'persuadee_posts': sum(p.user_id == '1' for p in posts),
        'strategy_counts': strategy_counts,
        'unique_strategies_used': unique_strategies,
    }

    strategy_rows = [
        {
            'conversation_id': conv_id,
            'strategy': strategy,
            'strategy_count': int(count),
            'te_persuader_to_persuadee': te_0_to_1,
            'te_persuadee_to_persuader': te_1_to_0,
            'persuasion_success': persuasion_success,
        }
        for strategy, count in strategy_counts.items()
    ]

    return conversation_row, strategy_rows, te_timeseries


def analyse_persuasion_dataset(
    dialog_csv_path,
    convo_info_csv_path,
    lag_window=DEFAULT_LAG_WINDOW,
    model_name=DEFAULT_MODEL_NAME,
    output_dir=RESULTS_DIR,
    load_in_4bit=DEFAULT_LOAD_IN_4BIT,
    load_in_8bit=DEFAULT_LOAD_IN_8BIT,
):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dialog_df = pd.read_csv(dialog_csv_path)
    convo_df = pd.read_csv(convo_info_csv_path)

    convo_meta = convo_df[['B2', 'B4', 'B5', 'B6']]
    valid_strategies = TARGET_STRATEGIES & frozenset(dialog_df['er_label_1'].dropna().unique())

    conversation_rows = []
    strategy_rows = []
    all_timeseries_rows = []

    grouped_dialog = list(dialog_df.sort_values(['B2', 'Turn', 'post_id']).groupby('B2', sort=False))
    payloads = [
        (
            conv_id,
            conv_dialog.reset_index(drop=True),
            convo_meta[convo_meta['B2'] == conv_id].reset_index(drop=True),
            lag_window,
        )
        for conv_id, conv_dialog in grouped_dialog
    ]

    if payloads:
        ctx = get_context("spawn")
        num_workers = min(cpu_count(), len(payloads))
        with ctx.Pool(
            processes=num_workers,
            initializer=_pool_initializer,
            initargs=(model_name, valid_strategies, load_in_4bit, load_in_8bit),
        ) as pool:
            for result in tqdm(
                pool.imap_unordered(_process_conversation, payloads, chunksize=1),
                total=len(payloads),
                desc="Conversations",
            ):
                if not result:
                    continue
                conv_row, strat_rows, ts_rows = result
                conversation_rows.append(conv_row)
                strategy_rows.extend(strat_rows)
                all_timeseries_rows.extend(ts_rows)

    conversation_df = pd.DataFrame(conversation_rows)
    strategy_df = pd.DataFrame(strategy_rows)
    timeseries_df = pd.DataFrame(all_timeseries_rows)

    conversation_path = output_dir / CONVERSATION_RESULTS_PATH.name
    strategy_path = output_dir / STRATEGY_RESULTS_PATH.name
    timeseries_path = output_dir / TIMESERIES_RESULTS_PATH.name

    conversation_df.to_csv(conversation_path, index=False)
    strategy_df.to_csv(strategy_path, index=False)
    timeseries_df.to_csv(timeseries_path, index=False)

    return {
        'conversation_features': conversation_df,
        'strategy_features': strategy_df,
        'timeseries_features': timeseries_df,
        'paths': {
            'conversation_csv': str(conversation_path),
            'strategy_csv': str(strategy_path),
            'timeseries_csv': str(timeseries_path),
        },
        'lag_window': lag_window,
        'model_name': model_name,
        'load_in_4bit': load_in_4bit,
        'load_in_8bit': load_in_8bit,
    }


if __name__ == "__main__":
    print("Running STE estimation with lag window =", DEFAULT_LAG_WINDOW)
    analyse_persuasion_dataset(
        "PersuasionForGood/dialog.csv",
        "PersuasionForGood/convo_info.csv",
        lag_window=DEFAULT_LAG_WINDOW,
        model_name=DEFAULT_MODEL_NAME,
        output_dir=RESULTS_DIR,
        load_in_4bit=DEFAULT_LOAD_IN_4BIT,
        load_in_8bit=DEFAULT_LOAD_IN_8BIT,
    )
