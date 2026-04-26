import os
import shutil
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from .helpers import setup_mlflow as setup_env_mlflow

CLIENT: Optional[MlflowClient] = None


def setup_mlflow(path: str = "./mlflow_creds/local.yaml") -> None:
    """Initialize MLflow client with credentials from a YAML configuration file."""
    config = setup_env_mlflow(path)
    mlflow.set_tracking_uri(config["uri"])
    global CLIENT
    CLIENT = MlflowClient()

def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculate accuracy as percentage of correct predictions.

    Args:
        y_pred: Predicted values (will be rounded to int)
        y_true: Ground truth values

    Returns:
        Accuracy in percent (0-100)
    """
    y_pred = np.rint(y_pred).astype(int)
    n = len(y_true)
    return 100 * np.sum(y_true == y_pred) / n


def grb_score(y_pred: np.ndarray, y_true: np.ndarray, abs: bool = False) -> float:
    """Calculate Gross Rating Bias (GRB) as percentage error in total sum.

    Args:
        y_pred: Predicted values
        y_true: Ground truth values
        abs: If True, return absolute GRB

    Returns:
        GRB as percentage, or NaN if y_true sum is 0
    """
    diff = y_pred - y_true
    res = 100 * np.sum(diff) / np.sum(y_true) if np.sum(y_true) != 0 else np.nan
    return res if not abs else np.abs(res)


def mae_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculate Mean Absolute Error.

    Args:
        y_pred: Predicted values
        y_true: Ground truth values

    Returns:
        Mean absolute error
    """
    return np.mean(np.abs(y_pred - y_true))


def smape_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error.

    Args:
        y_pred: Predicted values
        y_true: Ground truth values

    Returns:
        SMAPE as percentage
    """
    res = np.abs(y_pred - y_true) / (0.5 * (y_pred + y_true) + 1e-8)
    return 100 * np.sum(res) / len(y_pred)

def eval_csv(csv_path: Union[str, Path]) -> Dict[str, float]:
    """Evaluate a CSV file containing predictions and compute various metrics.

    Expects b_pred and a_pred columns for predictions and n_boarding and n_alighting for labels.
    Computes accuracy, SMAPE, MAE, and GRB for both boarding and alighting, plus aggregates.

    Args:
        csv_path: Path to the CSV file

    Returns:
        Dictionary with keys: acc, smape, mae, grb, abs_grb and _b/_a suffixed variants
    """
    csv_path = Path(csv_path)
    res = pd.read_csv(csv_path)

    # Extract labels and predictions, round predictions to integer
    p_b = np.round(res.b_pred)
    p_a = np.round(res.a_pred)
    n_b = res.n_boarding
    n_a = res.n_alighting

    # Calculate metrics for boarding and alighting
    acc_b, acc_a = accuracy(p_b, n_b), accuracy(p_a, n_a)
    smape_b, smape_a = smape_score(p_b, n_b), smape_score(p_a, n_a)
    mae_b, mae_a = mae_score(p_b, n_b), mae_score(p_a, n_a)
    grb_b, grb_a = grb_score(p_b, n_b), grb_score(p_a, n_a)
    abs_grb_b, abs_grb_a = grb_score(p_b, n_b, abs=True), grb_score(p_a, n_a, abs=True)

    # Aggregate metrics
    result = {
        "acc": (acc_a + acc_b) * 0.5,
        "smape": (smape_a + smape_b) * 0.5,
        "mae": (mae_a + mae_b) * 0.5,
        "grb": (grb_a + grb_b) * 0.5,
        "abs_grb": (abs_grb_a + abs_grb_b) * 0.5,
        "acc_b": acc_b,
        "acc_a": acc_a,
        "smape_b": smape_b,
        "smape_a": smape_a,
        "mae_b": mae_b,
        "mae_a": mae_a,
        "grb_b": grb_b,
        "grb_a": grb_a,
        "abs_grb_b": abs_grb_b,
        "abs_grb_a": abs_grb_a,
    }

    return result

def eval_csv_folder(folder_path: Union[str, Path], verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all CSVs in a folder and return results with summary statistics.

    Args:
        folder_path: Path to folder containing CSV files
        verbose: If True, print summary statistics

    Returns:
        Tuple of (results_df, summary_df) where:
            - results_df: One row per CSV with all computed metrics
            - summary_df: Summary statistics (mean, std, min, max, percentiles) for each metric
    """
    folder_path = Path(folder_path)
    results = []

    for csvfile in folder_path.rglob("*.csv"):
        try:
            result = eval_csv(csvfile)
            result["csv_name"] = csvfile.name.replace(".csv", "")
            results.append(result)
        except Exception as e:
            print(f"Error processing {csvfile.name}: {e}")

    results_df = pd.DataFrame(results)
    results_df["csv_name"] = results_df["csv_name"].astype("string")

    summary = results_df.describe(percentiles=[0.25, 0.5, 0.75])

    if verbose:
        print("\nSummary statistics:")
        print(summary)

    return results_df, summary

def get_runs_from_experiment(experiment_name: str) -> pd.DataFrame:
    """Retrieve all runs from a specified MLflow experiment.

    Args:
        experiment_name: Name of the MLflow experiment

    Returns:
        DataFrame containing all runs with their metadata and metrics

    Raises:
        ValueError: If experiment not found
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    return runs

def download_csvs(experiment_name: str, output_dir: Union[str, Path], sleep: int = 5) -> None:
    """Download predictions.csv from all runs in an MLflow experiment.

    For local MLflow: copies files directly from artifact directories.
    For remote MLflow: uses MLflow client to download artifacts.

    Args:
        experiment_name: Name of the MLflow experiment
        output_dir: Directory where CSV files will be saved
        sleep: Sleep duration (seconds) between remote downloads to avoid overwhelming servers
    """
    rows = get_runs_from_experiment(experiment_name).itertuples()
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    os.makedirs(output_dir, exist_ok=True)

    # Detect if using a local MLflow setup
    tracking_uri = mlflow.get_tracking_uri()
    is_local = tracking_uri.startswith("file://")

    for row in rows:
        run_id = row.run_id
        try:
            target_path = os.path.join(output_dir, f"{run_id}.csv")

            # Skip if already downloaded/copied
            if os.path.exists(target_path):
                print(f"Skipping run {run_id}: {run_id}.csv found locally.")
                continue

            if is_local:
                # LOCAL MODE: directly copy predictions.csv
                local_base = tracking_uri.replace("file://", "")
                artifact_path = os.path.join(local_base, experiment_id, run_id, "artifacts", "predictions.csv")
                print(artifact_path)

                if not os.path.exists(artifact_path):
                    print(f"Skipping run {run_id}: predictions.csv not found in local MLflow dir.")
                    continue

                shutil.copy2(artifact_path, target_path)
                print(f"Copied predictions.csv for run {run_id} -> {target_path}")

            else:
                # REMOTE MODE: use MLflow client to download
                artifacts = CLIENT.list_artifacts(run_id)
                artifact_files = [f.path for f in artifacts]
                if "predictions.csv" not in artifact_files:
                    print(f"Skipping run {run_id}: predictions.csv not found in remote repo.")
                    continue

                local_path = CLIENT.download_artifacts(run_id, "predictions.csv", output_dir)
                time.sleep(sleep)

                os.rename(local_path, target_path)
                print(f"Downloaded predictions.csv for run {run_id} -> {target_path}")

        except Exception as e:
            print(f"Skipping run {run_id}: {e}")

    print("Download complete.")

def get_joined_results(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Merge CSV evaluation results with MLflow run metadata.

    Args:
        csv_path: Path to directory containing CSV result files

    Returns:
        DataFrame with CSV metrics joined with MLflow run parameters and metrics
    """
    res = eval_csv_folder(csv_path, verbose=False)
    csv_df = res[0].rename(columns={"csv_name": "run_id"})

    # Get mlflow run df
    run_df = mlflow.search_runs(search_all_experiments=True).astype("string")

    joined_df = run_df.merge(csv_df, on="run_id", how="inner")

    return joined_df

CSV_METRICS = {
    "acc": "acc",
    "smape": "smape",
    "mae": "mae",
    "grb": "grb",
    "abs_grb": "abs_grb",
    "acc_b": "acc_b",
    "acc_a": "acc_a",
    "smape_b": "smape_b",
    "smape_a": "smape_a",
    "mae_b": "mae_b",
    "mae_a": "mae_a",
    "grb_b": "grb_b",
    "grb_a": "grb_a",
    "abs_grb_b": "abs_grb_b",
    "abs_grb_a": "abs_grb_a",
}

MLFLOW_METRICS = {
    "metrics.Acc_a": "acc_a",
    "metrics.Acc_b": "acc_b",
    "metrics.avg_Acc": "acc",
    "metrics.MAE_a": "mae_a",
    "metrics.MAE_b": "mae_b",
    "metrics.avg_MAE": "mae",
    "metrics.SMAPE_a": "smape_a",
    "metrics.SMAPE_b": "smape_b",
    "metrics.avg_SMAPE": "smape",
    "metrics.GRB_a": "grb_a",
    "metrics.GRB_b": "grb_b",
    "metrics.avg_GRB": "grb",
    "metrics.abs_grb_a": "abs_grb_a",
    "metrics.abs_grb_b": "abs_grb_b",
    "metrics.avg_abs_grb": "abs_grb",
    "metrics.grbacc_a": "grbacc_a",
    "metrics.grbacc_b": "grbacc_b",
    "metrics.avg_grbacc": "grbacc",
}

PARAMS_MAPPING = {
    "params.trainer.logger.experiment_name": "experiment_name",
    "run_id": "run_id",
    "params.dm.random_seed": "seed",
    "params.model.intermediate.embedding_dim": "embedding_dim",
    "params.model.intermediate.config.embedding_dim": "embedding_dim_xlstm",
    "params.model.intermediate.num_layers": "num_layers",
    "params.model.intermediate.config.num_blocks": "num_layers_xlstm",
    "params.model.intermediate._target_": "intmdt_target",
    "params.model.intermediate.config.slstm_at": "slstm_at",
    "params.model.input._target_": "input_type",
    "params.dm.train_ds_conf.hdf5_path": "hdf5_path",
    "params.ckpt_path": "model_selection",
}


def preprocess_df(
    df: pd.DataFrame,
    params_mapping: Dict[str, str] = PARAMS_MAPPING,
    metrics_mapping: Dict[str, str] = CSV_METRICS,
    long: bool = False,
) -> pd.DataFrame:
    """Preprocess and standardize experiment results DataFrame.

    Renames columns, extracts model configurations, and optionally converts to long format.

    Args:
        df: Input DataFrame with raw column names
        params_mapping: Dict mapping raw param column names to standardized names
        metrics_mapping: Dict mapping raw metric column names to standardized names
        long: If True, convert to long format (metric, value) with id_vars

    Returns:
        Preprocessed DataFrame with standardized columns
    """
    # Always work on a copy to avoid mutating the original df
    df = df.copy()

    # Merge mappings
    MAPPING = params_mapping.copy()
    MAPPING.update(metrics_mapping)
    value_vars = list(metrics_mapping.values())

    # Rename columns that exist
    existing_cols_mapping = {k: v for k, v in MAPPING.items() if k in df.columns}
    if existing_cols_mapping:
        df = df.rename(columns=existing_cols_mapping)

    # Keep only renamed columns that exist
    existing_target_cols = [v for v in existing_cols_mapping.values() if v in df.columns]
    if existing_target_cols:
        df = df.loc[:, existing_target_cols]

    # Shorten hdf5 path names
    if "hdf5_path" in df.columns:
        df["hdf5_path"] = df["hdf5_path"].apply(
            lambda x: Path(x).name.replace(".hdf5", "").replace("data_", "") if pd.notna(x) else x
        )

    # Determine LSTM type from model configuration
    if "lstm_type" not in df.columns:

        def get_lstm_type(row: pd.Series) -> str:
            if "slstm_at" in df.columns and not pd.isna(row.get("slstm_at")):
                return "slstm"
            elif row.get("intmdt_target") == "napc.model.LSTMModuleFast":
                return "lstm"
            else:
                return "mlstm"

        df["lstm_type"] = df.apply(get_lstm_type, axis=1)

    # Drop intermediate columns
    df = df.drop(columns=[c for c in ["slstm_at", "intmdt_target"] if c in df.columns])

    # Standardize input type
    if "input_type" in df.columns:
        df["input_type"] = df["input_type"].apply(lambda x: "conv" if x == "napc.model.ConvInput" else "dense")

    # Unify num_layers columns
    if "num_layers" in df.columns:
        if "num_layers_xlstm" in df.columns:
            df["num_layers"] = df.apply(
                lambda row: row["num_layers_xlstm"] if pd.notna(row["num_layers_xlstm"]) else row["num_layers"],
                axis=1,
            )
            df = df.drop(columns=["num_layers_xlstm"])

    # Unify embedding_dim columns
    if "embedding_dim" in df.columns:
        if "embedding_dim_xlstm" in df.columns:
            df["embedding_dim"] = df.apply(
                lambda row: row["embedding_dim_xlstm"] if pd.notna(row["embedding_dim_xlstm"]) else row["embedding_dim"],
                axis=1,
            )
            df = df.drop(columns=["embedding_dim_xlstm"])

    # Map model selection strategy
    if "model_selection" in df.columns:

        def _map_model_selection(x: str) -> str:
            if ("best" in x or "Acc" in x) and ("grbacc" not in x):
                return "accuracy"
            elif "grbacc" in x:
                return "grbaware_accuracy"
            else:
                return "abs_grb"

        df["model_selection"] = df["model_selection"].apply(_map_model_selection)

    # Optional long format conversion
    if long and all(v in df.columns for v in value_vars):
        id_vars = [col for col in df.columns if col not in value_vars]
        df = df.melt(id_vars=id_vars, value_vars=value_vars, var_name="metric", value_name="value")

    return df


def get_metric_history(run_id: str, metric_name: str) -> pd.DataFrame:
    """Retrieve metric history for a specific run and metric.

    Args:
        run_id: MLflow run ID
        metric_name: Name of the metric to retrieve history for

    Returns:
        DataFrame with columns [timestamp, step, value, run_id] sorted by step
    """
    metric_history = CLIENT.get_metric_history(run_id, metric_name)
    values = [(m.timestamp, m.step, m.value) for m in metric_history]
    df_metrics = pd.DataFrame(values, columns=["timestamp", "step", "value"])
    df_metrics["timestamp"] = pd.to_datetime(df_metrics["timestamp"], unit="ms")
    df_metrics["run_id"] = run_id
    df_metrics = df_metrics.sort_values("step").reset_index(drop=True)
    return df_metrics

def sync_csvs(
    sync_dir: Union[str, Path] = "results",
    sleep: int = 1,
) -> Dict[str, List[str]]:
    """Synchronize local experiment CSVs with remote MLflow state.

    Downloads new experiments, updates existing ones, and deletes outdated local files.

    Args:
        sync_dir: Directory to sync experiments to
        sleep: Sleep duration (seconds) between downloads

    Returns:
        Dict with 'downloaded', 'deleted', and 'updated' experiment names
    """
    run_df = mlflow.search_runs(search_all_experiments=True).astype("string")

    ex_names_remote = (
        run_df["params.trainer.logger.experiment_name"]
        .dropna()
        .unique()
        .tolist()
    )

    # Ensure sync dir exists
    os.makedirs(sync_dir, exist_ok=True)

    ex_names_local = os.listdir(sync_dir)

    to_download = set(ex_names_remote) - set(ex_names_local)
    to_delete = set(ex_names_local) - set(ex_names_remote)
    to_update = set(ex_names_local).intersection(set(ex_names_remote))

    # Download new experiments
    for ex_name in to_download:
        print(f"Downloading experiment {ex_name}...")
        p = Path(sync_dir) / ex_name
        download_csvs(ex_name, output_dir=p, sleep=sleep)

    # Delete experiments no longer remote
    for ex_name in to_delete:
        p = Path(sync_dir) / ex_name
        print(f"Deleting experiment {ex_name}...")
        shutil.rmtree(p)

    # Update existing experiments
    for ex_name in to_update:
        print(f"Updating experiment {ex_name}...")
        p = Path(sync_dir) / ex_name
        download_csvs(ex_name, output_dir=p, sleep=sleep)

        # Remove local runs that no longer exist remotely
        run_ids_local = [f.replace(".csv", "") for f in os.listdir(p) if f.endswith(".csv")]

        run_ids_remote = run_df[run_df["params.trainer.logger.experiment_name"] == ex_name]["run_id"].tolist()

        to_delete_runs = set(run_ids_local) - set(run_ids_remote)

        for run_id in to_delete_runs:
            f = p / f"{run_id}.csv"
            print(f"Deleting {f}...")
            os.remove(f)

    return {
        "downloaded": sorted(to_download),
        "deleted": sorted(to_delete),
        "updated": sorted(to_update),
    }

