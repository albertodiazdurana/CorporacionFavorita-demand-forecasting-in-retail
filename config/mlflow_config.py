"""
MLflow Configuration for Demand Forecasting Project

This module provides centralized MLflow configuration to ensure all notebooks
and scripts use the same tracking URI and avoid the mixed setup issue.

Usage in notebooks:
    from config.mlflow_config import setup_mlflow

    tracking_uri, experiment_name = setup_mlflow("my_experiment_name")
    # Now mlflow.log_* calls will use the correct location
"""

from pathlib import Path
import mlflow


def get_tracking_uri():
    """
    Get the centralized MLflow tracking URI.

    Returns:
        str: File URI pointing to mlflow_results directory
    """
    # Get project root (assuming config/ is at root level)
    project_root = Path(__file__).parent.parent
    mlflow_dir = project_root / "mlflow_results"
    mlflow_dir.mkdir(parents=True, exist_ok=True)

    return f"file://{mlflow_dir}"


def setup_mlflow(experiment_name="default"):
    """
    Configure MLflow with centralized tracking URI and set experiment.

    Args:
        experiment_name (str): Name of the MLflow experiment

    Returns:
        tuple: (tracking_uri, experiment_name)

    Example:
        >>> setup_mlflow("feature_engineering")
        ('file:///home/user/project/mlflow_results', 'feature_engineering')
    """
    tracking_uri = get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    print(f"MLflow Configuration:")
    print(f"  Tracking URI: {tracking_uri}")
    print(f"  Experiment: {experiment_name}")

    return tracking_uri, experiment_name


def start_mlflow_ui():
    """
    Print the command to start MLflow UI with correct backend.

    This is a helper function to remind users of the correct command.
    """
    project_root = Path(__file__).parent.parent
    venv_mlflow = project_root / ".venv" / "bin" / "mlflow"
    mlflow_dir = project_root / "mlflow_results"

    print("\nTo start MLflow UI, run:")
    print(f"  {venv_mlflow} ui --backend-store-uri file://{mlflow_dir} --port 5000")
    print("\nOr from project root with activated venv:")
    print(f"  cd {project_root}")
    print(f"  source .venv/bin/activate")
    print(f"  mlflow ui --backend-store-uri file://$(pwd)/mlflow_results --port 5000")
    print("\nThen open: http://127.0.0.1:5000")


if __name__ == "__main__":
    # When run directly, show MLflow UI startup command
    start_mlflow_ui()
