#!/bin/bash
# Start MLflow UI with correct backend configuration
#
# Usage: ./scripts/start_mlflow_ui.sh [port]
# Default port: 5000

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default port
PORT=${1:-5000}

# MLflow configuration
MLFLOW_DIR="$PROJECT_ROOT/mlflow_results"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
VENV_MLFLOW="$PROJECT_ROOT/.venv/bin/mlflow"

# Check if virtual environment exists
if [ ! -f "$VENV_MLFLOW" ]; then
    echo "Error: MLflow not found in virtual environment"
    echo "Please activate your virtual environment and install mlflow:"
    echo "  source .venv/bin/activate"
    echo "  pip install mlflow"
    exit 1
fi

# Check if MLflow is already running on this port
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "Warning: Port $PORT is already in use"
    echo "To kill existing process:"
    echo "  pkill -f 'mlflow ui'"
    echo "Or use a different port:"
    echo "  $0 5001"
    exit 1
fi

echo "Starting MLflow UI..."
echo "  Project root: $PROJECT_ROOT"
echo "  Tracking URI: file://$MLFLOW_DIR"
echo "  Port: $PORT"
echo ""
echo "MLflow UI will be available at: http://127.0.0.1:$PORT"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start MLflow UI
"$VENV_MLFLOW" ui \
    --backend-store-uri "file://$MLFLOW_DIR" \
    --port "$PORT"
