
# Backtest Viewer logic

A simple web application to visualize backtest results and trade entries/exits on an interactive chart.

## How to use

1.  **Server is running**: The server should be running in the background. Open your browser to:
    [http://127.0.0.1:8000](http://127.0.0.1:8000)

2.  **Select Data**:
    *   **Backtest Result**: Choose the CSV file from `backtest/results/` (e.g., `trades_stage1_...csv`).
    *   **Asset**: Choose the asset pair (e.g., `EURUSD`).
    *   Click **Load Chart**.

3.  **Interact**:
    *   **Zoom/Pan**: Use mouse wheel and click-drag.
    *   **Trades**: Buy entries trade are marked with **Green Up Arrows**, Sell entries with **Red Down Arrows**.
    *   **Details**: Hover over markers or the chart to see details in the tooltip and left sidebar.

## Project Structure

*   `backtest_viewer/app.py`: FastAPI backend. Serves files and provides API for parquet/csv data.
*   `backtest_viewer/templates/index.html`: Main UI.
*   `backtest_viewer/static/js/main.js`: Charting logic using TradingView Lightweight Charts.

## Troubleshooting

If the server is not running, start it manually:
```powershell
python -m uvicorn backtest_viewer.app:app --reload --port 8000
```
