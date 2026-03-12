#!/usr/bin/env python3
"""
Quick script to run backtest with RL risk model using LIVE EXECUTION THRESHOLDS
This script uses the existing backtest_combined.py with the default thresholds from LiveExecution/src/config.py
"""

import os
import sys
import subprocess


def run_backtest_with_live_thresholds():
    """Run backtest with live execution thresholds (meta=0.7071, qual=0.7, risk=0.1)"""

    # Set project root as working directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(project_root)

    # Add project root to Python path
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print("=" * 80)
    print("Running Backtest with RL Risk Model - LIVE EXECUTION ENVIRONMENT")
    print("=" * 80)
    print("Using thresholds from LiveExecution/src/config.py:")
    print("  Meta threshold: 0.7071")
    print("  Quality threshold: 0.70")
    print("  Risk threshold: 0.10")
    print()

    try:
        # Run the existing combined backtest with live execution thresholds
        command = [
            sys.executable,
            "backtest/backtest_combined.py",
            "--meta",
            "0.7071",
            "--qual",
            "0.70",
            "--risk",
            "0.10",
        ]

        print(f"Executing: {' '.join(command)}")
        print()

        result = subprocess.run(command, check=True, capture_output=True, text=True)

        print("=" * 80)
        print("Backtest Output:")
        print("=" * 80)
        print(result.stdout)

        if result.stderr:
            print("=" * 80)
            print("Error Output:")
            print("=" * 80)
            print(result.stderr)

    except subprocess.CalledProcessError as e:
        print(f"Error running backtest: {e}")
        if e.output:
            print(f"\nOutput:\n{e.output}")
        if e.stderr:
            print(f"\nError:\n{e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback

        print(traceback.format_exc())
        sys.exit(1)


def run_backtest_with_compounding():
    """Run backtest with compounding enabled (more realistic for real trading)"""

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(project_root)

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print("=" * 80)
    print("Running Backtest with RL Risk Model - LIVE EXECUTION + COMPOUNDING")
    print("=" * 80)
    print("Using thresholds from LiveExecution/src/config.py:")
    print("  Meta threshold: 0.7071")
    print("  Quality threshold: 0.70")
    print("  Risk threshold: 0.10")
    print("  Compounding: Enabled")
    print()

    try:
        command = [
            sys.executable,
            "backtest/backtest_combined.py",
            "--meta",
            "0.7071",
            "--qual",
            "0.70",
            "--risk",
            "0.10",
            "--compounding",
        ]

        print(f"Executing: {' '.join(command)}")
        print()

        result = subprocess.run(command, check=True, capture_output=True, text=True)

        print("=" * 80)
        print("Backtest Output:")
        print("=" * 80)
        print(result.stdout)

        if result.stderr:
            print("=" * 80)
            print("Error Output:")
            print("=" * 80)
            print(result.stderr)

    except subprocess.CalledProcessError as e:
        print(f"Error running backtest: {e}")
        if e.output:
            print(f"\nOutput:\n{e.output}")
        if e.stderr:
            print(f"\nError:\n{e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback

        print(traceback.format_exc())
        sys.exit(1)


def show_backtest_results():
    """Show the latest backtest results"""

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    results_dir = os.path.join(project_root, "backtest", "results")

    if not os.path.exists(results_dir):
        print("No backtest results directory found.")
        return

    # Find the latest results file
    import glob
    import os

    result_files = glob.glob(os.path.join(results_dir, "combined_results_*.json"))
    if not result_files:
        print("No backtest results found.")
        return

    # Get the latest file
    latest_file = max(result_files, key=os.path.getctime)

    print(f"Latest backtest results: {os.path.basename(latest_file)}")
    print()

    import json

    try:
        with open(latest_file, "r") as f:
            results = json.load(f)

        print("=" * 80)
        print("Key Backtest Metrics:")
        print("=" * 80)
        print(f"Final Equity: ${results['final_equity']:.2f}")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Number of Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print(f"Profit Factor: {results['profit_factor']:.3f}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Average Trade: ${results['avg_trade']:.2f}")

    except Exception as e:
        print(f"Error reading results file: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run backtest with RL risk model using live execution thresholds"
    )

    parser.add_argument(
        "action",
        choices=["run", "run-compounding", "results"],
        help="Action to perform: run (basic), run-compounding (with compounding), results (show latest results)",
    )

    args = parser.parse_args()

    if args.action == "run":
        run_backtest_with_live_thresholds()
    elif args.action == "run-compounding":
        run_backtest_with_compounding()
    elif args.action == "results":
        show_backtest_results()


if __name__ == "__main__":
    main()
