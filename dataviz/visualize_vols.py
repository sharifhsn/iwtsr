# /// script
# dependencies = [
#   "matplotlib",
#   "pandas",
#   "scipy",
# ]
# ///

import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import io
import os

# --- Configuration ---
IWTS_EXECUTABLE = "./target/release/iwtsr"
DATA_FILE = "data/synthetic_quotes.csv"
MODEL_FILE = "model.json"
F0 = 0.035
OUTPUT_FILENAME = "volatility_comparison.png"

def run_command(command):
    """Runs a shell command and returns its output."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(f"Stderr: {e.stderr}")
        raise

def main():
    """Main function to build model, get implied vols, and generate plots."""
    # 1. Build the model
    print("--- Building the Implied Willow Tree model ---")
    build_command = [
        IWTS_EXECUTABLE,
        "build",
        "--file", DATA_FILE,
        "--output", MODEL_FILE,
        "--f0", str(F0),
        "--alpha", "0.0",
    ]
    run_command(build_command)
    print("Model built successfully.")

    # 2. Load and process the synthetic data
    print("\n--- Loading synthetic data ---")
    df = pd.read_csv(DATA_FILE)
    maturities = sorted(df["maturity"].unique())
    n_maturities = len(maturities)

    # 3. Create a single figure with multiple subplots
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(n_maturities, 1, figsize=(12, 8 * n_maturities), squeeze=False)
    fig.suptitle("Model vs. Synthetic Implied Volatility Comparison", fontsize=20)

    # 4. Generate a plot for each maturity
    for i, maturity in enumerate(maturities):
        print(f"\n--- Processing maturity: {maturity} ---")
        maturity_df = df[df["maturity"] == maturity].sort_values("strike").reset_index()

        strikes = maturity_df["strike"].tolist()
        annuity = maturity_df["annuity"].iloc[0]
        synthetic_mid_vols = maturity_df["mid_vol"]

        # a. Get model-implied volatilities for each strike
        model_payer_vols = []
        model_receiver_vols = []
        for strike in strikes:
            price_command = [
                IWTS_EXECUTABLE,
                "price",
                "--model", MODEL_FILE,
                "--maturity", str(maturity),
                "--annuity", str(annuity),
                "--strike", str(strike),
                "--output-vols",
                "--f0", str(F0),
            ]
            model_output = run_command(price_command)

            # Parse the single line of volatility output
            lines = model_output.strip().split('\n')
            data_line = [line for line in lines if '|' in line and 'Payer' not in line and '---' not in line][0]
            parts = [p.strip() for p in data_line.split('|') if p.strip()]
            if len(parts) == 2:
                model_payer_vols.append(float(parts[0]))
                model_receiver_vols.append(float(parts[1]))

        # c. Plot the results on the appropriate subplot
        ax = axes[i, 0]

        ax.plot(strikes, synthetic_mid_vols, 'o-', label="Synthetic Mid Volatility", color='green', markersize=5)
        ax.plot(strikes, model_payer_vols, '--', label="Model Payer Volatility", color='cyan')
        ax.plot(strikes, model_receiver_vols, ':', label="Model Receiver Volatility", color='magenta')

        ax.set_title(f"Volatility Smile (Maturity: {maturity}y)", fontsize=14)
        ax.set_ylabel("Normal Volatility", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True)

    # Add a shared x-axis label
    fig.text(0.5, 0.04, 'Strike', ha='center', va='center', fontsize=14)
    
    # 5. Save the combined plot
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(OUTPUT_FILENAME)
    print(f"\nCombined volatility plot saved to {OUTPUT_FILENAME}")
    plt.close(fig)

if __name__ == "__main__":
    main()
