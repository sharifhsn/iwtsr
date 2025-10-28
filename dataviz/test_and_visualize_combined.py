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
OUTPUT_FILENAME = "comparison_all_maturities.png"

def calculate_bachelier_price(f0, strike, vol, expiry, annuity, is_call):
    """Calculates the Bachelier price for a single option."""
    if vol * expiry**0.5 < 1e-9:
        return annuity * (max(0, f0 - strike) if is_call else max(0, strike - f0))

    d = (f0 - strike) / (vol * expiry**0.5)
    price = annuity * (
        (f0 - strike) * norm.cdf(d) + vol * expiry**0.5 * norm.pdf(d)
    )
    if not is_call:
        # Use put-call parity for puts
        price -= annuity * (f0 - strike)
    return price

def run_command(command):
    """Runs a shell command and returns its output."""
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return result.stdout

def main():
    """Main function to build model, price, and generate plots."""
    # 1. Build the model
    print("--- Building the Implied Willow Tree model ---")
    build_command = [
        IWTS_EXECUTABLE,
        "build",
        "--file", DATA_FILE,
        "--output", MODEL_FILE,
        "--f0", str(F0),
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
    fig, axes = plt.subplots(
        n_maturities, 2, 
        figsize=(16, 6 * n_maturities), 
        sharex=True
    )
    fig.suptitle("Model vs. Synthetic Price Comparison", fontsize=20)


    # 4. Generate a plot for each maturity
    for i, maturity in enumerate(maturities):
        print(f"\n--- Processing maturity: {maturity} ---")
        maturity_df = df[df["maturity"] == maturity].sort_values("strike").reset_index()

        strikes = maturity_df["strike"].tolist()
        annuity = maturity_df["annuity"].iloc[0]
        mid_vols = maturity_df["mid_vol"].tolist()

        # a. Get model prices by calling the executable for each strike
        model_payer_prices = []
        model_receiver_prices = []
        for strike in strikes:
            price_command = [
                IWTS_EXECUTABLE,
                "price",
                "--model", MODEL_FILE,
                "--maturity", str(maturity),
                "--annuity", str(annuity),
                "--strike", str(strike),
            ]
            model_output = run_command(price_command)

            # Parse the single line of price output
            lines = model_output.strip().split('\n')
            data_line = [line for line in lines if '|' in line and 'Payer' not in line and '---' not in line][0]
            parts = [p.strip() for p in data_line.split('|') if p.strip()]
            if len(parts) == 2:
                model_payer_prices.append(float(parts[0]))
                model_receiver_prices.append(float(parts[1]))

        model_prices_df = pd.DataFrame({
            "Strike": strikes,
            "Payer Price": model_payer_prices,
            "Receiver Price": model_receiver_prices
        })

        # b. Calculate synthetic prices
        synthetic_payer = [
            calculate_bachelier_price(F0, k, v, maturity, annuity, is_call=True)
            for k, v in zip(strikes, mid_vols)
        ]
        synthetic_receiver = [
            calculate_bachelier_price(F0, k, v, maturity, annuity, is_call=False)
            for k, v in zip(strikes, mid_vols)
        ]

        # c. Plot the results on the appropriate subplots
        ax_payer = axes[i, 0]
        ax_receiver = axes[i, 1]

        # Payer (Call) Plot
        ax_payer.plot(strikes, synthetic_payer, 'o-', label="Synthetic Payer", color='blue', markersize=4)
        ax_payer.plot(model_prices_df["Strike"], model_prices_df["Payer Price"], '--', label="Model Payer", color='cyan')
        ax_payer.set_title(f"Payer Prices (Maturity: {maturity}y)", fontsize=14)
        ax_payer.set_ylabel("Price", fontsize=12)
        ax_payer.legend(fontsize=10)
        ax_payer.grid(True)

        # Receiver (Put) Plot
        ax_receiver.plot(strikes, synthetic_receiver, 'o-', label="Synthetic Receiver", color='red', markersize=4)
        ax_receiver.plot(model_prices_df["Strike"], model_prices_df["Receiver Price"], '--', label="Model Receiver", color='magenta')
        ax_receiver.set_title(f"Receiver Prices (Maturity: {maturity}y)", fontsize=14)
        ax_receiver.legend(fontsize=10)
        ax_receiver.grid(True)

    # Add a shared x-axis label
    fig.text(0.5, 0.04, 'Strike', ha='center', va='center', fontsize=14)
    
    # 5. Save the combined plot
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(OUTPUT_FILENAME)
    print(f"\nCombined plot saved to {OUTPUT_FILENAME}")
    plt.close(fig)

if __name__ == "__main__":
    main()
