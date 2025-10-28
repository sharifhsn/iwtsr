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
DATA_FILE = "data/synthetic_bachelier_realistic.csv"
MODEL_FILE = "model.json"
F0 = 0.035

# --- Hyperparameters to Test ---
GAMMA_VALUES = [0.4, 0.6, 0.8]
ALPHA_VALUES = [0.0, 1.0, 3.5, 10.0]


def calculate_bachelier_price(f0, strike, vol, expiry, annuity, is_call):
    """Calculates the Bachelier price for a single option."""
    if vol * expiry**0.5 < 1e-9:
        return annuity * (max(0, f0 - strike) if is_call else max(0, strike - f0))

    d = (f0 - strike) / (vol * expiry**0.5)
    price = annuity * (
        (f0 - strike) * norm.cdf(d) + vol * expiry**0.5 * norm.pdf(d)
    )
    if not is_call:
        price -= annuity * (f0 - strike)
    return price

def run_command(command):
    """Runs a shell command and returns its output."""
    print(f"Running: {' '.join(command)}")
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(f"Stderr: {e.stderr}")
        raise

def generate_comparison_plot(gamma, alpha):
    """
    Builds a model for a given gamma and alpha, then generates a
    combined comparison plot.
    """
    # 1. Build the model with the specified hyperparameters
    print(f"\n--- Building model for gamma={gamma}, alpha={alpha} ---")
    build_command = [
        IWTS_EXECUTABLE, "build",
        "--file", DATA_FILE,
        "--output", MODEL_FILE,
        "--f0", str(F0),
        "--gamma", str(gamma),
        "--alpha", str(alpha),
    ]
    run_command(build_command)
    print("Model built successfully.")

    # 2. Load data and set up plot
    df = pd.read_csv(DATA_FILE)
    maturities = sorted(df["maturity"].unique())
    n_maturities = len(maturities)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(n_maturities, 2, figsize=(16, 6 * n_maturities), sharex=True)
    fig.suptitle(f"Model vs. Synthetic (gamma={gamma}, alpha={alpha})", fontsize=20)

    # 3. Process each maturity
    for i, maturity in enumerate(maturities):
        print(f"--- Processing maturity: {maturity} ---")
        maturity_df = df[df["maturity"] == maturity].sort_values("strike").reset_index()
        strikes = maturity_df["strike"].tolist()
        annuity = maturity_df["annuity"].iloc[0]
        mid_vols = maturity_df["mid_vol"].tolist()

        # Get model prices by calling the executable for each strike
        model_payer_prices = []
        model_receiver_prices = []
        for strike in strikes:
            price_command = [
                IWTS_EXECUTABLE, "price",
                "--model", MODEL_FILE,
                "--maturity", str(maturity),
                "--annuity", str(annuity),
                "--strike", str(strike),
                "--f0", str(F0),
            ]
            model_output = run_command(price_command)

            # Parse output
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

        # Calculate synthetic prices
        synthetic_payer = [calculate_bachelier_price(F0, k, v, maturity, annuity, True) for k, v in zip(strikes, mid_vols)]
        synthetic_receiver = [calculate_bachelier_price(F0, k, v, maturity, annuity, False) for k, v in zip(strikes, mid_vols)]

        # Plot
        ax_payer, ax_receiver = axes[i, 0], axes[i, 1]
        ax_payer.plot(strikes, synthetic_payer, 'o-', label="Synthetic Payer", color='blue', markersize=4)
        ax_payer.plot(model_prices_df["Strike"], model_prices_df["Payer Price"], '--', label="Model Payer", color='cyan')
        ax_payer.set_title(f"Payer Prices (Maturity: {maturity}y)", fontsize=14)
        ax_payer.set_ylabel("Price", fontsize=12)
        ax_payer.legend()

        ax_receiver.plot(strikes, synthetic_receiver, 'o-', label="Synthetic Receiver", color='red', markersize=4)
        ax_receiver.plot(model_prices_df["Strike"], model_prices_df["Receiver Price"], '--', label="Model Receiver", color='magenta')
        ax_receiver.set_title(f"Receiver Prices (Maturity: {maturity}y)", fontsize=14)
        ax_receiver.legend()

    fig.text(0.5, 0.04, 'Strike', ha='center', va='center', fontsize=14)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    # 4. Save the final plot
    output_filename = f"comparison_gamma_{gamma}_alpha_{alpha}.png"
    plt.savefig(output_filename)
    print(f"\nCombined plot saved to {output_filename}")
    plt.close(fig)

def main():
    """Main loop to iterate through hyperparameters."""
    for gamma in GAMMA_VALUES:
        for alpha in ALPHA_VALUES:
            generate_comparison_plot(gamma, alpha)

if __name__ == "__main__":
    main()
