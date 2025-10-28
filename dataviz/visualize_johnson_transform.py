# /// script
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "scipy",
# ]
# ///
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
IWTS_EXECUTABLE = "./target/release/iwtsr"
DATA_FILE = "data/synthetic_bachelier_realistic.csv"
F0 = 0.03
MATURITY_IDX = 0
NODES = 50
GAMMA = 0.6
OUTPUT_FILENAME = "assets/johnson_transform_comparison.png"

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

def main():
    """
    Runs the johnson CLI command and generates a plot visualizing the
    transformation from standard normal quantiles to the final node values.
    """
    # 1. Call the new 'johnson' command
    print("--- Generating Johnson transformation data ---")
    command = [
        IWTS_EXECUTABLE, "johnson",
        "--file", DATA_FILE,
        "--f0", str(F0),
        "--maturity-idx", str(MATURITY_IDX),
        "--nodes", str(NODES),
        "--gamma", str(GAMMA),
    ]
    output = run_command(command)
    data = json.loads(output)

    z = np.array(data["initial_quantiles"])
    willow_nodes = np.array(data["transformed_nodes"])

    # 2. Create the plot
    print("--- Creating visualization ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # Determine common bins for both histograms
    combined_data = np.concatenate((z, willow_nodes))
    bins = np.linspace(np.min(combined_data), np.max(combined_data), 30)

    # Plot histogram for the initial standard normal quantiles
    ax.hist(z, bins=bins, density=True, color='gray', alpha=0.6, label='Standard Normal Quantiles (z)')

    # Plot histogram for the transformed (Johnson) nodes
    ax.hist(willow_nodes, bins=bins, density=True, color='royalblue', alpha=0.6, label='Transformed Nodes (RND)')

    # Overlay theoretical PDFs for reference
    from scipy.stats import norm
    x = np.linspace(np.min(combined_data), np.max(combined_data), 200)
    ax.plot(x, norm.pdf(x, 0, 1), color='black', linestyle='--', label='Standard Normal PDF')


    ax.set_title(
        f"Distribution Comparison (Maturity Index: {MATURITY_IDX})\n"
        f"({NODES} nodes, gamma={GAMMA})",
        fontsize=16
    )
    ax.set_xlabel("Node Value", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 3. Save the plot
    plt.tight_layout()
    plt.savefig(OUTPUT_FILENAME, dpi=300)
    print(f"Plot saved to {OUTPUT_FILENAME}")
    plt.close(fig)


if __name__ == "__main__":
    main()
