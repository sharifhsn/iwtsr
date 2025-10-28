# /// script
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "scipy",
# ]
# ///
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import re

def johnson_pdf(x, params_str):
    """Calculates the PDF of a Johnson distribution."""
    
    # Parse the params_str
    if "Su" in params_str:
        itype = 2
        match = re.search(r"gamma:\s*(-?[\d\.]+(?:e-?\d+)?),\s*delta:\s*(-?[\d\.]+(?:e-?\d+)?),\s*lambda:\s*(-?[\d\.]+(?:e-?\d+)?),\s*xi:\s*(-?[\d\.]+(?:e-?\d+)?)", params_str)
        gamma, delta, lambda_p, xi = [float(v) for v in match.groups()]
    elif "Sb" in params_str:
        itype = 3
        match = re.search(r"gamma:\s*(-?[\d\.]+(?:e-?\d+)?),\s*delta:\s*(-?[\d\.]+(?:e-?\d+)?),\s*lambda:\s*(-?[\d\.]+(?:e-?\d+)?),\s*xi:\s*(-?[\d\.]+(?:e-?\d+)?)", params_str)
        gamma, delta, lambda_p, xi = [float(v) for v in match.groups()]
    else:
        raise ValueError(f"Unsupported Johnson distribution type in string: {params_str}")

    # This logic is ported from the MATLAB f_john_dens.m
    u = (x - xi) / lambda_p
    
    if itype == 2: # SU
        g_prime = delta / (lambda_p * np.sqrt(u**2 + 1))
        pdf_val = g_prime * norm.pdf(gamma + delta * np.arcsinh(u))
    elif itype == 3: # SB
        g_prime = delta / (lambda_p * u * (1 - u))
        pdf_val = g_prime * norm.pdf(gamma + delta * np.log(u / (1 - u)))
    else:
        return np.zeros_like(x)
        
    return pdf_val

def main():
    maturity_idx = 0 # Use the first maturity (0-indexed)

    # 1. Run the Rust CLI to get moments and parameters
    result = subprocess.run(
        [
            "./target/release/iwtsr",
            "moments",
            "--file", "data/synthetic_bachelier_realistic.csv",
            "--f0", "0.03",
            "--maturity-idx", str(maturity_idx),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Failed to run iwtsr moments command:")
        print(result.stderr)
        return

    data = json.loads(result.stdout)
    skew = data["skewness"]
    kurt = data["kurtosis"]
    params_str = data["johnson_params"]

    # 2. Set up the plot
    std_dev = np.sqrt(data["variance"])
    x = np.linspace(-5 * std_dev, 5 * std_dev, 500)
    
    # 3. Calculate PDFs
    johnson_y = johnson_pdf(x, params_str)
    normal_y = norm.pdf(x, loc=0, scale=std_dev) # Centered at 0 for absolute change

    # 4. Plot the results
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(x, normal_y, label=f'Normal PDF (Std Dev: {std_dev:.4f})', color='royalblue', linestyle='--')
    ax.plot(x, johnson_y, label='Fitted Johnson PDF', color='firebrick', linewidth=2)
    
    ax.fill_between(x, johnson_y, color='firebrick', alpha=0.1)
    ax.fill_between(x, normal_y, color='royalblue', alpha=0.1)

    ax.set_title(
        f"Johnson vs. Normal Distribution (Maturity Index {maturity_idx})\n"
        f"Skewness: {skew:.4f} | Kurtosis: {kurt:.4f}",
        fontsize=16
    )
    ax.set_xlabel("Absolute Change in Forward Rate (Y_T)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    output_path = "assets/johnson_fit_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
