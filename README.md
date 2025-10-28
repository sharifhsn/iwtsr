# Implied Willow Tree for Swaptions (iwtsr)

This Rust library provides a powerful, non-parametric tool for building an Implied Willow Tree (IWT) calibrated to the European swaption market. It is a faithful adaptation of the model presented by Dong et al. (2024), specifically tailored for the normal (Bachelier) pricing framework used in interest rate derivatives.

The library takes market data for swaptions (volatilities or prices) and constructs a complete risk-neutral process—a discrete tree of forward swap rates and the transition probabilities between them. This calibrated tree can then be used for the pricing and risk management of other interest rate derivatives.

## Key Features

- **Data-Driven:** The model is non-parametric and derives its structure directly from market option prices, rather than assuming a specific stochastic process like Heston or GBM.
- **Swaption-Specific:** The entire framework is built around the **normal (Bachelier) model** and the **forward swap measure**, correctly handling the additive dynamics of interest rates (including negative rates).
- **Weighted & Regularized Calibration:** The calibration process uses a weighted least squares objective function to prioritize fitting more liquid options (i.e., those with tighter bid-ask spreads) and includes a smoothness regularization term to ensure a well-behaved probability distribution, as recommended by the academic literature.
- **High-Level API:** A simple, high-level API allows you to build a complete, calibrated tree from a single function call.
- **Serializable Model:** The calibrated `ImpliedWillowTree` struct can be easily serialized to and deserialized from a file (e.g., JSON), allowing you to store and reuse your models.

## How It Works: The Data Pipeline

The library follows a clear, multi-step process to construct the tree:

1.  **Data Loading:** Parses a CSV file of swaption quotes (maturities, strikes, and bid/mid/ask volatilities).
2.  **Price Conversion:** Converts the market-standard normal volatilities into the currency-denominated payer and receiver swaption prices required by the algorithm.
3.  **Moment Calculation:** Calculates the first four risk-neutral moments of the *absolute change in the forward rate* using the Trapezoidal Rule for accurate numerical integration.
4.  **Node Generation:** Fits a Johnson distribution to the calculated moments and generates a grid of discrete forward swap rate nodes for each time step.
5.  **Calibration:** Solves two constrained non-linear optimization problems to find:
    a.  The **node probabilities (`q`)**, which match the tree's prices to the market prices.
    b.  The **transition probabilities (`p`)**, which define the dynamics between the nodes over time.

## How to Use

### 1. Project Setup

Add the following to your `Cargo.toml`:

```toml
[dependencies]
iwtsr = { git = "https://github.com/your-repo/iwtsr" } # Or path, once published
ndarray = "0.15"
```

### 2. Prepare Your Data

Create a CSV file with your swaption data. The format must be a "long" format where each row represents a single option, with the following columns: `maturity`, `strike`, `annuity`, `bid_vol`, `mid_vol`, `ask_vol`.

**Example: `data/my_quotes.csv`**

```csv
maturity,strike,annuity,bid_vol,mid_vol,ask_vol
0.25,0.025,0.24,0.00405,0.005,0.00595
0.25,0.030,0.24,0.004175,0.005,0.005825
0.50,0.025,0.48,0.00405,0.005,0.00595
0.50,0.030,0.48,0.004175,0.005,0.005825
...
```

### 3. Build and Use the CLI

The primary way to use this library is through its command-line interface.

#### Building the CLI

First, compile the project in release mode for optimal performance:
```sh
cargo build --release
```
The executable will be located at `./target/release/iwtsr`.

#### Step 1: Build and Calibrate a Tree

The `build` command takes your market data and creates a calibrated `ImpliedWillowTree` model, saving it to a file.

**Usage:**
```sh
./target/release/iwtsr build \
    --file <PATH_TO_CSV> \
    --output <PATH_TO_MODEL.json> \
    --f0 <INITIAL_FORWARD_RATE> \
    --nodes <NUM_NODES> \
    --gamma <GAMMA_VALUE> \
    [--alpha <ALPHA_VALUE>]
```

**Arguments Explained:**

*   `--file`: The path to your input CSV file containing the swaption quotes. The annuity factors will be read directly from this file.
*   `--output`: The path where the calibrated model will be saved (e.g., `my_model.json`).
*   `--f0`: The initial forward swap rate for the underlying swap, expressed as a decimal (e.g., `0.03` for 3%). This is the "at-the-money" point around which the model is centered.
*   `--nodes <m>`: The number of discrete nodes (or states) the willow tree will have at each time step. A typical value is between 30 and 100. More nodes can lead to a more accurate fit but will increase computation time.
*   `--gamma`: A parameter for the node-spacing algorithm. Defaults to `0.6`, which is standard in the literature.
*   `--alpha` (Optional): The strength of the smoothness regularization penalty. A higher value will force the resulting probability distribution to be smoother. If not provided, it defaults to `100 * f0`, as recommended in the original paper for empirical data.

#### Step 2: Price Swaptions with a Calibrated Tree

The `price` command loads a saved model and uses it to price European swaptions for a specific maturity and set of strikes.

**Usage:**
```sh
./target/release/iwtsr price \
    --model <PATH_TO_MODEL.json> \
    --strikes <STRIKE_LIST> \
    --maturity <MATURITY> \
    --annuity <ANNUITY_VALUE>
```

**Arguments Explained:**

*   `--model`: The path to a calibrated model file previously saved with the `build` command.
*   `--strikes`: A comma-separated list of strike rates you want to price, expressed as decimals (e.g., `0.025,0.03,0.035`).
*   `--maturity`: The specific swaption maturity (expiry) you want to price, in years (e.g., `0.5` for 6 months). This maturity **must** be one of the maturities the model was originally calibrated with.
*   `--annuity`: The single annuity factor that corresponds to the specific `--maturity` you are pricing. This is used to convert the calculated option value into a price. You can find this value in your original input data file.

### Example Workflow

1.  **Build a model (using default alpha):**
    ```sh
    ./target/release/iwtsr build \
        --file data/synthetic_quotes.csv \
        --output calibrated_model.json \
        --f0 0.03
    ```

2.  **Price a 6-month swaption using the model:**
    ```sh
    ./target/release/iwtsr price \
        --model calibrated_model.json \
        --strikes 0.025,0.03,0.035 \
        --maturity 0.5 \
        --annuity 0.48
    ```

## Future Work & Limitations

- **Pricing Functions:** The library currently focuses on building the tree. Functions for pricing specific derivatives (e.g., American or Asian style swaptions) using the calibrated tree are a natural next step.
- **Input Data:** The model assumes the input data is reasonably clean and free of arbitrage. A production-grade implementation would benefit from a pre-processing step to smooth the input volatility surface.
- **Numerical Stability:** The underlying calibration is a complex numerical optimization problem. For some "difficult" market data sets, the solver may fail to converge. Robust error handling and the ability to tune solver parameters would be valuable additions.
