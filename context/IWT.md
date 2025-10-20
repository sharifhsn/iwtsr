# Implied Willow Tree (IWT) Implementation Guide

This document provides a detailed explanation of the Matlab code for implementing the Implied Willow Tree (IWT) model. The goal is to structure this information for a clear path to re-implementation in other programming languages, such as Rust.

## Core Concepts

The Implied Willow Tree is a numerical method for pricing options and other derivatives. It constructs a discrete tree of future asset prices and associated probabilities, calibrated to match market option prices. This allows for the pricing of exotic options and the calculation of risk sensitivities (Greeks).

The implementation is divided into two main parts:
1.  **Construction of the tree under the risk-neutral measure (Q-measure):** This is used for pricing derivatives.
2.  **Construction of the tree under the physical measure (P-measure):** This is used for risk management and forecasting.

## Data Structures

A Rust implementation would benefit from the following structures:

```rust
struct WillowTree {
    s: Vec<Vec<f64>>, // Asset prices at each node (m x N)
    p: Vec<Vec<Vec<f64>>>, // Transition probabilities (m x m x N-1)
    q: Vec<Vec<f64>>, // Node probabilities (m x N)
}

struct MarketData {
    call_prices: Vec<Vec<f64>>, // (num_strikes x num_maturities)
    put_prices: Vec<Vec<f64>>,  // (num_strikes x num_maturities)
    strikes: Vec<f64>,
    maturities: Vec<f64>,
    spot_price: f64,
    discount_factors: Vec<f64>,
}

struct Moments {
    mean: Vec<f64>,
    variance: Vec<f64>,
    skewness: Vec<f64>,
    kurtosis: Vec<f64>,
}
```

## Implementation Steps

### 1. Moment Calculation (`Imp_Moments_underQ` and `Imp_Moments_underP`)

The first step is to calculate the first four moments of the log-returns of the underlying asset. These moments are implied from the market prices of European call and put options.

-   **Inputs:** Market option prices, strikes, spot price, discount factors.
-   **Process:** The code iterates through each maturity, calculating the moments using the formulas from Bakshi, Kapadia, and Madan (2003). The functions `Imp_Moments_underQ_oneMaturity` and `Imp_Moments_underP_oneMaturity` contain the core logic for a single maturity.
-   **Outputs:** A `Moments` structure containing the mean, variance, skewness, and kurtosis for each maturity.

### 2. Willow Tree Node Generation (`WTnodes_from_JohnsonCurve`)

The nodes of the willow tree, representing discrete asset prices at each time step, are generated from the calculated moments.

-   **Inputs:** `Moments`, number of time steps `N`, number of nodes per step `m`, and a parameter `gamma`.
-   **Process:**
    1.  A set of standard normal quantiles `z` is generated based on the `gamma` parameter.
    2.  For each time step, the `f_hhh` function (a compiled C++ function) is called to determine the parameters of the Johnson distribution that matches the moments for that time step.
    3.  The Johnson curve inverse transformation is applied to the quantiles `z` to generate the log-returns.
    4.  These log-returns are then used to calculate the asset prices `S` at each node.
-   **Outputs:** The asset price tree `S`.

### 3. Probability Calculation (`ImpWT_given_moments_underQ` and `ImpWT_given_moments_underP`)

This is the core of the calibration process, where the node and transition probabilities are determined.

#### a. Node Probabilities (`q`)

-   **Process:** For each time step, an optimization problem is solved to find the node probabilities `q` that minimize the mean squared error between the model-implied option prices and the market option prices.
-   **Objective Function (`obj_q`):** The objective function calculates the sum of squared errors for call and put prices across all strikes for a given maturity.
-   **Constraints:**
    -   The probabilities must sum to 1.
    -   Under the Q-measure, there is an additional martingale constraint: the discounted expected future asset price must equal the current asset price.
-   **Optimization:** The Matlab code uses `fmincon` to solve this constrained optimization problem. A Rust implementation would require a similar numerical optimization library (e.g., `argmin` or `nlopt`).

#### b. Transition Probabilities (`p`)

-   **Process:** After calculating the node probabilities `q`, the transition probabilities `p` between nodes at consecutive time steps are determined. This is also an optimization problem.
-   **Objective Function (`obj_p_givenq`):** The objective function minimizes the squared difference between the propagated node probabilities from time `t` to `t+1` and the calculated node probabilities at `t+1`.
-   **Constraints:**
    -   For each starting node, the transition probabilities to all nodes at the next time step must sum to 1.
    -   Under the Q-measure, there is a martingale constraint on the expected asset price.
-   **Optimization:** Again, `fmincon` is used in the Matlab code.

### 4. Construction under Q and P Measures

-   `Construct_ImpWT_underQ`: This function orchestrates the entire process for the Q-measure.
-   `Construct_ImpWT_underP`: This function first calls `trans_optionprice_fromQ_toP` to convert the Q-measure option prices to P-measure prices using a CRRA utility function. Then, it follows the same steps as the Q-measure construction, but with the P-measure moments and prices, and without the martingale constraint.

### 5. Option Pricing and Greeks

Once the willow tree is constructed, it can be used for pricing.

-   **European Options (`Price_EuroOption_WT`):** Priced by backward induction through the tree. The value at each node is the discounted expected value of the option at the next time step.
-   **American Options (`Price_AmerOption_WT`):** Also priced by backward induction, but at each node, the value is the maximum of the continuation value (as in the European case) and the early exercise value.
-   **Asian Options (`Price_AsianCall_WT`):** More complex, requiring an expanded state space to keep track of the average asset price. The implementation uses linear interpolation to manage the state space.
-   **Greeks (`Greeks_EuroOption`, `Greeks`):** Calculated using finite differences on the willow tree. The `Greeks` function uses a Taylor expansion around the initial stock price to estimate delta, gamma, and theta.

## Key Functions for Re-implementation

-   **`Imp_Moments_underQ` / `Imp_Moments_underP`:** The core logic for moment calculation.
-   **`WTnodes_from_JohnsonCurve`:** The node generation logic. The `f_hhh` C++ function would need to be re-implemented or a similar library found.
-   **`obj_q` and `obj_p_givenq`:** The objective functions for the optimization problems.
-   **`Price_*_WT` functions:** The backward induction logic for pricing different option types.

A successful Rust implementation will heavily rely on a robust numerical optimization library to replace Matlab's `fmincon`. The rest of the code is primarily matrix and vector operations, which can be efficiently handled by libraries like `ndarray`.
