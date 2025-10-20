# A Detailed Walkthrough of the Implied Willow Tree Implementation

This document provides an in-depth, natural language explanation of the Matlab code in the `/ImpliedWT` directory. We will trace the execution flow starting from a practical example, `demo_construct_ImpWT_underQ.m`, to understand how the risk-neutral (Q-measure) implied willow tree is constructed. We will then extend this analysis to the physical (P-measure) construction using `demo_construct_ImpWT_underP.m`.

## Part 1: Constructing the Tree under the Q-Measure

Our journey begins with `demo_construct_ImpWT_underQ.m`. This script first sets up the market parameters and then calls the main construction function.

### Step 0: The Setup in `demo_construct_ImpWT_underQ.m`

Before we enter the `/ImpliedWT` directory, the demo script defines the universe for our analysis:

1.  **Market and Option Parameters are defined:**
    *   `r = 0.05;`: A 5% risk-free interest rate.
    *   `sigma = 0.3;`: A 30% volatility. This is only used to *generate* synthetic option prices for the demo. In a real-world scenario, you would use actual market prices.
    *   `S0 = 1;`: The initial price of the underlying asset is $1.
    *   `T = 3/12;`: The total time to maturity is 3 months.
    *   `N = 3;`: The tree will have 3 time steps.
    *   `dt = T/N;`: The duration of each time step is 1 month.
    *   `time_nodes`: A vector of maturities `[1/12, 2/12, 3/12]`.
    *   `B0 = exp(-r*time_nodes);`: A vector of discount factors corresponding to each maturity.
    *   `K`: A vector of strike prices, e.g., `[0.5, 0.6, ..., 1.5]`.
    *   `dK`: The constant difference between strikes, which is 0.1.

2.  **Synthetic Market Data is Generated:**
    *   The script calls `BSM_OptionPrice_addNoise` to create `call_mkt` and `put_mkt` matrices. These matrices hold the prices of European call and put options for each strike (`K`) and each maturity (`time_nodes`). For this demo, the prices are calculated using the Black-Scholes model.

3.  **The Main Function Call:**
    *   `m = 20;`: We specify that the willow tree should have 20 discrete price nodes at each time step.
    *   The script then makes the primary call that initiates the entire process:
        ```matlab
        [S, p, q] = Construct_ImpWT_underQ(call_mkt, put_mkt, S0, B0, K, dK, N, m);
        ```
    *   This call takes us into the `/ImpliedWT` directory, starting with `Construct_ImpWT_underQ.m`.

### Step 1: `Construct_ImpWT_underQ.m` - The Orchestrator

This function is the main entry point for building the Q-measure tree. It doesn't perform the core calculations itself but orchestrates the sequence of operations.

1.  **Function Signature:** It takes all the market data and tree parameters as input and aims to return the three key components of the willow tree:
    *   `S`: An `m x N` matrix of asset prices. `S(i, j)` is the price at node `i` at time step `j`.
    *   `p`: An `m x m x (N-1)` tensor of transition probabilities. `p(i, j, k)` is the probability of moving from node `i` at time `k` to node `j` at time `k+1`.
    *   `q`: An `m x N` matrix of node probabilities. `q(i, j)` is the probability of being at node `i` at time `j`.

2.  **First Operation: Moment Calculation**
    *   The first thing it does is calculate the risk-neutral moments implied by the market option prices:
        ```matlab
        [mean_dd, var_dd, skew_dd, kurt_dd] = Imp_Moments_underQ(call_mkt, put_mkt, S0, B0, K, dK, N);
        ```
    *   This call leads us to the `Imp_Moments_underQ.m` file.

### Step 2: `Imp_Moments_underQ.m` - Deriving the Moments

This function's sole purpose is to extract the first four moments (mean, variance, skewness, kurtosis) of the risk-neutral log-return distribution for each maturity.

1.  **Looping Through Maturities:** The function iterates from `n_t = 1` to `N` (i.e., for each of the 3 maturities).
2.  **Calling the "One Maturity" Worker:** Inside the loop, it calls a helper sub-function, `Imp_Moments_underQ_oneMaturity`, which does the actual calculation for a single slice of time. It passes the call and put prices for that specific maturity.
3.  **Inside `Imp_Moments_underQ_oneMaturity`:**
    *   The code implements the formulas from Bakshi, Kapadia, and Madan (2003) to find moments from a set of option prices.
    *   It first creates a vector of `out_of_money` option prices. For strikes below the current price `S0`, it uses put prices; for strikes above, it uses call prices.
    *   It then calculates `k1`, `k2`, `k3`, and `k4`, which are the raw moments. These calculations are essentially numerical integrations (approximated as sums) over the strike prices, weighted by the option prices.
    *   Finally, it converts these raw moments into the standard central moments: `mean_dd`, `var_dd`, `skew_dd`, and `kurt_dd`.
4.  **Return Value:** The function returns four vectors, one for each moment, where each element corresponds to a maturity.

### Step 3: Back in `Construct_ImpWT_underQ.m`, Calling the Tree Builder

With the moments calculated, the orchestrator proceeds to the main event: building the tree.

1.  `gamma = 0.6;`: A parameter for the node-spacing algorithm is defined.
2.  The function calls `ImpWT_given_moments_underQ`, passing in all the market data *and* the newly calculated moments.
    ```matlab
    [S, p, q] = ImpWT_given_moments_underQ(S0, B0, K, call_mkt, put_mkt, m, N, gamma, mean_dd, var_dd, skew_dd, kurt_dd);
    ```

### Step 4: `ImpWT_given_moments_underQ.m` - The Core Construction

This is where the willow tree is actually built and calibrated. The process has three distinct phases.

#### Phase 4.1: Generating the Asset Price Nodes (`S`)

1.  The four moment vectors are combined into a single `4 x N` matrix `G`.
2.  A call is made to `WTnodes_from_JohnsonCurve(G,N,m,gamma)`. This utility function (from the `/Utils` directory) is a crucial black box. It takes the moments for each time step and uses the Johnson curve system to generate a set of `m` log-return values for each of the `N` time steps that match these moments.
3.  The output from the Johnson curve function is a matrix of log-returns. This is converted into a matrix of asset prices `S` by the formula `S = S0 * exp(log_returns)`.
4.  The prices at each time step are sorted. The result is the `m x N` matrix `S`, which defines the structure of our tree.

#### A Deep Dive into the `/Utils` Directory

This is the most computationally complex part of the node generation. Let's break down the components.

**Part A: The Matlab Script (`WTnodes_from_JohnsonCurve.m`)**

The script's job is to prepare inputs for the C function and process its outputs.

1.  **Compute `z` and `q`:**
    *   It first creates a vector of probabilities `q` of size `M`. The formula `(k-0.5)^gamma/M` creates a set of non-uniform probabilities that are symmetric around the midpoint. This is a heuristic to define the quantiles for the distribution.
    *   It then normalizes `q` so that it sums to 1.
    *   Using these probabilities, it computes a vector of `M` quantiles `z` from the standard normal distribution using `norminv`. `z` now represents a set of points from a standard normal distribution that we want to map to our target distribution.

2.  **Loop and Call C Code:**
    *   The script loops `N` times (once for each time step).
    *   Inside the loop, it extracts the moments for the current time step: `mu` (mean), `sd` (standard deviation), `ka3` (skewness), and `ka4` (kurtosis).
    *   It then calls the compiled C function: `[a,b,d,c,itype(i),ifault(i)] = f_hhh(mu,sd,ka3,ka4);`. This is the core of the process. The script is asking the C code: "Find me the parameters of a Johnson distribution that has these four moments."

**Part B: The C Code (`f_hhh.c`) - The Hill, Hill, and Holder Algorithm**

This C code is compiled into a MEX file, which is a special type of shared library that Matlab can call directly.

1.  **The Gateway: `mexFunction`**
    *   This function is the entry point for Matlab. Its only job is to act as an interpreter.
    *   It receives the inputs from Matlab (`prhs`, or "right-hand side") which are `mxArray` structures. It extracts the scalar double values for `xbar` (mean), `sd`, `rb1` (skew), and `bb2` (kurtosis).
    *   It creates `mxArray` structures for the outputs (`plhs`, or "left-hand side").
    *   It calls the main computational routine, `hhh`, passing pointers to the input and output variables.
    *   Finally, it assigns the results from the C variables back into the Matlab output structures.

2.  **The Dispatcher: `hhh` function**
    *   This function implements the main logic of the Hill, Hill, and Holder (1976) algorithm. Its purpose is to determine which family of distributions is appropriate based on the input skewness and kurtosis, and then to call the correct fitting routine.
    *   It first checks for invalid input (e.g., negative standard deviation).
    *   It checks for the trivial case of a constant value (zero standard deviation).
    *   It then checks the relationship between skewness (`b1 = rb1*rb1`) and kurtosis (`b2`). The (B1, B2) plane is divided into regions corresponding to different distribution families.
        *   If `b2` is very close to `b1 + 1`, it's a degenerate case (ST distribution).
        *   If `b1` and `b2-3` are both near zero, it's a **Normal Distribution (`itype=4`)**. The parameters are calculated directly.
        *   It calculates the "lognormal line" in the (B1, B2) plane. If the moments fall on this line, it's a **Lognormal Distribution (`itype=1`)**. The parameters are calculated directly.
        *   If the moments fall below the lognormal line, it's a **Johnson SU Distribution (`itype=2`)**. It calls the `sufit` function.
        *   If the moments fall above the lognormal line, it's a **Johnson SB Distribution (`itype=3`)**. It calls the `sbfit` function.
    *   The function returns the distribution type (`itype`) and the four Johnson parameters (`gamma`, `delta`, `xi`, `xlam`), which are renamed to `a, b, c, d` back in the Matlab script.

3.  **The Fitters: `sufit` and `sbfit` functions**
    *   These are the iterative workhorses. They take the moments and find the parameters of the SU or SB distribution that match them.
    *   `sufit` (for unbounded distributions) uses a direct, but complex, iterative formula to solve for the parameters.
    *   `sbfit` (for bounded distributions) is more complex. It uses a Newton-Raphson-like iterative method to find the parameters `g` and `d`. Inside its main loop (`g80`), it calls another function, `mom`, to get the moments for its current guess of the parameters. It then calculates derivatives and updates its guess, repeating until it converges.

**Part C: Back in Matlab - The Final Transformation**

1.  Once `f_hhh` returns, the Matlab script has the Johnson parameters (`a`, `b`, `c`, `d`) and the distribution type (`itype`).
2.  It uses these parameters to transform the standard normal quantiles `z` into the log-returns `x` for the current time step. This is the "Johnson curve inverse transformation".
    *   `u = (z - a) / b`
    *   The transformation of `u` depends on `itype`:
        *   `itype=1` (Lognormal): `gi = exp(u)`
        *   `itype=2` (SU): `gi = sinh(u)`
        *   `itype=3` (SB): `gi = 1 / (1 + exp(-u))` (Logistic function)
        *   `itype=4` (Normal): `gi = u`
    *   `x = c + d * gi`
3.  The resulting vector `x` contains the `M` log-return values for the current time step that collectively match the target moments. This vector becomes one column of the `Willow` matrix.
4.  After the loop finishes, the `Willow` matrix (which is a matrix of log-returns) is returned.

**Part D: The Matlab Utility Functions (`f_john_dens.m`, `f_john_mom.m`)**

These files provide a pure-Matlab implementation for working with Johnson distributions. While they are not used in the final tree-building process (which uses the faster C code), they are important for understanding the underlying mathematics.

1.  **`f_john_dens.m` - Johnson PDF**
    *   **Purpose:** To calculate the Probability Density Function (PDF) for a value `x`, given a set of Johnson parameters.
    *   **Process:** It directly implements the mathematical formula for the PDF of a Johnson distribution. This involves two helper sub-functions:
        *   `g = fg(u, type)`: This computes the core transformation that maps the variable `x` to a standard normal equivalent. `u` is the standardized variable `(x-c)/d`. The function `g` is the inverse of the transformation seen in `WTnodes_from_JohnsonCurve` (e.g., `log(u)` for Lognormal, `asinh(u)` for SU).
        *   `gp = fgp(u, type)`: This computes `g'`, the derivative of the transformation function `g`. This is required by the change of variables formula for PDFs.
    *   The final value `fv` is the PDF value.

2.  **`f_john_mom.m` - Johnson Moment Calculator**
    *   **Purpose:** To calculate the first four moments of a Johnson distribution given its parameters.
    *   **Note:** This function is **not** called by the main workflow. It represents a pure-Matlab alternative to the fitting logic inside `f_hhh.c`. The C code's internal `mom` function is used during its fitting process, not this M-file.
    *   **Process:**
        *   It takes the Johnson parameters (`a,b,c,d`) and `type` as input.
        *   It determines the correct integration limits based on the distribution type (e.g., `(-inf, inf)` for SU, `(c, c+d)` for SB).
        *   It uses Matlab's numerical integrator, `quad`, to compute the raw moments E[X], E[X^2], E[X^3], and E[X^4]. The function it integrates is `x^k * pdf(x)`, where the PDF is calculated by `f_john_dens`.
        *   It then converts these raw moments to the standard central moments (variance, skewness, kurtosis).
        *   The final lines of the function compare these calculated moments to target moments (`ka3t`, `ka4t`) and compute a squared error. This confirms the function was designed to be used as an objective function for a Matlab-based solver to find the Johnson parameters, serving the same ultimate purpose as the C function `f_hhh`.

#### Phase 4.2: Calibrating Node Probabilities (`q`)

Now that we have the price levels, we need to find the probability `q(i, n_t)` of being at each node. This is done by ensuring the tree prices options correctly.

1.  **Looping Through Time:** The code iterates through each time step `n_t` from 1 to `N`.
2.  **Setting up Optimization:** For each time step, it sets up a constrained optimization problem to find the `m` probabilities for that time.
    *   `x0`: An initial guess for the probabilities is set to a uniform distribution (`1/m`).
    *   `LB`, `UB`: The lower and upper bounds for each probability are set to `0` and `1`.
    *   `Aeq`, `beq`: The linear equality constraints are defined. This is critical.
        *   **Constraint 1:** `sum(q) = 1`. The probabilities must sum to one.
        *   **Constraint 2 (Martingale):** `sum(q .* S_t) * discount_factor = S_0`. The discounted expected asset price under the risk-neutral probabilities must equal the initial price. This enforces the no-arbitrage condition.
3.  **Running the Optimization:** The `fmincon` solver is called.
    *   It is tasked with minimizing the output of an objective function, `obj_q`, subject to the defined constraints.
    *   The objective function is passed as a function handle: `@(qq)obj_q(qq, S(:,n_t), K, call_mkt(:,n_t), put_mkt(:,n_t), B0(n_t))`.

#### Step 4.2.1: `obj_q.m` - The Node Probability Objective Function

This function calculates how "wrong" a given set of probabilities `q` is.

1.  It iterates through every strike price `K`.
2.  For each strike, it calculates the model's price for the call and put option.
    *   Model Price = `discount_factor * sum(q .* payoff)`.
3.  It calculates the squared error between the model's price and the market price for both the call and the put.
4.  It sums these squared errors and returns the mean squared error.
5.  `fmincon` adjusts `q` until this error is minimized.

#### Phase 4.3: Calibrating Transition Probabilities (`p`)

Once the node probabilities `q` are known for all time steps, the final step is to find the transition probabilities `p` between the nodes.

1.  **Looping Through Time Intervals:** The code iterates through each time interval, from `n_t = 1` to `N-1`.
2.  **Setting up Optimization:** It again sets up a constrained optimization problem to find the `m x m` transition probabilities for that interval.
    *   `p0`: The initial guess is a uniform `1/m` probability for all transitions.
    *   `LB`, `UB`: Bounds are `0` and `1`.
    *   `Aeq`, `beq`: The linear equality constraints are:
        *   **Constraint 1:** For each starting node `i`, the sum of probabilities of transitioning to any node `j` must be 1.
        *   **Constraint 2 (Martingale):** The local martingale condition must hold. The expected value of the asset price at `t+1`, given the price at node `i` at time `t`, must follow the risk-free growth.
3.  **Running the Optimization:** `fmincon` is called to minimize the objective function `obj_p_givenq`.

#### Step 4.3.1: `obj_p_givenq.m` - The Transition Probability Objective Function

This function's goal is to ensure the transition probabilities are consistent with the node probabilities we already found.

1.  It takes the transition matrix `p` (as a vector), the node probabilities at the start of the interval (`q1`), and the node probabilities at the end of the interval (`q2`).
2.  It calculates `q1' * p`. This is the Chapman-Kolmogorov forward equation, which gives the probability distribution at time `t+1` based on the distribution at `t` and the transitions between them.
3.  It calculates the sum of squared errors between this propagated distribution and the known target distribution `q2`.
4.  `fmincon` adjusts `p` until this error is minimized.

After the loops complete, `ImpWT_given_moments_underQ` returns the fully defined `S`, `p`, and `q`, completing the construction of the risk-neutral willow tree.

---

## Part 2: Constructing the Tree under the P-Measure

The process for the physical (P) measure, initiated by `demo_construct_ImpWT_underP.m`, is very similar but has a few fundamental differences.

### Step 1: `Construct_ImpWT_underP.m` - The P-Measure Orchestrator

1.  **Transform Option Prices:** The first step is entirely new. The risk-neutral (Q) market prices are not the same as the physical (P) prices. The code calls `trans_optionprice_fromQ_toP` to perform this conversion.

### Step 1.1: `trans_optionprice_fromQ_toP.m` - The Q to P Bridge

This function uses economic theory to bridge the two measures.

1.  It takes a `gam` parameter, which represents the coefficient of relative risk aversion in a CRRA utility function.
2.  It implements a formula that relates P-measure prices to Q-measure prices via a pricing kernel derived from the utility function. This involves pricing digital options on the previously built Q-measure tree.
3.  It returns `call_P` and `put_P`, the estimated P-measure option prices.

### Step 2: `Imp_Moments_underP.m` - P-Measure Moments

This function is structurally identical to its Q-measure counterpart. However, the underlying formulas in the `Imp_Moments_underP_oneMaturity` sub-function are slightly different, reflecting the different properties of the P-measure. It uses the `call_P` and `put_P` prices to derive the physical moments.

### Step 3: `ImpWT_given_moments_underP.m` - The P-Measure Construction

This function is the P-measure equivalent of `ImpWT_given_moments_underQ`. The overall structure is the same, but the optimization constraints are different.

1.  **Node Generation:** Identical to the Q-measure. The tree structure `S` is generated from the P-measure moments.
2.  **Node Probability Calibration:**
    *   **Initial Guess:** The initial guess `q0` is much more sophisticated. It uses the probabilities from the Q-measure tree (`q_Q`) and the CRRA utility parameter `gam` to create a starting point that is already close to the final answer.
    *   **CONSTRAINTS:** This is the most important difference. The `fmincon` call for the node probabilities **DOES NOT** include the martingale constraint. Under the physical measure, the expected return is the actual expected return, not the risk-free rate. The only constraint is that probabilities must sum to 1.
3.  **Transition Probability Calibration:**
    *   **CONSTRAINTS:** Similarly, the `fmincon` call for the transition probabilities **DOES NOT** include the local martingale constraint. It only ensures that the probabilities from each node sum to 1. The objective function `obj_p_givenq` is the same, ensuring mathematical consistency between the `p` and `q` matrices.

After these steps, the P-measure willow tree (`S_P`, `p_P`, `q_P`) is fully constructed.
