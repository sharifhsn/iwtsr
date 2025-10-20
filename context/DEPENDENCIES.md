# Project Dependencies

This document outlines the external dependencies used in the Matlab implementation of the Implied Willow Tree. It details the API surface exposed to the application to guide the replacement of these dependencies in a Rust implementation.

## 1. Matlab Toolboxes

These are standard Matlab functions that are not part of the core language. They are typically found in specialized toolboxes.

### Optimization Toolbox

This toolbox provides functions for solving optimization problems. The primary function used is `fmincon`.

-   **`fmincon`**: Solves constrained nonlinear optimization problems.
    -   **Purpose:** This is the core of the calibration process. It's used to find the node probabilities (`q`) and transition probabilities (`p`) that minimize a given objective function subject to a set of constraints.
    -   **API Surface:** `[x, fval] = fmincon(fun, x0, A, b, Aeq, beq, lb, ub, nonlcon, options)`
        -   `fun`: The objective function to minimize (e.g., `@(qq)obj_q(...)`).
        -   `x0`: Initial guess for the parameters.
        -   `A`, `b`: Linear inequality constraints (`A*x <= b`). Not used in this project.
        -   `Aeq`, `beq`: Linear equality constraints (`Aeq*x = beq`). Used extensively to enforce probability and martingale constraints.
        -   `lb`, `ub`: Lower and upper bounds on the variables (e.g., probabilities are between 0 and 1).
        -   `nonlcon`: Nonlinear constraint function. Not used in this project.
        -   `options`: Optimization options set by `optimoptions`.
    -   **Rust Equivalent:** A numerical optimization library like `nlopt` or `argmin`.

-   **`optimoptions`**: Configures the options for the solver.
    -   **Purpose:** To set parameters for the `fmincon` solver, such as the algorithm (`sqp`), display level (`iter`), and tolerance.
    -   **API Surface:** `options = optimoptions(solver, Name, Value, ...)`
        -   `solver`: The solver to configure (e.g., `@fmincon`).
        -   `Name, Value`: Pairs of option names and their values (e.g., `'Algorithm', 'sqp'`).
    -   **Rust Equivalent:** The configuration struct or builder pattern for the chosen optimization library.

### Statistics and Machine Learning Toolbox

This toolbox provides functions for statistical analysis and probability distributions.

-   **`norminv`**: Inverse of the standard normal cumulative distribution function (CDF).
    -   **Purpose:** Used in `WTnodes_from_JohnsonCurve` to generate the initial `z` quantiles from a set of probabilities `q`.
    -   **API Surface:** `z = norminv(q, mu, sigma)`
    -   **Rust Equivalent:** A statistics library like `statrs` provides the `inverse_cdf` method for the normal distribution.

-   **`normcdf`**: Standard normal cumulative distribution function (CDF).
    -   **Purpose:** Used in `BS_GBM_Euro` and `BS_Greeks_GBM_Euro` to calculate Black-Scholes option prices and deltas.
    -   **API Surface:** `val = normcdf(x)`
    -   **Rust Equivalent:** A statistics library like `statrs` provides the `cdf` method for the normal distribution.

-   **`normpdf`**: Standard normal probability density function (PDF).
    -   **Purpose:** Used in `BS_Greeks_GBM_Euro` to calculate option gammas and thetas.
    -   **API Surface:** `val = normpdf(x)`
    -   **Rust Equivalent:** A statistics library like `statrs` provides the `pdf` method for the normal distribution.

-   **`interp1`**: 1-D data interpolation.
    -   **Purpose:** Used to find values at intermediate points. For example, in `Imp_Moments_underP`, it's used to find the option price at the specific strike `S0`. In `ImpWT_given_moments_underP`, it's used to interpolate Q-measure probabilities onto the P-measure tree nodes.
    -   **API Surface:** `vq = interp1(x, v, xq, method)`
        -   `x`, `v`: The sample points and corresponding values.
        -   `xq`: The query points.
        -   `method`: The interpolation method (e.g., `'spline'`).
    -   **Rust Equivalent:** A numerical library like `ndarray-interp` or `interpolator`.

### Standard Matlab Functions

These functions are part of the standard Matlab environment.

-   **`quad`**: Numerical integration using adaptive Simpson quadrature.
    -   **Purpose:** Used in `f_john_mom.m` to calculate the moments of a Johnson distribution by integrating over its density.
    -   **API Surface:** `Q = quad(fun, a, b)`
        -   `fun`: The function to integrate.
        -   `a`, `b`: The lower and upper limits of integration.
    -   **Rust Equivalent:** A numerical integration library like `quadrature` or `gauleg`.

-   **`blkdiag`**: Block diagonal matrix construction.
    -   **Purpose:** Used to efficiently construct the large, sparse `Aeq` matrix for the transition probability optimization problem in `ImpWT_given_moments_underQ` and `ImpWT_given_moments_underP`.
    -   **API Surface:** `M = blkdiag(A, B, ...)`
    -   **Rust Equivalent:** This can be constructed manually in Rust using a sparse matrix representation from a library like `nalgebra` or by carefully placing blocks into a dense `ndarray` matrix.

-   **`eval`**: Execute a string containing a Matlab expression.
    -   **Purpose:** Used to dynamically create the call to `blkdiag` based on the number of nodes `m`. This is a workaround for `blkdiag` not accepting a cell array of matrices.
    -   **API Surface:** `eval(expression)`
    -   **Rust Equivalent:** This is a language feature that should be avoided. The equivalent in Rust would be to programmatically construct the matrix in a loop, which is safer and more efficient.

## 2. Compiled C/C++ Dependencies

This project relies on a pre-compiled C function for a performance-critical calculation.

-   **`f_hhh.c`**: A C function that calculates the parameters of a Johnson distribution given the first four moments.
    -   **Purpose:** This is a key part of the `WTnodes_from_JohnsonCurve` function. It takes the moments of the log-return distribution and finds the `a, b, c, d` parameters and the type (`itype`) of the Johnson curve that matches them.
    -   **API Surface (from `.m` file):** `[a, b, d, c, itype, ifault] = f_hhh(mu, sd, ka3, ka4)`
        -   **Inputs:** `mu` (mean), `sd` (std dev), `ka3` (skewness), `ka4` (kurtosis).
        -   **Outputs:** `a, b, c, d` (Johnson parameters), `itype` (curve type 1-4), `ifault` (error flag).
    -   **Rust Equivalent:** The `f_hhh.c` source file is provided. This could be compiled and linked to the Rust project using a C Foreign Function Interface (FFI) via a crate like `cc`. Alternatively, the logic could be translated directly into Rust.
