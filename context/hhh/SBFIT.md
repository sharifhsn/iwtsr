# Deep Analysis of the `sbfit` Function

The `sbfit` function is the iterative heart of the Johnson SB (bounded) distribution fitting process. Its purpose is to find the distribution parameters (`gamma`, `delta`, `lambda`, `xi`) that match a target set of the first four statistical moments: mean (`xbar`), standard deviation (`sigma`), skewness (`rtb1`), and kurtosis (`b2`).

It implements a Newton-Raphson-like numerical method to solve this multi-dimensional root-finding problem. The function is highly complex due to the non-linear relationship between the parameters and the moments.

#### C Function Signature and Memory Management

```c
void sbfit(double xbar, double sigma, double rtb1, double b2, double tol,
           double *gamma, double *delta, double *xlam, double *xi, double *fault)
```

-   **Inputs:** The first four moments and a `tol` for floating-point comparisons.
-   **Outputs:** All results are written to memory via out-parameters (`gamma`, `delta`, `xlam`, `xi`, `fault`). The caller must provide valid pointers to `double` variables.
-   **Memory:** It allocates three temporary arrays (`hmu`, `deriv`, `dd`) on the heap using `dvector` and is responsible for freeing them via `free_dvector` before returning, regardless of the exit path.

#### The Algorithm: A `goto`-Based Fitter

The function's control flow can be broken down into three main phases, all connected by `goto` statements.

1.  **Phase 1: Initial Guess Calculation (labels `g5` through `g70`)**
    -   This is a long, branching sequence of calculations designed to produce a high-quality initial guess for the `g` (gamma) and `d` (delta) parameters.
    -   The logic follows a decision tree based on the input moments. The numerous constants (`a1` through `a22`) are coefficients for various approximation formulas derived from the original paper by Hill, Hill, and Holder.
    -   This section is essentially a hard-coded empirical rulebook for starting the main iteration as close to the solution as possible to ensure convergence.

2.  **Phase 2: Main Iteration Loop (label `g80`)**
    -   This is the core of the function, implementing a numerical solver.
    -   **Step 2a: Calculate Moments:** It calls the `mom(g, d, ...)` function with its current guess for `g` and `d` to get the first six raw moments (`hmu`) that these parameters produce.
    -   **Step 2b: Calculate Resulting Skewness/Kurtosis:** It converts the raw moments from `mom` into the corresponding central moments, skewness (`rbet`), and kurtosis (`bet2`).
    -   **Step 2c: Calculate Derivatives (Jacobian Matrix):** This is the most mathematically intense part. It calculates the partial derivatives of skewness and kurtosis with respect to the parameters `g` and `d`. This forms a 2x2 Jacobian matrix (`deriv`). The calculation is done in the nested loops between labels `g90` and `g100`.
    -   **Step 2d: Solve for the Update Step:** It computes the difference between the current and target moments (`rbet-rb1`, `bet2-b2`). It then solves the 2x2 linear system `Jacobian * [du; dy] = [error_skew; error_kurtosis]` to find the optimal update step (`u` for `g`, `y` for `d`). This is the essence of the Newton-Raphson method.
    -   **Step 2e: Update and Check Convergence:** It updates the parameters (`g = g - u; d = d - y;`) and checks if the magnitude of the update step is below a tolerance (`tt`). If not, it jumps back to the start of the loop (`g80`).

3.  **Phase 3: Finalization and Cleanup (labels `g130`, `g140`, `gmem`)**
    -   Once the loop converges, it calculates the final output parameters (`*delta`, `*gamma`, `*xlam`, `*xi`) from the converged `g` and `d`.
    -   It handles the sign of skewness correctly.
    -   Finally, it jumps to `gmem` to free the heap-allocated memory before returning.

### Translating `sbfit` to Idiomatic Rust

This function is a prime candidate for refactoring. The goal is to improve safety, clarity, and maintainability.

#### 1. Rust Function Signature and Data Structures

We'll define a struct to hold the results and use the `Result` enum for error handling.

```rust
// A dedicated struct makes the return type clear and self-documenting.
pub struct JohnsonSbParams {
    pub gamma: f64,
    pub delta: f64,
    pub lambda: f64,
    pub xi: f64,
}

// The function signature clearly defines inputs and the possible outcomes.
pub fn sb_fit(
    mean: f64,
    std_dev: f64,
    skew: f64,
    kurtosis: f64,
    tolerance: f64,
) -> Result<JohnsonSbParams, &'static str> {
    // ... implementation ...
}
```

**Key Improvements:**

-   **Struct for Results:** Returning a `JohnsonSbParams` struct is much cleaner than passing 7 raw pointers. It's a single, cohesive value.
-   **Safe Memory:** All temporary arrays will be allocated on the stack (e.g., `let mut hmu = [0.0; 6];`). Rust's ownership rules guarantee they are cleaned up automatically (RAII), eliminating the need for manual `free` calls and the risk of memory leaks.
-   **Robust Error Handling:** The `Result` enum will clearly communicate success or failure. The `?` operator can be used to propagate errors from the `mom` function call, simplifying the code.

#### 2. Refactoring Control Flow: From `goto` to Functions and Loops

The complex logic can be broken down into smaller, more manageable pieces.

```rust
pub fn sb_fit(
    mean: f64,
    std_dev: f64,
    skew: f64,
    kurtosis: f64,
    tolerance: f64,
) -> Result<JohnsonSbParams, &'static str> {
    // --- Constants ---
    // Define a1-a22 constants here...
    const TT: f64 = 1.0e-4; // Iteration tolerance
    const LIMIT: i32 = 50;

    // --- Phase 1: Initial Guess ---
    // This complex logic is encapsulated in its own function.
    let (mut g, mut d) = calculate_initial_guess(skew, kurtosis, tolerance)?;

    // --- Phase 2: Main Iteration Loop ---
    for _ in 0..LIMIT {
        // Step 2a: Call the Rust version of `mom`. The `?` handles the error.
        let hmu = mom(g, d)?;

        // Step 2b & 2c: Encapsulate the derivative calculation.
        let (current_skew, current_kurtosis, jacobian) =
            calculate_derivatives(&hmu, g, d);

        // Step 2d: Solve for the update step.
        let error_skew = current_skew - skew.abs();
        let error_kurtosis = current_kurtosis - kurtosis;
        let (update_g, update_d) = solve_update_step(jacobian, error_skew, error_kurtosis);

        // Step 2e: Update parameters and check for convergence.
        g -= update_g;
        if skew.abs() * skew.abs() < f64::EPSILON || g < 0.0 {
            g = 0.0;
        }
        d -= update_d;

        if update_g.abs() < TT && update_d.abs() < TT {
            // --- Phase 3: Finalization ---
            // The loop converged, calculate final parameters.
            let h2 = hmu[1] - hmu[0] * hmu[0];
            if h2 <= 0.0 {
                return Err("Non-positive variance encountered during fitting.");
            }
            let lambda = std_dev / h2.sqrt();
            let final_gamma = if skew < 0.0 { -g } else { g };
            let final_hmu1 = if skew < 0.0 { 1.0 - hmu[0] } else { hmu[0] };
            let xi = mean - lambda * final_hmu1;

            return Ok(JohnsonSbParams {
                gamma: final_gamma,
                delta: d,
                lambda,
                xi,
            });
        }
    }

    Err("SB fit failed to converge within the iteration limit.")
}
```

**Detailed Breakdown of the Refactoring:**

-   **Encapsulation:** The logic for the initial guess and the derivative calculation, which are complex but self-contained, are moved into their own helper functions (`calculate_initial_guess`, `calculate_derivatives`). This makes the main `sb_fit` function much easier to read and understand.
-   **Structured Loop:** The main iteration is a simple `for` loop. The success condition is a `return Ok(...)` from inside the loop. If the loop finishes without converging, it's an error.
-   **Variable Naming:** Variables are given more descriptive names (e.g., `skew` instead of `rtb1`, `update_g` instead of `u`).
-   **Error Propagation:** The call to `mom(g, d)?` is a perfect example of idiomatic Rust. If `mom` returns an `Err`, `sb_fit` will immediately stop and propagate that error to its own caller. This avoids manual checking of fault flags.
-   **Clarity and Intent:** The code now reads like a description of the algorithm: get an initial guess, then loop until convergence by calculating derivatives and updating the parameters. The `goto`s that obscured this simple structure are gone.
