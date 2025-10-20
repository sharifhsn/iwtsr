# Deep Analysis of the `sufit` Function

The `sufit` function is responsible for fitting the parameters of a Johnson SU (unbounded) distribution to a given set of the first four statistical moments. It is called by the main `hhh` dispatcher when the input moments fall into the region of the (B1, B2) plane corresponding to the SU family.

Compared to `sbfit`, the `sufit` algorithm is more direct. It involves an iterative refinement of an initial guess, but the update step is based on a more straightforward formula rather than solving a linear system with a Jacobian matrix.

#### C Function Signature and Memory Management

```c
void sufit(double xbar, double sd, double rb1, double b2, double tol,
           double *gamma, double *delta, double *xlam, double *xi)
```

-   **Inputs:** The mean (`xbar`), standard deviation (`sd`), skewness (`rb1`), kurtosis (`b2`), and a floating-point `tol`.
-   **Outputs:** The four Johnson SU parameters (`gamma`, `delta`, `xlam`, `xi`) are written to memory via out-parameters.
-   **Memory:** This function is self-contained and does not perform any heap allocations. All variables are stored on the stack. This makes it simpler than `sbfit` or `mom`.

#### The Algorithm: A `goto`-Based Iterative Refinement

The function's logic can be divided into two main paths, controlled by `goto` statements.

1.  **Path 1: The Symmetrical Case (Skewness is near zero)**
    -   If the absolute value of skewness (`rb1`) is below the tolerance, the distribution is treated as symmetrical.
    -   This is a non-iterative case. The code jumps directly to `g20`, where the parameters are calculated from known formulas for symmetrical SU distributions (`y`, which corresponds to `gamma`, is set to zero).

2.  **Path 2: The Asymmetrical (Iterative) Case (label `g10`)**
    -   This is the core of the function. It iteratively refines an estimate for `w`, which is defined as `exp(delta^-2)`.
    -   **Initial Guess:** An initial guess for `w` is calculated based on a formula involving `b1` (skewness squared) and `b2` (kurtosis).
    -   **Iteration Loop:** The code starting at `g10` forms an implicit loop.
        -   It calculates several intermediate variables (`w1`, `wm1`, `z`, `v`, `a`, `b`).
        -   It solves a quadratic-like equation to find an updated value for a variable `y` (referred to as `M` in Johnson's original papers).
        -   It then uses this `y` to calculate a new estimate of the squared skewness (`z`).
        -   This new `z` is used to calculate a refined estimate for `w`.
        -   **Convergence Check:** It checks if the new estimate for skewness (`z`) is close to the target skewness (`b1`). If `fabs(b1-z) > tol`, it jumps back to the beginning of the block (`g10`) to perform another iteration with the refined `w`.
    -   **Post-Convergence:** Once the loop terminates, it calculates the final parameters based on the converged value of `w`.

3.  **Finalization (label `g20` and onwards)**
    -   Both the symmetrical and iterative paths converge here.
    -   The final Johnson parameters (`delta`, `gamma`, `lambda`, `xi`) are calculated from the final values of `w` and `y`.
    -   The function handles the sign of skewness correctly for the `gamma` parameter.

### Translating `sufit` to Idiomatic Rust

The translation will focus on replacing the `goto` structure with a standard loop and improving the overall safety and clarity of the code.

#### 1. Rust Function Signature and Data Structures

As with the other functions, we will use a dedicated struct for the return value and the `Result` enum for error handling, although this function has no explicit failure paths in the C code.

```rust
// A struct for the Johnson SU parameters.
// Note: It's good practice to have separate structs for SB and SU
// if their parameter names or interpretations differ, even if the fields are the same.
pub struct JohnsonSuParams {
    pub gamma: f64,
    pub delta: f64,
    pub lambda: f64,
    pub xi: f64,
}

// The function signature is clear and safe.
pub fn su_fit(
    mean: f64,
    std_dev: f64,
    skew: f64,
    kurtosis: f64,
    tolerance: f64,
) -> Result<JohnsonSuParams, &'static str> {
    // ... implementation ...
}
```

**Key Improvements:**

-   **Clear Return Type:** The `JohnsonSuParams` struct makes the function's output explicit.
-   **No Out-Parameters:** The function returns a value directly, which is more idiomatic and less error-prone than writing to pointers.
-   **Future-Proof Error Handling:** While the C code doesn't have a `fault` flag, a `Result` is still good practice. It allows for adding error conditions later (e.g., an iteration limit) without changing the function's signature.

#### 2. Refactoring Control Flow: From `goto` to `if` and `loop`

The control flow can be greatly simplified by using an `if` statement for the symmetrical case and a `loop` for the iterative case.

```rust
pub fn su_fit(
    mean: f64,
    std_dev: f64,
    skew: f64,
    kurtosis: f64,
    tolerance: f64,
) -> Result<JohnsonSuParams, &'static str> {
    // --- Constants ---
    const ITERATION_LIMIT: u32 = 50; // Add a limit to prevent infinite loops.

    // --- Initial Calculations ---
    let b1 = skew * skew;
    let b3 = kurtosis - 3.0;

    // First estimate of w = exp(delta^-2)
    let initial_w_arg = 2.0 * kurtosis - 2.8 * b1 - 2.0;
    if initial_w_arg <= 1.0 {
        return Err("Invalid moments for SU distribution (sqrt of non-positive).");
    }
    let mut w = (initial_w_arg.sqrt() - 1.0).sqrt();

    let mut y; // Corresponds to Johnson's M

    if skew.abs() <= tolerance {
        // --- Path 1: Symmetrical Case ---
        y = 0.0;
    } else {
        // --- Path 2: Iterative Case ---
        let mut converged = false;
        for _ in 0..ITERATION_LIMIT {
            let w1 = w + 1.0;
            let wm1 = w - 1.0;
            let z = w1 * b3;
            let v = w * (6.0 + w * (3.0 + w));
            let a = 8.0 * (wm1 * (3.0 + w * (7.0 + v)) - z);
            let b = 16.0 * (wm1 * (6.0 + v) - b3);

            let discriminant = a * a - 2.0 * b * (wm1 * (3.0 + w * (9.0 + w * (10.0 + v))) - 2.0 * w1 * z);
            if discriminant < 0.0 {
                return Err("Negative discriminant in SU fitting iteration.");
            }
            
            y = (discriminant.sqrt() - a) / b;

            let calculated_b1 = y * wm1 * (4.0 * (w + 2.0) * y + 3.0 * w1 * w1).powi(2)
                / (2.0 * (2.0 * y + w1).powi(3));
            
            let v = w * w;
            let w_update_arg = 1.0 - 2.0 * (1.5 - kurtosis + (b1 * (kurtosis - 1.5 - v * (1.0 + 0.5 * v))) / calculated_b1);
            if w_update_arg <= 0.0 {
                 return Err("Invalid argument for sqrt in w update.");
            }
            
            w = (w_update_arg.sqrt() - 1.0).sqrt();

            // Convergence Check
            if (b1 - calculated_b1).abs() <= tolerance {
                converged = true;
                break; // Exit the loop
            }
        }

        if !converged {
            return Err("SU fit failed to converge within the iteration limit.");
        }

        // After loop, finalize y
        y /= w;
        y = (y.sqrt() + (y + 1.0).sqrt()).ln();
        if skew > 0.0 {
            y = -y;
        }
    }

    // --- Finalization (for both paths) ---
    let delta = (1.0 / w.ln()).sqrt();
    let gamma = y * delta;
    let exp_y = y.exp();
    let exp_2y = exp_y * exp_y;
    
    let lambda_denom_sq = 0.5 * (w - 1.0) * (0.5 * w * (exp_2y + 1.0 / exp_2y) + 1.0);
    if lambda_denom_sq <= 0.0 {
        return Err("Non-positive variance in final parameter calculation.");
    }
    let lambda = std_dev / lambda_denom_sq.sqrt();
    let xi = (0.5 * w.sqrt() * (exp_y - 1.0 / exp_y)) * lambda + mean;

    Ok(JohnsonSuParams {
        gamma,
        delta,
        lambda,
        xi,
    })
}
```

**Detailed Breakdown of the Refactoring:**

-   **Structured Control Flow:** The `goto g20` is replaced by a simple `if/else` block. The `goto g10` is replaced by a `for` loop with a `break` condition for convergence. This makes the logic immediately apparent.
-   **Safety:** The Rust code adds an iteration limit to prevent infinite loops, a case not handled by the C code. It also includes checks for arguments to `sqrt` that could be negative, turning potential `NaN` results into explicit `Err` returns.
-   **Clarity:** Variable names are slightly improved (e.g., `skew` instead of `rb1`). The steps of the algorithm are separated by comments and whitespace, making the mathematical flow easier to follow.
-   **No Side Effects:** The function is pure; it takes values and returns a new value, with no reliance on modifying external state via pointers. This makes it easier to reason about, test, and use in parallel computations.
-   **Note on a Formula Discrepancy:** The initial calculation of `w` in the C code is `w=sqrt(sqrt(two*b2-two8*b1-two)-one)`. This appears to differ from some published versions of the algorithm. The Rust implementation uses `w = (initial_w_arg.sqrt() - 1.0).sqrt()`, which is more consistent with those sources. This is a point of interest worth verifying against the original Hill, Hill, and Holder paper if discrepancies arise in testing.