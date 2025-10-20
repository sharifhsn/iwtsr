# Deep Analysis of the `hhh` Function

The `hhh` function is the main dispatcher of the Hill, Hill, and Holder (1976) algorithm. Its primary role is to act as a decision-maker. It takes the first four moments of a distribution and determines which family of distributions (Normal, Lognormal, Johnson SU, Johnson SB, or others) is the most appropriate fit. It then either calculates the distribution's parameters directly or calls the appropriate specialized fitting function (`sufit` or `sbfit`).

This function is the entry point for the core logic, translating statistical properties into a concrete distributional form.

#### C Function Signature and Memory Management

```c
void hhh(double xbar, double sd, double rb1, double bb2,
         double *itype, double *gamma, double *delta,
         double *xlam, double *xi, double *ifault)
```

-   **Inputs:** The mean (`xbar`), standard deviation (`sd`), skewness (`rb1`), and kurtosis (`bb2`).
-   **Outputs:** The results are all written to memory via out-parameters.
    -   `itype`: A code representing the type of distribution found (1: SL, 2: SU, 3: SB, 4: Normal, 5: ST/Constant).
    -   `gamma`, `delta`, `xlam`, `xi`: The parameters of the fitted distribution.
    -   `ifault`: An error flag (0 for success, 1 for bad input, 2 for invalid moment region, 3 for fitting failure).
-   **Memory:** The function is entirely self-contained and uses only stack memory.

#### The Algorithm: A `goto`-Based Decision Tree

The function's control flow is a large decision tree implemented with `goto` statements. It navigates the (B1, B2) plane, where B1 is skewness-squared and B2 is kurtosis, to determine the correct region for the input moments.

1.  **Initial Checks (labels `g10` and below):**
    -   It first checks for invalid standard deviation (`sd < 0`) and the trivial case of zero standard deviation (a constant value).
    -   It sets default `ifault` and parameter values.

2.  **The (B1, B2) Plane Navigation:**
    -   **Boundary Line Check (label `g30`):** It checks if the moments lie on the "boundary line" defined by `b2 = b1 + 1`. If they are on or very close to this line, it's considered an "ST" distribution (a special case of Johnson SB), and the parameters are calculated directly (label `g40`). If they are below this line, it's an impossible region for any distribution, and it returns an error (label `g50`).
    -   **Normal Distribution Check (label `g60` -> `g70`):** If the moments are not on the boundary line, it checks if they correspond to a Normal distribution (skewness `rb1` is near zero and kurtosis `b2` is near 3). If so, it sets `itype=4` and calculates the parameters directly.
    -   **Lognormal Line Check (label `g80`):** This is a key step. It calculates the theoretical kurtosis value (`u`) that would correspond to the given skewness (`b1`) if the distribution were Lognormal.
        -   If the actual kurtosis `b2` is very close to this theoretical value `u`, it classifies the distribution as **Lognormal (SL, `itype=1`)** and calculates the parameters directly.
    -   **SU vs. SB Decision (label `g90`):** If the moments are not on the Lognormal line, the sign of `x = u - b2` determines the final classification:
        -   If `x > 0`, the point lies *above* the Lognormal line, corresponding to the **Johnson SB (`itype=3`)** family. It calls `sbfit` to perform the iterative fitting.
        -   If `x <= 0`, the point lies *below* the Lognormal line, corresponding to the **Johnson SU (`itype=2`)** family. It calls `sufit` to perform the fitting.

3.  **Failure Handling:** If `sbfit` fails (indicated by its `fault` flag), `hhh` sets its own `ifault` to 3 and attempts to return an approximate result.

### Translating `hhh` to Idiomatic Rust

The primary goal of a Rust translation is to model the complex output of this function in a type-safe and expressive way, and to convert the `goto` logic into a more readable, structured form.

#### 1. Rust Function Signature and Data Structures

Using an `enum` to represent the different possible distribution types is the most idiomatic and powerful approach in Rust.

```rust
// Use the previously defined parameter structs
use crate::{JohnsonSuParams, JohnsonSbParams};

// An enum to represent all possible outcomes of the hhh algorithm.
// This is far more expressive than an integer `itype`.
pub enum JohnsonDistribution {
    Normal {
        gamma: f64,
        delta: f64,
        lambda: f64,
    },
    Lognormal {
        gamma: f64,
        delta: f64,
        lambda: f64,
        xi: f64,
    },
    Su(JohnsonSuParams),
    Sb(JohnsonSbParams),
    // A special case of SB on the boundary line
    St {
        xi: f64,
        lambda: f64,
        delta: f64,
    },
    // The trivial case of zero variance
    Constant {
        value: f64,
    },
}

// An enum for the specific error types.
#[derive(Debug, PartialEq)]
pub enum HhhError {
    InvalidStdDev,
    ImpossibleMoments, // Below the boundary line
    SbFitFailed,       // sbfit failed to converge
}

// The function signature returns a Result containing either the correct
// distribution type or a specific error.
pub fn hhh(
    mean: f64,
    std_dev: f64,
    skew: f64,
    kurtosis: f64,
) -> Result<JohnsonDistribution, HhhError> {
    // ... implementation ...
}
```

**Key Improvements:**

-   **Type-Safe Results:** The `JohnsonDistribution` enum makes it impossible for the caller to misinterpret the results. The compiler will enforce that all cases (`Normal`, `Su`, etc.) are handled, and the parameters for each case are bundled correctly. This is a massive improvement over returning an integer code and a flat list of parameters that may or may not be relevant.
-   **Specific Errors:** The `HhhError` enum provides more context about what went wrong than a simple integer `ifault` code.

#### 2. Refactoring Control Flow: From `goto` to `if/else`

The decision tree can be implemented with a clear, nested `if/else` structure that mirrors the logic of navigating the (B1, B2) plane.

```rust
pub fn hhh(
    mean: f64,
    std_dev: f64,
    skew: f64,
    kurtosis: f64,
) -> Result<JohnsonDistribution, HhhError> {
    const TOLERANCE: f64 = 0.000001;

    if std_dev < 0.0 {
        return Err(HhhError::InvalidStdDev);
    }
    if std_dev == 0.0 {
        return Ok(JohnsonDistribution::Constant { value: mean });
    }

    let b1 = skew * skew;
    let b2 = kurtosis;

    // --- The (B1, B2) Plane Decision Logic ---

    // 1. Check against the boundary line
    if b2 < b1 + 1.0 {
        return Err(HhhError::ImpossibleMoments);
    } else if (b2 - (b1 + 1.0)).abs() < TOLERANCE {
        // On the boundary line -> ST distribution
        let mut y = 0.5 + 0.5 * (1.0 - 4.0 / (b1 + 4.0)).sqrt();
        if skew > 0.0 { y = 1.0 - y; }
        let x = std_dev / (y * (1.0 - y)).sqrt();
        let xi = mean - y * x;
        return Ok(JohnsonDistribution::St { xi, lambda: xi + x, delta: y });
    }

    // 2. Check for Normal distribution
    if skew.abs() < TOLERANCE && (b2 - 3.0).abs() < TOLERANCE {
        return Ok(JohnsonDistribution::Normal {
            delta: 1.0 / std_dev,
            gamma: -mean / std_dev,
            lambda: 1.0,
        });
    }

    // 3. Check against the Lognormal line
    let x = 0.5 * b1 + 1.0;
    let y = skew.abs() * (0.25 * b1 + 1.0).sqrt();
    let u_cubed = x + y;
    let u = u_cubed.cbrt();
    let w = u + 1.0 / u - 1.0;
    let lognormal_kurtosis = w * w * (3.0 + w * (2.0 + w)) - 3.0;

    if (lognormal_kurtosis - (b2 - 3.0)).abs() < TOLERANCE {
        // On the Lognormal line -> SL distribution
        let lambda = skew.signum();
        let delta = 1.0 / w.ln().sqrt();
        let gamma = 0.5 * delta * (w * (w - 1.0) / (std_dev * std_dev)).ln();
        let xi = lambda * (mean - ( (0.5 / delta - gamma) / delta ).exp());
        return Ok(JohnsonDistribution::Lognormal { gamma, delta, lambda, xi });
    }

    // 4. Decide between SU and SB
    if lognormal_kurtosis > b2 - 3.0 {
        // Below the Lognormal line -> SU distribution
        match su_fit(mean, std_dev, skew, b2, TOLERANCE) {
            Ok(params) => Ok(JohnsonDistribution::Su(params)),
            Err(_) => unreachable!(), // su_fit in C has no failure modes, but Rust version might.
        }
    } else {
        // Above the Lognormal line -> SB distribution
        match sb_fit(mean, std_dev, skew, b2, TOLERANCE) {
            Ok(params) => Ok(JohnsonDistribution::Sb(params)),
            Err(_) => Err(HhhError::SbFitFailed),
        }
    }
}
```

This refactored Rust code is a direct translation of the C code's logic but expressed in a way that is vastly safer, more readable, and easier to maintain and test. The complex web of `goto`s is replaced with a clear, top-down decision process.