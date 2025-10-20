# Deep Analysis of the `mom` Function

The `mom` function is a C port of a Fortran subroutine. Its purpose is to calculate the first six raw moments of a distribution related to the Johnson SB (bounded) system. This is a crucial helper function for the `sbfit` routine, which tries to find the Johnson SB distribution parameters that match a given set of moments.

The function is purely numerical, implementing an iterative algorithm to approximate the value of an integral that defines the moments. The integral is calculated by summing an infinite series.

#### C Function Signature and Memory Management

```c
void mom(double g, double d, double *a, double *fault)
```

-   **Inputs:** `g` and `d` are the parameters (gamma and delta) that define the shape of the distribution whose moments are being calculated.
-   **Outputs:**
    -   `a`: A raw pointer to a `double` array. This is an **out-parameter**. The caller must ensure that this pointer points to a block of memory large enough to hold at least 6 doubles. The function writes the six calculated moments into this memory.
    -   `fault`: A pointer to a `double`, used as an error flag. The function sets `*fault = 1.0` on failure.
-   **Memory:** The function uses a custom allocator `dvector` (from `nrutil.c`) to allocate two temporary arrays, `b` and `c`, on the heap. It is responsible for freeing this memory with `free_dvector` before it returns.

#### The Algorithm: A `goto`-Based Nested Loop

The most challenging aspect of this code is its control flow, which is managed by `goto` statements. This style is a direct translation from old Fortran and makes the logic difficult to follow. By tracing the jumps, we can uncover the underlying structure: a nested loop.

1.  **The Outer Loop (label `g20`):** This is an iterative refinement loop. It searches for an appropriate step size, `h`, for the numerical integration.
    -   It starts with a trial value for `h`.
    -   If the algorithm fails to converge after an inner loop, it halves `h` (`h=half*h;`) and retries the entire calculation.
    -   The `c` array is used to store the results (`a`) of the *previous* outer loop iteration. At the end of an iteration, it checks if the change between `c` and the new `a` is within a tolerance (`zz`). If it is, the outer loop has converged, and the function can return successfully.

2.  **The Inner Loop (label `g60`):** This loop approximates the value of an infinite series for a *given* step size `h`.
    -   The `b` array stores the results (`a`) of the *previous* inner loop iteration.
    -   The core of the loop is a `for` loop that adds new terms to the series for each of the six moments. The terms `p` and `q` are calculated based on the current state.
    -   After adding new terms, it checks if the change between `b` and the new `a` is within a tolerance (`vv`). If it is, the inner loop has converged for the current `h`, and control flow proceeds to the outer loop's convergence check.

3.  **Error Handling (label `g140`):** This is a failure state. It's triggered if an iteration limit is exceeded or if an input (`w`) is too large, risking numerical overflow. It sets the fault flag and jumps to the cleanup code.

4.  **Cleanup (label `gmem`):** This section frees the heap-allocated memory for `b` and `c`. All successful and failed paths must end here to prevent memory leaks.

### Translating `mom` to Idiomatic Rust

Translating this function to Rust requires a complete restructuring of the control flow and memory management to align with Rust's principles of safety and clarity.

#### 1. Rust Function Signature: Safety and Expressiveness

In Rust, we avoid raw pointers and manual memory management. The function's signature should be self-documenting and safe.

```rust
// Rust is 0-indexed, so arrays will be of size 6 (indices 0-5).
// The function returns a Result: Ok with a 6-element array on success,
// or an Err with a string describing the failure.
pub fn mom(g: f64, d: f64) -> Result<[f64; 6], &'static str> {
    // ... implementation ...
}
```

**Key Improvements:**

-   **No Raw Pointers:** The output is a fixed-size array `[f64; 6]`, which is a value type in Rust. The caller receives the result by value, eliminating any ambiguity about memory ownership.
-   **RAII for Memory:** Temporary arrays like `b` and `c` will be stack-allocated (`let mut b = [0.0; 6];`). Rust's ownership model guarantees they are automatically cleaned up when they go out of scope. There is no need for manual `malloc`/`free`.
-   **Explicit Error Handling:** The `Result` enum is the idiomatic way to handle operations that can fail. It forces the calling code to explicitly handle the `Ok` and `Err` cases, preventing bugs where an error flag might be ignored.

#### 2. Refactoring Control Flow: From `goto` to Structured Loops

The core of the translation is to rewrite the `goto` spaghetti as structured loops.

```rust
pub fn mom(g: f64, d: f64) -> Result<[f64; 6], &'static str> {
    // --- Constants ---
    let zz = 1.0e-5;  // Outer loop tolerance
    let vv = 1.0e-8;  // Inner loop tolerance
    let limit = 500;
    // ... other constants ...

    // --- State Variables ---
    let mut a = [0.0; 6];
    let mut c = [0.0; 6]; // Stores previous outer loop state

    let w = g / d;
    if w > 80.0 { // expa
        return Err("Input 'g/d' is too large, may cause overflow.");
    }

    let e = w.exp() + 1.0;
    let r = 1.414213562 / d; // rttwo / d
    let mut h = if d < 3.0 { 0.25 * d } else { 0.75 };

    // --- The Outer Loop ---
    for k in 1..=limit {
        // On subsequent iterations, update `c` and refine `h`
        if k > 1 {
            c.copy_from_slice(&a);
            h *= 0.5;
        }

        // --- Initialization for the current `h` ---
        let mut t = w;
        let mut u = w;
        let mut y = h * h;
        let x = 2.0 * y;
        a[0] = 1.0 / e;
        for i in 1..6 {
            a[i] = a[i - 1] / e;
        }
        let mut v = y;
        let f = r * h;

        // --- The Inner Loop (Series Summation) ---
        let inner_loop_succeeded = 'inner: loop {
            // ... implementation of the inner loop ...
            // If it converges, we break the loop successfully.
            // let inner_converged = ...;
            // if inner_converged {
            //     break 'inner true;
            // }
            // If it exceeds its limit, we fail.
            // let m = 0;
            // if m > limit {
            //     break 'inner false;
            // }
        };

        if !inner_loop_succeeded {
            return Err("Inner loop (series summation) failed to converge.");
        }

        // --- Finalize `a` and Check for Outer Loop Convergence ---
        let v_final = 0.5641895835 * h; // rrtpi * h
        for val in a.iter_mut() {
            *val *= v_final;
        }

        if k > 1 {
            let mut outer_converged = true;
            for i in 0..6 {
                if a[i] == 0.0 { return Err("Moment calculation resulted in zero."); }
                if ((a[i] - c[i]) / a[i]).abs() > zz {
                    outer_converged = false;
                    break;
                }
            }
            if outer_converged {
                return Ok(a); // Success!
            }
        }
    }

    Err("Outer loop failed to converge.")
}
```

**Detailed Breakdown of the Refactoring:**

-   **Outer Loop:** A `for k in 1..=limit` loop naturally handles the iteration count and the failure condition.
-   **Inner Loop:** A named loop `'inner: loop` would be used. This allows us to `break` out of it specifically. It can return a boolean to indicate whether it converged or timed out.
-   **State Updates:** The logic for updating `c` and `h` is now cleanly placed at the beginning of the outer loop, guarded by an `if k > 1` check.
-   **Indexing:** All loops and array accesses are converted to be 0-indexed (e.g., `for i in 1..6` and `a[i-1]`).
-   **Clarity:** The nested structure makes the relationship between the two convergence processes (inner series vs. outer step-size) much clearer than the interleaved `goto` statements.

This refactored Rust code is not only safer due to the language's features but is also vastly more readable and maintainable. The algorithmic complexity remains the same, but the cognitive complexity for a developer is significantly reduced.
