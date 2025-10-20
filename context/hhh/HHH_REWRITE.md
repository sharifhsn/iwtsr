# Plan for Rewriting `f_hhh.c` in Rust

This document outlines the plan to create a self-contained, idiomatic, and verifiable Rust library that replicates the functionality of the original `f_hhh.c` code. The project will include the pure Rust rewrite, a C FFI wrapper for the original code, and a test harness to compare them directly.

## 1. Idiomatic Rust Project Structure

The project will be set up as a standard Cargo library. The file layout will be as follows:

```
.
├── Cargo.toml        # Project manifest, includes dependencies like `cc`.
├── build.rs          # Build script to compile the C code.
├── c_code/           # Directory to hold the original C source files.
│   ├── f_hhh.c
│   ├── nr.h
│   ├── nrutil.c
│   └── nrutil.h
└── src/
    └── lib.rs        # The main library file containing all Rust code.
```

-   **`Cargo.toml`**: Will define the library and its dependencies. The key dependency is `cc = "1.0"` in the `[build-dependencies]` section, which is used by the build script.
-   **`build.rs`**: This script will be responsible for compiling `f_hhh.c` and `nrutil.c` into a static library that the Rust code can link against. This automates the C compilation process, making the project self-contained.
-   **`c_code/`**: Isolating the original C source files keeps the project clean and clearly separates the legacy code from the new Rust implementation.
-   **`src/lib.rs`**: For simplicity and to keep the logic tightly coupled, this single file will contain:
    1.  **The Public API**: The top-level data structures (`JohnsonDistribution`, `HhhError`) and the main `hhh` function.
    2.  **A `rust_impl` module**: The pure Rust implementation of `hhh`, `sufit`, `sbfit`, and `mom`.
    3.  **A `c_ffi` module**: The FFI bindings to the compiled C code and a safe wrapper function.
    4.  **A `tests` module**: The test harness that calls both the `rust_impl` and `c_ffi` modules and compares their outputs.

## 2. The C Wrapper (FFI)

The goal is to call the original C code from Rust to create a baseline for testing.

1.  **Compilation (`build.rs`)**: The `cc` crate will be used in `build.rs` to compile `c_code/f_hhh.c` and `c_code/nrutil.c`. It will create a static library named `libf_hhh.a` that Cargo will automatically link.
2.  **FFI Bindings (`src/lib.rs` -> `c_ffi` module)**:
    -   An `extern "C"` block will declare the function signature of the C `hhh` function. This block will be marked `unsafe` as Rust cannot guarantee its safety.
    -   A **safe public wrapper function** will be created. This function will take standard Rust types as input, call the `unsafe` C function using pointers, and convert the C-style output (integer codes, pointers) into the idiomatic Rust `Result<JohnsonDistribution, HhhError>` type. This provides a safe, easy-to-use interface for the rest of the Rust code, specifically the test harness.

## 3. The Pure Rust Implementation

This will be a complete rewrite based on the deep analysis in `MOM.md`, `SBFIT.md`, `SUFIT.md`, and `HHH.md`.

-   **Control Flow**: All `goto` statements will be refactored into structured `if/else` blocks and `for`/`loop` constructs.
-   **Memory Safety**: All memory will be managed by Rust's ownership system. Raw pointers and manual allocations (`dvector`) will be eliminated in favor of stack-allocated arrays and standard library types.
-   **API Design**: The function signatures will be idiomatic, using structs and enums for inputs and the `Result` type for outputs, ensuring type safety and expressive error handling.
-   **Code Organization**: The implementation will be placed inside a `rust_impl` module to keep it separate from the FFI code.

## 4. The Test Harness

The verification of the rewrite is the most critical step.

-   **Framework**: Rust's built-in test framework (`#[cfg(test)]`) will be used.
-   **Strategy**: A series of test functions will be created, each corresponding to a specific region in the (B1, B2) plane (e.g., `test_normal_case`, `test_su_case`, `test_sb_case`).
-   **Execution**: Inside each test:
    1.  A set of input moments (mean, std_dev, skew, kurtosis) will be defined.
    2.  The safe C wrapper function from the `c_ffi` module will be called with these inputs.
    3.  The pure Rust function from the `rust_impl` module will be called.
    4.  The results of both calls will be compared. `assert_eq!` will be used to check that both implementations returned the same `JohnsonDistribution` variant and the same `HhhError` type.
    5.  For the parameters within the successful results, a floating-point comparison with a small tolerance (e.g., `1e-9`) will be performed to ensure numerical equivalence.

This comprehensive approach ensures that the final Rust library is not only safer and more maintainable but is also a functionally correct and verified replacement for the original C code.