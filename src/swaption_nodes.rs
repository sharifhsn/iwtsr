//! Generates interest rate nodes for swaption pricing using the Implied Willow Tree method.
//!
//! This module adapts the generic Johnson-curve-based node generation logic for the specific
//! use case of swaptions, which operate in a normal (Bachelier) world rather than a
//! lognormal (Black-Scholes) world.

use crate::nodes::{wt_nodes_from_johnson_curve, WtNodesError};
use ndarray::Array2;

/// A struct to hold the results of the swaption node generation process.
#[derive(Debug)]
pub struct WtNodesSwaptionResult {
    /// A vector of integers representing the Johnson distribution type for each time step.
    /// (1: SL, 2: SU, 3: SB, 4: Normal)
    pub itype: Vec<i32>,
    /// A matrix containing the absolute forward swap rate values for each node and time step. (M x N)
    pub forward_rate_nodes: Array2<f64>,
    /// A matrix containing the fitted Johnson parameters (a, b, c, d) for each time step. (4 x N)
    pub h_params: Array2<f64>,
}

/// Generates the forward swap rate nodes for a willow tree, adapted for the normal model.
///
/// This function serves as a specific wrapper around the generic `wt_nodes_from_johnson_curve`
/// function. It correctly interprets the inputs and outputs for swaption pricing.
///
/// # Arguments
///
/// * `f0` - The initial forward swap rate at time 0.
/// * `g` - A 2D array (`4 x N`) containing the first four central moments (mean, variance, skew, kurtosis)
///   of the **absolute change in the forward rate (Y_T = F_T - F_0)** for each of the `N` time steps.
///   Note: The mean (first row of `g`) should be zero, as the forward rate is a martingale
///   under the forward swap measure.
/// * `m` - The number of discrete nodes at each time step.
/// * `gamma` - A parameter for the node-spacing algorithm (typically 0.6).
///
/// # Returns
///
/// A `Result` containing a `WtNodesSwaptionResult` struct on success, or a `WtNodesError` on failure.
pub fn generate_swaption_nodes(
    f0: f64,
    g: &Array2<f64>,
    m: usize,
    gamma: f64,
) -> Result<WtNodesSwaptionResult, WtNodesError> {
    // Call the generic node generator. The result's `willow` field will contain the
    // nodes for the absolute change, Y_T.
    let absolute_change_nodes_result = wt_nodes_from_johnson_curve(g, m, gamma)?;

    // Perform the additive mapping to get the final forward rate nodes.
    // F_i^n = F_0 + Y_i^n
    let forward_rate_nodes = f0 + absolute_change_nodes_result.willow;

    Ok(WtNodesSwaptionResult {
        itype: absolute_change_nodes_result.itype,
        forward_rate_nodes,
        h_params: absolute_change_nodes_result.h_params,
    })
}
