//! A new module for generating the nodes of the Willow Tree.

use crate::f_hhh::{HhhError, JohnsonDistribution, hhh};
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use thiserror::Error;

#[derive(Error, Debug, PartialEq)]
pub enum WtNodesError {
    #[error("An error occurred during the Johnson distribution fitting")]
    Hhh(#[from] HhhError),
    #[error("Statrs Normal distribution creation failed: {0}")]
    Statrs(String),
}

/// A struct to hold the results of the node generation process,
/// mirroring the multiple return values of the original MATLAB function.
#[derive(Debug)]
pub struct WtNodesResult {
    /// A vector of integers representing the Johnson distribution type for each time step.
    /// (1: SL, 2: SU, 3: SB, 4: Normal)
    pub itype: Vec<i32>,
    /// A matrix containing the generated values (e.g., log-returns or absolute changes) for each
    /// node and time step. (M x N)
    pub willow: Array2<f64>,
    /// A matrix containing the fitted Johnson parameters (a, b, c, d) for each time step. (4 x N)
    pub h_params: Array2<f64>,
}

/// Generates the nodes of a willow tree by fitting moments to the Johnson distribution system.
///
/// This function is a generic implementation of the MATLAB `WTnodes_from_JohnsonCurve.m` script.
/// It takes the first four moments of any target distribution and produces a discrete set of
/// node values that match those moments at each time step.
///
/// # Arguments
///
/// * `g` - A 2D array (`4 x N`) containing the first four moments (mean, variance, skew, kurtosis)
///   of the target distribution for each of the `N` time steps.
/// * `m` - The number of discrete nodes at each time step.
/// * `gamma` - A parameter for the node-spacing algorithm (typically 0.6).
///
/// # Returns
///
/// A `Result` containing a `WtNodesResult` struct on success, or a `WtNodesError` on failure.
pub fn wt_nodes_from_johnson_curve(
    g: &Array2<f64>,
    m: usize,
    gamma: f64,
) -> Result<WtNodesResult, WtNodesError> {
    let n = g.shape()[1];

    // --- Compute z and q ---
    let mut q = vec![0.0; m];
    // Create a symmetric vector of probabilities `q`
    for k in 0..=(m / 2) {
        let val = ((k as f64 + 0.5).powf(gamma)) / m as f64;
        q[k] = val;
        if k < m {
            q[m - 1 - k] = val;
        }
    }

    let q_sum: f64 = q.iter().sum();
    q.iter_mut().for_each(|x| *x /= q_sum);

    let mut z = Array1::zeros(m);
    let normal = Normal::new(0.0, 1.0).map_err(|e| WtNodesError::Statrs(e.to_string()))?;

    // Calculate the standard normal quantiles `z` from `q`
    z[0] = normal.inverse_cdf(q[0] / 2.0);
    let mut cumulative_q = 0.0;
    for k in 1..m {
        cumulative_q += q[k - 1];
        let tmp = cumulative_q + q[k] / 2.0;
        z[k] = normal.inverse_cdf(tmp);
    }

    // --- Main Loop: Call f_hhh and transform z ---
    let mut itype_vec = Vec::with_capacity(n);
    let mut willow = Array2::zeros((m, n));
    let mut h_params = Array2::zeros((4, n));

    for i in 0..n {
        let mu = g[[0, i]];
        let var = g[[1, i]];
        let sd = var.sqrt();
        let ka3 = g[[2, i]];
        let ka4 = g[[3, i]];

        let result = hhh(mu, sd, ka3, ka4)?;

        let (itype, a, b, c, d);
        let x = match result {
            JohnsonDistribution::Lognormal {
                gamma,
                delta,
                lambda,
                xi,
            } => {
                itype = 1;
                a = gamma;
                b = delta;
                c = xi;
                d = lambda;
                let u = (&z - a) / b;
                c + d * u.mapv(f64::exp)
            }
            JohnsonDistribution::Su(p) => {
                itype = 2;
                a = p.gamma;
                b = p.delta;
                c = p.xi;
                d = p.lambda;
                let u = (&z - a) / b;
                c + d * u.mapv(f64::sinh)
            }
            JohnsonDistribution::Sb(p) => {
                itype = 3;
                a = p.gamma;
                b = p.delta;
                c = p.xi;
                d = p.lambda;
                let u = (&z - a) / b;
                c + d * u.mapv(|v| 1.0 / (1.0 + (-v).exp()))
            }
            JohnsonDistribution::Normal {
                gamma,
                delta,
                lambda,
            } => {
                itype = 4;
                a = gamma;
                b = delta;
                c = 0.0; // Inferred from MATLAB, c/d are not returned for Normal
                d = lambda;
                let u = (&z - a) / b;
                c + d * u
            }
            // The C code does not return parameters for these, so we cannot proceed.
            // This indicates a modeling choice mismatch that should be an error.
            _ => {
                return Err(WtNodesError::Hhh(HhhError::ImpossibleMoments(
                    "ST or Constant distribution returned, which is not handled by the node transformation logic.".to_string(),
                )));
            }
        };

        itype_vec.push(itype);
        h_params
            .column_mut(i)
            .assign(&Array1::from(vec![a, b, c, d]));
        willow.column_mut(i).assign(&x);
    }

    Ok(WtNodesResult {
        itype: itype_vec,
        willow,
        h_params,
    })
}
