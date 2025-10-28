//! Calibrates the Implied Willow Tree to market option prices.
use crate::{
    moments::{implied_moments_normal, MomentsError, MomentsResult},
    nodes::WtNodesError,
    swaption_nodes::{generate_swaption_nodes, WtNodesSwaptionResult},
};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2};
use nlopt::{Algorithm, Nlopt, Target};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, instrument, trace, warn};

#[derive(Error, Debug)]
pub enum IwtBuildError {
    #[error("An error occurred during moment calculation")]
    Moments(#[from] MomentsError),
    #[error("An error occurred during tree calibration")]
    Calibration(#[from] CalibrationError),
}

#[derive(Error, Debug)]
pub enum CalibrationError {
    #[error("An error occurred during the node generation step")]
    NodeGeneration(#[from] WtNodesError),
    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),
    #[error("nlopt operation failed: {0:?}")]
    Nlopt(nlopt::FailState),
}

impl From<nlopt::FailState> for CalibrationError {
    fn from(fail_state: nlopt::FailState) -> Self {
        CalibrationError::Nlopt(fail_state)
    }
}

/// A struct holding the fully calibrated Implied Willow Tree.
#[derive(Debug, Serialize, Deserialize)]
pub struct ImpliedWillowTree {
    /// The initial forward swap rate at time 0.
    pub f0: f64,
    /// The maturities for which the tree was calibrated. (N)
    pub maturities: Array1<f64>,
    /// A matrix containing the forward swap rate values for each node and time step. (M x N)
    pub forward_rate_nodes: Array2<f64>,
    /// A 3D tensor of transition probabilities. `p(k, i, j)` is the probability of
    /// moving from node `i` at time `k` to node `j` at time `k+1`. (N-1 x M x M)
    pub transition_probabilities: Array3<f64>,
    /// A matrix of node probabilities. `q(i, j)` is the probability of being at
    /// node `i` at time `j`. (M x N)
    pub node_probabilities: Array2<f64>,
}

pub struct MarketData<'a> {
    pub f0: f64,
    pub annuity_factors: &'a Array1<f64>,
    pub maturities: &'a Array1<f64>,
    pub strikes: &'a Array1<f64>,
    pub payer_prices: &'a Array2<f64>,
    pub receiver_prices: &'a Array2<f64>,
}

pub fn build_implied_willow_tree(
    f0: f64,
    annuity_factors: &Array1<f64>,
    maturities: &Array1<f64>,
    strikes: &Array1<f64>,
    payer_prices: &Array2<f64>,
    receiver_prices: &Array2<f64>,
    weights: &Array2<f64>,
    m: usize,
    gamma: f64,
    alpha: f64,
) -> Result<ImpliedWillowTree, IwtBuildError> {
    let moments = implied_moments_normal(
        payer_prices,
        receiver_prices,
        strikes,
        f0,
        annuity_factors,
    )?;

    let market_data = MarketData {
        f0,
        annuity_factors,
        maturities,
        strikes,
        payer_prices,
        receiver_prices,
    };

    let mut tree = calibrate_willow_tree_normal(&market_data, weights, m, gamma, &moments, alpha)?;
    tree.maturities = maturities.clone();

    Ok(tree)
}

#[instrument(level = "debug", skip_all)]
pub fn calibrate_willow_tree_normal(
    market_data: &MarketData,
    weights: &Array2<f64>,
    m: usize,
    gamma: f64,
    moments: &MomentsResult,
    alpha: f64,
) -> Result<ImpliedWillowTree, CalibrationError> {
    let n = market_data.annuity_factors.len();

    debug!("Generating willow tree nodes from moments.");
    let g = ndarray::stack(
        ndarray::Axis(0),
        &[
            moments.mean.view(),
            moments.variance.view(),
            moments.skewness.view(),
            moments.kurtosis.view(),
        ],
    )
    .expect("Failed to stack moments");

    let WtNodesSwaptionResult {
        mut forward_rate_nodes,
        ..
    } = generate_swaption_nodes(market_data.f0, &g, m, gamma)?;

    forward_rate_nodes
        .axis_iter_mut(ndarray::Axis(1))
        .for_each(|mut col| {
            // A column of a row-major array is not contiguous, so we can't get a mutable slice.
            // The idiomatic way to sort it is to copy it to a Vec, sort the Vec,
            // and then assign the sorted data back to the column.
            let mut col_vec = col.to_vec();
            col_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
            col.assign(&Array1::from_vec(col_vec));
        });
    trace!(
        ?forward_rate_nodes,
        "Generated and sorted forward rate nodes"
    );

    debug!("Calibrating node probabilities (q).");
    let mut q = Array2::zeros((m, n));
    for n_t in 0..n {
        let q_col = solve_q_optimization(
            forward_rate_nodes.column(n_t),
            market_data.strikes.view(),
            market_data.payer_prices.column(n_t),
            market_data.receiver_prices.column(n_t),
            weights.column(n_t),
            market_data.annuity_factors[n_t],
            market_data.f0,
            alpha,
        )?;
        q.column_mut(n_t).assign(&q_col);
    }
    trace!(?q, "Calibrated node probabilities");

    debug!("Calibrating transition probabilities (p).");
    let mut p = Array3::zeros((n - 1, m, m));
    for n_t in 0..(n - 1) {
        let p_slice = solve_p_optimization(
            q.column(n_t),
            q.column(n_t + 1),
            forward_rate_nodes.column(n_t),
            forward_rate_nodes.column(n_t + 1),
        )?;
        p.slice_mut(ndarray::s![n_t, .., ..]).assign(&p_slice);
    }
    trace!(?p, "Calibrated transition probabilities");

    Ok(ImpliedWillowTree {
        f0: market_data.f0,
        maturities: market_data.maturities.clone(),
        forward_rate_nodes,
        transition_probabilities: p,
        node_probabilities: q,
    })
}

#[derive(Copy, Clone)]
struct QObjectiveData<'a> {
    f_nodes: ArrayView1<'a, f64>,
    strikes: ArrayView1<'a, f64>,
    payer_prices: ArrayView1<'a, f64>,
    receiver_prices: ArrayView1<'a, f64>,
    weights: ArrayView1<'a, f64>,
    annuity: f64,
    f0: f64,
    alpha: f64,
}

fn nlopt_objective_q(x: &[f64], _gradient: Option<&mut [f64]>, data: &mut QObjectiveData) -> f64 {
    let q = ArrayView1::from(x);
    let objective_value = objective_q_normal(
        &q,
        data.f_nodes,
        data.strikes,
        data.payer_prices,
        data.receiver_prices,
        data.weights,
        data.annuity,
        data.alpha,
    );
    trace!(?x, objective_value, "nlopt_objective_q");
    objective_value
}

fn nlopt_mconstraint_q(
    result: &mut [f64],
    x: &[f64],
    _gradient: Option<&mut [f64]>,
    data: &mut QObjectiveData,
) {
    let q = ArrayView1::from(x);
    result[0] = q.sum() - 1.0;
    result[1] = q.dot(&data.f_nodes) - data.f0;
}

#[instrument(level = "debug", skip_all)]
fn solve_q_optimization<'a>(
    f_nodes: ArrayView1<'a, f64>,
    strikes: ArrayView1<'a, f64>,
    payer_prices: ArrayView1<'a, f64>,
    receiver_prices: ArrayView1<'a, f64>,
    weights: ArrayView1<'a, f64>,
    annuity: f64,
    f0: f64,
    alpha: f64,
) -> Result<Array1<f64>, CalibrationError> {
    let m = f_nodes.len();
    let data = QObjectiveData {
        f_nodes,
        strikes,
        payer_prices,
        receiver_prices,
        weights,
        annuity,
        f0,
        alpha,
    };

    let mut optimizer = Nlopt::new(
        Algorithm::Slsqp,
        m,
        nlopt_objective_q,
        Target::Minimize,
        data,
    );
    optimizer.set_lower_bound(0.0)?;
    optimizer.set_upper_bound(1.0)?;
    optimizer.set_ftol_rel(1e-8)?;

    optimizer.add_equality_mconstraint(2, nlopt_mconstraint_q, data, &[1e-8, 1e-8])?;

    let mut x_init = vec![1.0 / m as f64; m];
    let result = optimizer.optimize(&mut x_init);

    match result {
        Ok((_, final_val)) => {
            debug!(objective_value = final_val, "Q optimization successful.");
            Ok(Array1::from_vec(x_init))
        }
        Err((fail_state, final_val)) => {
            match fail_state {
                nlopt::FailState::RoundoffLimited => {
                    warn!(
                        ?fail_state,
                        objective_value = final_val,
                        "Q optimization finished with RoundoffLimited, accepting result."
                    );
                    Ok(Array1::from_vec(x_init))
                }
                _ => {
                    warn!(
                        ?fail_state,
                        objective_value = final_val,
                        "Q optimization failed."
                    );
                    Err(CalibrationError::from(fail_state))
                }
            }
        }
    }
}

fn objective_q_normal(
    q: &ArrayView1<f64>,
    f_nodes: ArrayView1<f64>,
    strikes: ArrayView1<f64>,
    payer_prices: ArrayView1<f64>,
    receiver_prices: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    annuity: f64,
    alpha: f64,
) -> f64 {
    let mut total_weighted_error = 0.0;
    let total_weight = weights.sum();

    if total_weight.abs() < 1e-12 {
        return 0.0; // Avoid division by zero if all weights are zero
    }

    for i in 0..strikes.len() {
        let k = strikes[i];
        let market_payer = payer_prices[i];
        let market_receiver = receiver_prices[i];
        let weight = weights[i];

        // Calculate model prices
        let model_payer_payoff: f64 = q
            .iter()
            .zip(f_nodes.iter())
            .map(|(&prob, &f)| prob * (f - k).max(0.0))
            .sum();
        let model_receiver_payoff: f64 = q
            .iter()
            .zip(f_nodes.iter())
            .map(|(&prob, &f)| prob * (k - f).max(0.0))
            .sum();

        let model_payer = model_payer_payoff * annuity;
        let model_receiver = model_receiver_payoff * annuity;

        let error =
            (model_payer - market_payer).powi(2) + (model_receiver - market_receiver).powi(2);
        total_weighted_error += weight * error;
    }

    // Add smoothness regularization term
    let mut smoothness_term = 0.0;
    if q.len() > 2 {
        for i in 1..q.len() - 1 {
            smoothness_term += (q[i - 1] - 2.0 * q[i] + q[i + 1]).powi(2);
        }
    }

    let objective_value = (total_weighted_error / total_weight) + alpha * smoothness_term;

    // Scale the objective function to give the optimizer more to work with,
    // forcing it to satisfy constraints more tightly.
    objective_value * 1e6
}

#[derive(Copy, Clone)]
struct PObjectiveData<'a> {
    q_t: ArrayView1<'a, f64>,
    q_t1: ArrayView1<'a, f64>,
    f_nodes_t: ArrayView1<'a, f64>,
    f_nodes_t1: ArrayView1<'a, f64>,
}

fn nlopt_objective_p(x: &[f64], _gradient: Option<&mut [f64]>, _data: &mut PObjectiveData) -> f64 {
    // The objective is to find the "smoothest" transition matrix by minimizing the sum of squares
    // of its elements, as described in the project's technical documentation.
    x.iter().map(|v| v.powi(2)).sum()
}

fn nlopt_mconstraint_p(
    result: &mut [f64],
    x: &[f64],
    _gradient: Option<&mut [f64]>,
    data: &mut PObjectiveData,
) {
    let m = data.q_t.len();
    let p_matrix = ArrayView2::from_shape((m, m), x).expect("Failed to reshape p vector");

    // Propagated probabilities: q_t1_propagated = q_t . p_matrix
    let q_propagated = data.q_t.dot(&p_matrix);

    for i in 0..m {
        // Constraint 1: Rows of p must sum to 1
        result[i] = p_matrix.row(i).sum() - 1.0;
        // Constraint 2: Local martingale property
        result[m + i] = p_matrix.row(i).dot(&data.f_nodes_t1) - data.f_nodes_t[i];
        // Constraint 3: Chapman-Kolmogorov forward equation
        result[2 * m + i] = q_propagated[i] - data.q_t1[i];
    }
}

#[instrument(level = "debug", skip_all)]
fn solve_p_optimization<'a>(
    q_t: ArrayView1<'a, f64>,
    q_t1: ArrayView1<'a, f64>,
    f_nodes_t: ArrayView1<'a, f64>,
    f_nodes_t1: ArrayView1<'a, f64>,
) -> Result<Array2<f64>, CalibrationError> {
    let m = q_t.len();
    let data = PObjectiveData {
        q_t,
        q_t1,
        f_nodes_t,
        f_nodes_t1,
    };

    let mut optimizer = Nlopt::new(
        Algorithm::Slsqp,
        m * m,
        nlopt_objective_p,
        Target::Minimize,
        data,
    );

    optimizer.set_lower_bound(0.0)?;
    optimizer.set_upper_bound(1.0)?;
    optimizer.set_ftol_rel(1e-8)?;

    // There are now 3*m constraints
    let num_constraints = 3 * m;
    let tolerances = vec![1e-8; num_constraints];
    optimizer.add_equality_mconstraint(num_constraints, nlopt_mconstraint_p, data, &tolerances)?;

    let mut x_init = vec![1.0 / m as f64; m * m];
    let result = optimizer.optimize(&mut x_init);

            match result {

                Ok((_, final_val)) => {

                    debug!(objective_value = final_val, "P optimization successful.");

                    let p_matrix = Array2::from_shape_vec((m, m), x_init)

                        .expect("Failed to reshape final p vector");

                    Ok(p_matrix)

                }

                Err((fail_state, final_val)) => {

                    match fail_state {

                        nlopt::FailState::RoundoffLimited => {

                            warn!(

                                ?fail_state,

                                objective_value = final_val,

                                "P optimization finished with RoundoffLimited, accepting result."

                            );

                            let p_matrix = Array2::from_shape_vec((m, m), x_init)

                                .expect("Failed to reshape final p vector");

                            Ok(p_matrix)

                        }

                        _ => {

                            warn!(

                                ?fail_state,

                                objective_value = final_val,

                                "P optimization failed."

                            );

                            Err(CalibrationError::from(fail_state))

                        }

                    }

                }

            }
}

fn objective_p(p_matrix: &ArrayView2<f64>, q_t: &ArrayView1<f64>, q_t1: &ArrayView1<f64>) -> f64 {
    let q_propagated = q_t.dot(p_matrix);
    let error = &q_propagated - q_t1;
    error.mapv(|x| x.powi(2)).sum()
}
