//! Prices derivatives using a calibrated Implied Willow Tree.
//!
//! This module implements the backward induction algorithm to price European-style options
//! (in this case, swaptions) on a pre-calibrated willow tree.

use crate::calibration::ImpliedWillowTree;
use ndarray::Array1;
use thiserror::Error;
use tracing::{debug, instrument, trace};

#[derive(Error, Debug, PartialEq)]
pub enum PricingError {
    #[error("Maturity {0} not found in the calibrated tree.")]
    MaturityNotFound(f64),
}

/// A struct to hold the calculated prices for a single payer and receiver swaption.
#[derive(Debug, PartialEq)]
pub struct SwaptionPrice {
    /// The price for a payer swaption (call).
    pub payer_price: f64,
    /// The price for a receiver swaption (put).
    pub receiver_price: f64,
}

/// Prices a single European payer and receiver swaption for a specific maturity and strike.
#[instrument(level = "debug", skip(tree))]
pub fn price_european_swaption(
    tree: &ImpliedWillowTree,
    strike: f64,
    maturity: f64,
    annuity: f64,
) -> Result<SwaptionPrice, PricingError> {
    trace!(
        requested_maturity = maturity,
        tree_maturities = ?tree.maturities,
        "Searching for maturity in tree"
    );

    // Find the index of the maturity closest to the requested one.
    let closest = tree
        .maturities
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (*a - maturity)
                .abs()
                .partial_cmp(&(*(b) - maturity).abs())
                .unwrap()
        });

    trace!(?closest, "Found closest maturity in tree");

    match closest {
        Some((idx, m)) if (m - maturity).abs() < 1e-9 => {
            trace!(
                found_maturity = m,
                index = idx,
                "Closest maturity is within tolerance"
            );
            let maturity_idx = idx;
            let (m_nodes, _) = tree.forward_rate_nodes.dim();

            trace!(strike, "Pricing for strike");
            let mut v_payer = Array1::zeros(m_nodes);
            let mut v_receiver = Array1::zeros(m_nodes);

            // 1. Set the payoff at the specified maturity
            let final_nodes = tree.forward_rate_nodes.column(maturity_idx);
            v_payer.assign(&final_nodes.mapv(|f| (f - strike).max(0.0)));
            v_receiver.assign(&final_nodes.mapv(|f| (strike - f).max(0.0)));

            // 2. Perform backward induction from the maturity index
            for t in (0..maturity_idx).rev() {
                trace!(time_step = t, ?v_payer, ?v_receiver, "Before induction step");
                let p_t = tree.transition_probabilities.slice(ndarray::s![t, .., ..]);
                v_payer = p_t.dot(&v_payer);
                v_receiver = p_t.dot(&v_receiver);
                trace!(time_step = t, ?v_payer, ?v_receiver, "After induction step");
            }

            // 3. Calculate the final price at t=0
            let q_t1 = tree.node_probabilities.column(0);
            let expected_payer_value_t1 = q_t1.dot(&v_payer);
            let expected_receiver_value_t1 = q_t1.dot(&v_receiver);

            let final_payer_price = annuity * expected_payer_value_t1;
            let final_receiver_price = annuity * expected_receiver_value_t1;

            debug!(
                strike,
                payer_price = final_payer_price,
                receiver_price = final_receiver_price,
                "Calculated final prices for strike"
            );

            Ok(SwaptionPrice {
                payer_price: final_payer_price,
                receiver_price: final_receiver_price,
            })
        }
        _ => {
            trace!("No maturity found within tolerance");
            Err(PricingError::MaturityNotFound(maturity))
        }
    }
}
