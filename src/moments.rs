//! Calculates the risk-neutral moments of an underlying asset's returns.
//!
//! This module provides functions to derive the first four statistical moments (mean, variance,
//! skewness, and kurtosis) implied by market option prices. It contains implementations for both
//! the lognormal model (used for equities, based on the provided MATLAB code) and the normal
//! model (required for interest rate swaptions).

use ndarray::{Array1, Array2};
use thiserror::Error;
use tracing::{debug, instrument};

#[derive(Error, Debug, PartialEq)]
pub enum MomentsError {
    #[error("Input dimension mismatch: {0}")]
    DimensionMismatch(String),
}


/// A struct to hold the calculated moments for each maturity.
#[derive(Debug, PartialEq)]
pub struct MomentsResult {
    pub mean: Array1<f64>,
    pub variance: Array1<f64>,
    pub skewness: Array1<f64>,
    pub kurtosis: Array1<f64>,
}

/// Calculates implied moments for a **normal (Bachelier) model**, suitable for swaptions.
///
/// This function implements the moment formulas from the Carr-Madan spanning formula, adapted
/// for the forward swap measure where the forward rate is a true martingale. This is the
/// primary function to be used for the swaption IWT model.
///
/// The formula is:
/// μ_k = ∫(-∞ to F₀) k(k-1)(K-F₀)^(k-2) V̂_receiver(K)dK + ∫(F₀ to ∞) k(k-1)(K-F₀)^(k-2) V̂_payer(K)dK
///
/// # Arguments
///
/// * `payer_prices` - A 2D array (`num_strikes x num_maturities`) of payer swaption prices (calls).
/// * `receiver_prices` - A 2D array (`num_strikes x num_maturities`) of receiver swaption prices (puts).
/// * `strikes` - A 1D array (`num_strikes`) of strike rates.
/// * `f0` - The current forward swap rate.
/// * `annuity_factors` - A 1D array (`num_maturities`) of the annuity factor for each maturity.
/// * `dk` - The step size between strikes, used for numerical integration.
///
/// # Returns
///
/// A `Result` containing the calculated central moments in a `MomentsResult` struct.
#[instrument(level = "debug", skip(payer_prices, receiver_prices, strikes, annuity_factors))]
pub fn implied_moments_normal(
    payer_prices: &Array2<f64>,
    receiver_prices: &Array2<f64>,
    strikes: &Array1<f64>,
    f0: f64,
    annuity_factors: &Array1<f64>,
) -> Result<MomentsResult, MomentsError> {
    let (num_strikes, num_maturities) = payer_prices.dim();
    // --- Input Validation ---
    if receiver_prices.dim() != (num_strikes, num_maturities) {
        return Err(MomentsError::DimensionMismatch(
            "Payer and receiver price matrix dimensions do not match.".to_string(),
        ));
    }
    if strikes.len() != num_strikes {
        return Err(MomentsError::DimensionMismatch(format!(
            "Strikes length ({}) does not match price matrix dimension ({})",
            strikes.len(),
            num_strikes
        )));
    }
    if annuity_factors.len() != num_maturities {
        return Err(MomentsError::DimensionMismatch(format!(
            "Annuity factors length ({}) does not match price matrix dimension ({})",
            annuity_factors.len(),
            num_maturities
        )));
    }

    let mut variance_vec = Vec::with_capacity(num_maturities);
    let mut skewness_vec = Vec::with_capacity(num_maturities);
    let mut kurtosis_vec = Vec::with_capacity(num_maturities);

    for n_t in 0..num_maturities {
        let annuity = annuity_factors[n_t];
        let payer_prices_t = payer_prices.column(n_t);
        let receiver_prices_t = receiver_prices.column(n_t);

        // Normalize prices by the annuity factor to get Q_swap
        let v_hat_payer = &payer_prices_t / annuity;
        let v_hat_receiver = &receiver_prices_t / annuity;

        let mut mu2 = 0.0; // Raw 2nd moment (Variance)
        let mut mu3 = 0.0; // Raw 3rd moment
        let mut mu4 = 0.0; // Raw 4th moment

        // Riemann Sum Integration, as specified by the formula document.
        // This calculates Σ ΔK_i * g_n''(K_i) * Q_swap(K_i)
        for i in 0..num_strikes - 1 {
            let k = strikes[i];
            let dk = strikes[i + 1] - strikes[i];

            // Select the Out-of-The-Money (OTM) annuity-normalized price (Q_swap)
            let q_swap = if k < f0 {
                v_hat_receiver[i]
            } else {
                v_hat_payer[i]
            };

            // g_2''(K) = 2
            mu2 += 2.0 * q_swap * dk;
            // g_3''(K) = 6 * (K - f0)
            mu3 += 6.0 * (k - f0) * q_swap * dk;
            // g_4''(K) = 12 * (K - f0)^2
            mu4 += 12.0 * (k - f0).powi(2) * q_swap * dk;
        }

        let variance = mu2;
        let skewness = if variance > 1e-12 {
            mu3 / variance.powf(1.5)
        } else {
            0.0
        };
        let kurtosis = if variance > 1e-12 {
            mu4 / variance.powi(2)
        } else {
            // Kurtosis of a normal distribution is 3.0. The hhh function expects full kurtosis.
            3.0
        };

        debug!(
            maturity_idx = n_t,
            variance,
            skewness,
            kurtosis,
            "Calculated moments for maturity"
        );
        variance_vec.push(variance);
        skewness_vec.push(skewness);
        kurtosis_vec.push(kurtosis);
    }

    Ok(MomentsResult {
        // The mean of (F_T - F_0) is always 0 under the forward measure.
        mean: Array1::zeros(num_maturities),
        variance: Array1::from_vec(variance_vec),
        skewness: Array1::from_vec(skewness_vec),
        kurtosis: Array1::from_vec(kurtosis_vec),
    })
}

/// Calculates implied moments for a **lognormal (Black-Scholes) model**, suitable for equities.
///
/// This is a direct Rust conversion of the `Imp_Moments_underQ.m` MATLAB script.
/// It computes moments of the log-return `ln(S_T/S_0)`.
///
/// # Arguments
///
/// * `call_mkt` - A 2D array (`num_strikes x num_maturities`) of call option prices.
/// * `put_mkt` - A 2D array (`num_strikes x num_maturities`) of put option prices.
/// * `s0` - The initial stock price.
/// * `b0` - A 1D array (`num_maturities`) of discount factors.
/// * `k` - A 1D array (`num_strikes`) of strike prices.
/// * `dk` - The step size between strikes.
///
/// # Returns
///
/// A `Result` containing the calculated central moments in a `MomentsResult` struct.
#[instrument(level = "debug", skip(call_mkt, put_mkt, b0, k))]
pub fn implied_moments_lognormal(
    call_mkt: &Array2<f64>,
    put_mkt: &Array2<f64>,
    s0: f64,
    b0: &Array1<f64>,
    k: &Array1<f64>,
    dk: f64,
) -> Result<MomentsResult, MomentsError> {
    let (num_strikes, num_maturities) = call_mkt.dim();
    // --- Input Validation ---
    if put_mkt.dim() != (num_strikes, num_maturities) {
        return Err(MomentsError::DimensionMismatch(
            "Call and put price matrix dimensions do not match.".to_string(),
        ));
    }
    if k.len() != num_strikes {
        return Err(MomentsError::DimensionMismatch(format!(
            "Strikes length ({}) does not match price matrix dimension ({})",
            k.len(),
            num_strikes
        )));
    }
    if b0.len() != num_maturities {
        return Err(MomentsError::DimensionMismatch(format!(
            "Discount factors length ({}) does not match price matrix dimension ({})",
            b0.len(),
            num_maturities
        )));
    }

    let mut mean_vec = Vec::with_capacity(num_maturities);
    let mut var_vec = Vec::with_capacity(num_maturities);
    let mut skew_vec = Vec::with_capacity(num_maturities);
    let mut kurt_vec = Vec::with_capacity(num_maturities);

    for n_t in 0..num_maturities {
        let (mean, var, skew, kurt) = implied_moments_lognormal_one_maturity(
            call_mkt.column(n_t),
            put_mkt.column(n_t),
            s0,
            b0[n_t],
            k,
            dk,
        );
        debug!(
            maturity_idx = n_t,
            mean,
            variance = var,
            skewness = skew,
            kurtosis = kurt,
            "Calculated moments for maturity"
        );
        mean_vec.push(mean);
        var_vec.push(var);
        skew_vec.push(skew);
        kurt_vec.push(kurt);
    }

    Ok(MomentsResult {
        mean: Array1::from_vec(mean_vec),
        variance: Array1::from_vec(var_vec),
        skewness: Array1::from_vec(skew_vec),
        kurtosis: Array1::from_vec(kurt_vec),
    })
}

/// Helper function to calculate lognormal moments for a single maturity.
#[instrument(level = "trace", skip(call, put, k))]
fn implied_moments_lognormal_one_maturity(
    call: ndarray::ArrayView1<f64>,
    put: ndarray::ArrayView1<f64>,
    s0: f64,
    b0_t: f64,
    k: &Array1<f64>,
    dk: f64,
) -> (f64, f64, f64, f64) {
    let k0 = s0;
    let out_of_money: Array1<f64> = k
        .iter()
        .zip(put.iter())
        .zip(call.iter())
        .map(|((&strike, &p), &c)| if strike < k0 { p } else { c })
        .collect();

    let k_sq = k.mapv(|v| v.powi(2));
    let log_k_s0 = k.mapv(|v| (v / s0).ln());
    let log_k_k0 = k.mapv(|v| (v / k0).ln());
    let log_k0_s0 = (k0 / s0).ln();

    // --- Calculate Raw Moments (k1, k2, k3, k4) ---
    let k1 = (log_k0_s0 + s0 / (b0_t * k0) - 1.0) - (dk / (&k_sq * b0_t) * &out_of_money).sum();

    let k2 = log_k0_s0.powi(2) + 2.0 * log_k0_s0 / k0 * (s0 / b0_t - k0)
        + (dk * 2.0 / &k_sq * (1.0 - &log_k_s0) / b0_t * &out_of_money).sum();

    let k3 = log_k0_s0.powi(3) + 3.0 * log_k0_s0.powi(2) / k0 * (s0 / b0_t - k0)
        + (dk * 3.0 / &k_sq
            * (&log_k_k0 * (2.0 - &log_k_k0) + 2.0 * log_k0_s0 * (1.0 - &log_k_s0)
                - log_k0_s0.powi(2))
            / b0_t
            * &out_of_money)
            .sum();

    let k4 = log_k0_s0.powi(4) + 4.0 * log_k0_s0.powi(3) / k0 * (s0 / b0_t - k0)
        + (dk * 4.0 / &k_sq
            * (log_k_k0.mapv(|v| v.powi(2)) * (3.0 - &log_k_k0)
                + 3.0 * log_k0_s0 * &log_k_k0 * (2.0 - &log_k_k0)
                + 3.0 * log_k0_s0.powi(2) * (1.0 - &log_k_s0)
                - log_k0_s0.powi(3))
            / b0_t
            * &out_of_money)
            .sum();

    // --- Convert to Central Moments ---
    let mean_dd = k1;
    let var_dd = k2 - k1.powi(2);
    let skew_dd = (k3 - 3.0 * k1 * var_dd - k1.powi(3)) / var_dd.powf(1.5);
    let kurt_dd = (k4 - 4.0 * k3 * k1 + 6.0 * k2 * k1.powi(2) - 3.0 * k1.powi(4)) / var_dd.powi(2);

    (mean_dd, var_dd, skew_dd, kurt_dd)
}


