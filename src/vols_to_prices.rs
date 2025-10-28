//! A module to convert swaption normal volatilities into option prices.
//!
//! This module uses the `implied-vol` crate to perform the conversion from
//! the Bachelier (Normal) model volatility quotes common in interest rate markets
//! to the currency-denominated prices required by the IWT algorithm.

use implied_vol::{DefaultSpecialFn, ImpliedNormalVolatility, PriceBachelier};
use ndarray::{Array1, Array2};
use thiserror::Error;
use tracing::trace;

#[derive(Error, Debug, PartialEq)]
pub enum VolConversionError {
    #[error("Input dimension mismatch: {0}")]
    DimensionMismatch(String),
    #[error("The implied-vol builder failed for parameters: {0}")]
    BuilderFailed(String),
    #[error(
        "Implied volatility calculation failed for price {price}, forward {forward}, strike {strike}, expiry {expiry}"
    )]
    ImpliedVolFailed {
        price: f64,
        forward: f64,
        strike: f64,
        expiry: f64,
    },
}

/// Holds the converted payer and receiver swaption prices.
///
/// In the context of the IWT model:
/// - Payer swaptions are equivalent to call options.
/// - Receiver swaptions are equivalent to put options.
pub struct SwaptionPrices {
    pub payer_prices: Array2<f64>,
    pub receiver_prices: Array2<f64>,
}

/// Holds the calculated payer and receiver swaption volatilities for a single strike.
pub struct SwaptionVols {
    pub payer_vol: f64,
    pub receiver_vol: f64,
}

/// Converts a matrix of normal volatilities into matrices of payer and receiver swaption prices.
///
/// # Arguments
///
/// * `normal_vols` - A 2D array (`num_strikes x num_maturities`) of normal (Bachelier) volatilities.
/// * `strikes` - A 1D array (`num_strikes`) of strike rates.
/// * `maturities` - A 1D array (`num_maturities`) of expiries in years.
/// * `f0` - The current forward swap rate.
/// * `annuity_factors` - A 1D array (`num_maturities`) of the annuity factor for each maturity.
///
/// # Returns
///
/// A `Result` containing a `SwaptionPrices` struct on success, or a `VolConversionError` on failure.
pub fn convert_vols_to_prices(
    normal_vols: &Array2<f64>,
    strikes: &Array1<f64>,
    maturities: &Array1<f64>,
    f0: f64,
    annuity_factors: &Array1<f64>,
) -> Result<SwaptionPrices, VolConversionError> {
    let (num_strikes, num_maturities) = normal_vols.dim();
    if strikes.len() != num_strikes {
        return Err(VolConversionError::DimensionMismatch(format!(
            "Strikes length ({}) does not match vols dimension ({})",
            strikes.len(),
            num_strikes
        )));
    }
    if maturities.len() != num_maturities || annuity_factors.len() != num_maturities {
        return Err(VolConversionError::DimensionMismatch(format!(
            "Maturities ({}) or annuities ({}) length does not match vols dimension ({})",
            maturities.len(),
            annuity_factors.len(),
            num_maturities
        )));
    }

    let mut payer_prices = Array2::zeros((num_strikes, num_maturities));
    let mut receiver_prices = Array2::zeros((num_strikes, num_maturities));

    for (j, &maturity) in maturities.iter().enumerate() {
        let annuity = annuity_factors[j];
        for (i, &strike) in strikes.iter().enumerate() {
            let vol = normal_vols[[i, j]];

            // Payer swaption (Call)
            let payer_price_builder = PriceBachelier::builder()
                .forward(f0)
                .strike(strike)
                .volatility(vol)
                .expiry(maturity)
                .is_call(true)
                .build();

            if let Some(builder) = payer_price_builder {
                let undiscounted_price = builder.calculate::<DefaultSpecialFn>();
                payer_prices[[i, j]] = undiscounted_price * annuity;
            } else {
                return Err(VolConversionError::BuilderFailed(format!(
                    "Payer vol={vol}, strike={strike}, maturity={maturity}"
                )));
            }

            // Receiver swaption (Put)
            let receiver_price_builder = PriceBachelier::builder()
                .forward(f0)
                .strike(strike)
                .volatility(vol)
                .expiry(maturity)
                .is_call(false)
                .build();

            if let Some(builder) = receiver_price_builder {
                let undiscounted_price = builder.calculate::<DefaultSpecialFn>();
                receiver_prices[[i, j]] = undiscounted_price * annuity;
            } else {
                return Err(VolConversionError::BuilderFailed(format!(
                    "Receiver vol={vol}, strike={strike}, maturity={maturity}"
                )));
            }
        }
    }

    Ok(SwaptionPrices {
        payer_prices,
        receiver_prices,
    })
}

/// Converts vectors of payer and receiver swaption prices into normal volatilities.
pub fn convert_prices_to_vols(
    payer_price: f64,
    receiver_price: f64,
    strike: f64,
    maturity: f64,
    f0: f64,
    annuity: f64,
) -> Result<SwaptionVols, VolConversionError> {
    // Payer (Call)
    let undiscounted_payer_price = payer_price / annuity + 1e-9; // Add epsilon for stability
    trace!(
        price = undiscounted_payer_price,
        forward = f0,
        strike,
        expiry = maturity,
        is_call = true,
        "Payer IV inputs"
    );
    let iv_payer_builder = ImpliedNormalVolatility::builder()
        .option_price(undiscounted_payer_price)
        .forward(f0)
        .strike(strike)
        .expiry(maturity)
        .is_call(true)
        .build();

    let payer_vol = if let Some(builder) = iv_payer_builder {
        builder
            .calculate::<DefaultSpecialFn>()
            .ok_or_else(|| VolConversionError::ImpliedVolFailed {
                price: undiscounted_payer_price,
                forward: f0,
                strike,
                expiry: maturity,
            })?
    } else {
        return Err(VolConversionError::BuilderFailed(
            "Payer IV builder failed".to_string(),
        ));
    };

    // Receiver (Put)
    let undiscounted_receiver_price = receiver_price / annuity + 1e-9; // Add epsilon for stability
    trace!(
        price = undiscounted_receiver_price,
        forward = f0,
        strike,
        expiry = maturity,
        is_call = false,
        "Receiver IV inputs"
    );
    let iv_receiver_builder = ImpliedNormalVolatility::builder()
        .option_price(undiscounted_receiver_price)
        .forward(f0)
        .strike(strike)
        .expiry(maturity)
        .is_call(false)
        .build();

    let receiver_vol = if let Some(builder) = iv_receiver_builder {
        builder
            .calculate::<DefaultSpecialFn>()
            .ok_or_else(|| VolConversionError::ImpliedVolFailed {
                price: undiscounted_receiver_price,
                forward: f0,
                strike,
                expiry: maturity,
            })?
    } else {
        return Err(VolConversionError::BuilderFailed(
            "Receiver IV builder failed".to_string(),
        ));
    };

    Ok(SwaptionVols {
        payer_vol,
        receiver_vol,
    })
}
