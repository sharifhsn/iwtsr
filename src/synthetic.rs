//! A framework for generating synthetic swaption market data for testing.
//!
//! This module provides structures and functions to create a controlled testing environment
//! based on the Bachelier (normal) model. It allows for the generation of "perfect"
//! market prices from a known volatility surface, which can then be fed into the IWT
//! pipeline to validate its components.

use crate::{
    moments::{MomentsError, MomentsResult},
    vols_to_prices::{SwaptionPrices, VolConversionError, convert_vols_to_prices},
};
use ndarray::{Array1, Array2, Zip};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SyntheticDataError {
    #[error("An error occurred during volatility-to-price conversion")]
    VolConversion(#[from] VolConversionError),
    #[error("An error occurred during moment calculation")]
    Moments(#[from] MomentsError),
    #[error("Could not find an at-the-money strike to determine theoretical variance")]
    AtmStrikeNotFound,
}

/// Represents a synthetic market governed by the Bachelier (normal) model.
///
/// This struct holds all the necessary parameters to generate a full set of
/// swaption prices based on a known volatility surface.
pub struct BachelierMarket {
    pub f0: f64,
    pub normal_vols: Array2<f64>,
    pub strikes: Array1<f64>,
    pub maturities: Array1<f64>,
    pub annuity_factors: Array1<f64>,
    pub dk: f64,
}

impl BachelierMarket {
    /// Generates payer and receiver swaption prices from the market's volatility surface.
    ///
    /// This is the "forward pass" of the testing framework, creating the ideal market data.
    pub fn generate_prices(&self) -> Result<SwaptionPrices, VolConversionError> {
        convert_vols_to_prices(
            &self.normal_vols,
            &self.strikes,
            &self.maturities,
            self.f0,
            &self.annuity_factors,
        )
    }

    /// Calculates the true, theoretical moments of the Bachelier process.
    ///
    /// This is used as the "ground truth" to compare against the moments calculated
    /// by the IWT `moments` module.
    ///
    /// Note: For a non-flat volatility surface, this function uses the at-the-money
    /// volatility for each maturity to calculate the theoretical variance.
    pub fn theoretical_moments(&self) -> Result<MomentsResult, SyntheticDataError> {
        let num_maturities = self.maturities.len();

        // Find the index of the strike closest to the forward rate f0.
        let atm_strike_index = self
            .strikes
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                (*a - self.f0)
                    .abs()
                    .partial_cmp(&(*b - self.f0).abs())
                    .unwrap()
            })
            .map(|(i, _)| i)
            .ok_or(SyntheticDataError::AtmStrikeNotFound)?;

        let atm_vols = self.normal_vols.row(atm_strike_index);

        let variance = Zip::from(&self.maturities)
            .and(&atm_vols)
            .map_collect(|&t, &vol| vol.powi(2) * t);

        Ok(MomentsResult {
            mean: Array1::zeros(num_maturities),
            variance,
            skewness: Array1::zeros(num_maturities),
            kurtosis: Array1::from_elem(num_maturities, 3.0),
        })
    }
}


