//! A module for loading swaption data from CSV files and building an Implied Willow Tree.

use crate::{
    calibration::{build_implied_willow_tree, ImpliedWillowTree, IwtBuildError},
    vols_to_prices::{convert_vols_to_prices, VolConversionError},
};
use ndarray::{Array1, Array2};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DataLoaderError {
    #[error("CSV reading error")]
    Csv(#[from] csv::Error),
    #[error("File not found: {0}")]
    Io(#[from] std::io::Error),
    #[error("Failed to parse float from string: {0}")]
    ParseFloat(#[from] std::num::ParseFloatError),
    #[error("CSV format error: {0}")]
    Format(String),
    #[error("Inconsistent annuity factor for maturity {maturity}: found {annuity1} and {annuity2}")]
    InconsistentAnnuity {
        maturity: f64,
        annuity1: f64,
        annuity2: f64,
    },
    #[error("Volatility to price conversion failed")]
    VolConversion(#[from] VolConversionError),
    #[error("Implied Willow Tree construction failed")]
    IwtBuild(#[from] IwtBuildError),
    #[error("JSON serialization/deserialization error")]
    Json(#[from] serde_json::Error),
}

/// Represents a single row in the swaption quotes CSV file.
#[derive(Debug, Deserialize)]
struct QuoteRecord {
    maturity: f64,
    strike: f64,
    annuity: f64,
    bid_vol: f64,
    mid_vol: f64,
    ask_vol: f64,
}

/// Holds the structured market data parsed from a CSV file.
pub struct MarketQuotes {
    pub strikes: Array1<f64>,
    pub maturities: Array1<f64>,
    pub annuity_factors: Array1<f64>,
    pub mid_vols: Array2<f64>,
    pub weights: Array2<f64>,
}

/// Parses a CSV file containing swaption quotes in a "long" format.
pub fn parse_quotes_data(file_path: &Path) -> Result<MarketQuotes, DataLoaderError> {
    let mut reader = csv::Reader::from_path(file_path)?;
    let records: Vec<QuoteRecord> = reader.deserialize().collect::<Result<_, _>>()?;

    let mut strikes_map = BTreeMap::new();
    let mut maturities_map = BTreeMap::new();
    for record in &records {
        strikes_map.insert((record.strike * 1e9) as i64, record.strike);
        maturities_map.insert((record.maturity * 1e9) as i64, record.maturity);
    }

    let strikes: Vec<f64> = strikes_map.values().cloned().collect();
    let maturities: Vec<f64> = maturities_map.values().cloned().collect();
    let num_strikes = strikes.len();
    let num_maturities = maturities.len();

    let mut mid_vols = Array2::zeros((num_strikes, num_maturities));
    let mut weights = Array2::zeros((num_strikes, num_maturities));
    let mut annuity_map: BTreeMap<i64, f64> = BTreeMap::new();

    let strike_indices: BTreeMap<_, _> = strikes.iter().enumerate().map(|(i, &s)| ((s * 1e9) as i64, i)).collect();
    let maturity_indices: BTreeMap<_, _> = maturities.iter().enumerate().map(|(i, &m)| ((m * 1e9) as i64, i)).collect();

    for record in records {
        let i = *strike_indices.get(&((record.strike * 1e9) as i64)).unwrap();
        let j = *maturity_indices.get(&((record.maturity * 1e9) as i64)).unwrap();

        mid_vols[[i, j]] = record.mid_vol;
        let spread = record.ask_vol - record.bid_vol;
        weights[[i, j]] = 1.0 / (spread + 1e-12);

        // Store and verify annuity factor
        let maturity_key = (record.maturity * 1e9) as i64;
        if let Some(&existing_annuity) = annuity_map.get(&maturity_key) {
            if (existing_annuity - record.annuity).abs() > 1e-9 {
                return Err(DataLoaderError::InconsistentAnnuity {
                    maturity: record.maturity,
                    annuity1: existing_annuity,
                    annuity2: record.annuity,
                });
            }
        } else {
            annuity_map.insert(maturity_key, record.annuity);
        }
    }

    // Normalize weights for each maturity so they sum to 1
    for mut col in weights.axis_iter_mut(ndarray::Axis(1)) {
        let sum = col.sum();
        if sum > 1e-12 {
            col.mapv_inplace(|x| x / sum);
        }
    }
    
    let annuity_factors = Array1::from_vec(maturities.iter().map(|m| annuity_map[&((*m * 1e9) as i64)]).collect());

    Ok(MarketQuotes {
        strikes: Array1::from_vec(strikes),
        maturities: Array1::from_vec(maturities),
        annuity_factors,
        mid_vols,
        weights,
    })
}

/// Builds an Implied Willow Tree from a CSV file of swaption quotes.
pub fn build_tree_from_csv(
    file_path: &Path,
    f0: f64,
    m: usize,
    gamma: f64,
    alpha: f64,
) -> Result<ImpliedWillowTree, DataLoaderError> {
    let quote_data = parse_quotes_data(file_path)?;
    let prices = convert_vols_to_prices(
        &quote_data.mid_vols,
        &quote_data.strikes,
        &quote_data.maturities,
        f0,
        &quote_data.annuity_factors,
    )?;

    let tree = build_implied_willow_tree(
        f0,
        &quote_data.annuity_factors,
        &quote_data.maturities,
        &quote_data.strikes,
        &prices.payer_prices,
        &prices.receiver_prices,
        &quote_data.weights,
        m,
        gamma,
        alpha,
    )?;
    Ok(tree)
}

/// Saves a calibrated Implied Willow Tree to a file in JSON format.
pub fn save_tree_to_file(
    tree: &ImpliedWillowTree,
    file_path: &Path,
) -> Result<(), DataLoaderError> {
    let serialized = serde_json::to_string_pretty(tree)?;
    std::fs::write(file_path, serialized)?;
    Ok(())
}

/// Loads a calibrated Implied Willow Tree from a JSON file.
pub fn load_tree_from_file(file_path: &Path) -> Result<ImpliedWillowTree, DataLoaderError> {
    let data = std::fs::read_to_string(file_path)?;
    let tree: ImpliedWillowTree = serde_json::from_str(&data)?;
    Ok(tree)
}