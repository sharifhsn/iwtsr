use f_hhh::{
    data_loader::{
        build_tree_from_csv, load_tree_from_file, save_tree_to_file, MarketQuotes,
        parse_quotes_data,
    },
    f_hhh::hhh,
    moments::implied_moments_normal,
    nodes::wt_nodes_from_johnson_curve,
    pricing::price_european_swaption,
    vols_to_prices::{convert_prices_to_vols, convert_vols_to_prices},
};
use clap::{Parser, Subcommand};
use csv::Writer;
use serde::Serialize;
use statrs::distribution::ContinuousCDF;
use std::error::Error;
use std::path::PathBuf;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build an Implied Willow Tree from market data and save it to a file.
    Build(BuildArgs),
    /// Price a European swaption using a pre-calibrated model file.
    Price(PriceArgs),
    /// Generate a synthetic swaption dataset based on the Bachelier model.
    GenerateData(GenerateDataArgs),
    /// Calculate and output the moments and Johnson fit for a specific maturity.
    Moments(MomentsArgs),
    /// Generate and output the results of the Johnson curve inverse transformation.
    Johnson(JohnsonArgs),
}

#[derive(clap::Args)]
struct BuildArgs {
    /// Path to the input CSV file containing swaption quotes.
    #[arg(short, long, value_name = "FILE")]
    file: PathBuf,

    /// Path to save the output model file (e.g., model.json).
    #[arg(short, long, value_name = "FILE")]
    output: PathBuf,

    /// The initial forward swap rate (e.g., 0.03 for 3%).
    #[arg(long)]
    f0: f64,

    /// The number of nodes (m) for the willow tree.
    #[arg(long, default_value_t = 30)]
    nodes: usize,

    /// The node spacing parameter (gamma).
    #[arg(long, default_value_t = 0.6)]
    gamma: f64,

    /// The smoothness regularization parameter (alpha). Defaults to 100 * f0 if not provided.
    #[arg(long)]
    alpha: Option<f64>,
}

#[derive(clap::Args)]
struct PriceArgs {
    /// Path to the pre-calibrated model file (e.g., model.json).
    #[arg(short, long, value_name = "FILE")]
    model: PathBuf,

    /// The strike to price (e.g., 0.035).
    #[arg(long)]
    strike: f64,

    /// The specific maturity to price for (e.g., 0.5 for 6 months).
    #[arg(long)]
    maturity: f64,

    /// The annuity factor corresponding to the specified maturity.
    #[arg(long)]
    annuity: f64,

    /// The initial forward swap rate (e.g., 0.03 for 3%).
    #[arg(long)]
    f0: f64,

    /// Output the implied normal volatilities instead of prices.
    #[arg(long)]
    output_vols: bool,
}

#[derive(clap::Args)]
struct GenerateDataArgs {
    /// Path to save the output CSV file.
    #[arg(long, value_name = "FILE")]
    output: PathBuf,

    /// The central forward swap rate (e.g., 0.03 for 3%).
    #[arg(long, default_value_t = 0.03)]
    f0: f64,

    /// Comma-separated list of maturities in years (e.g., "0.25,0.5,1.0").
    #[arg(long, default_value = "0.25,0.5,0.75,1.0")]
    maturities: String,

    /// Number of strikes to generate.
    #[arg(long, default_value_t = 21)]
    num_strikes: usize,

    /// Width of the strike range in basis points (bps) around f0.
    #[arg(long, default_value_t = 200)]
    strike_range_bps: u32,

    /// The at-the-money normal volatility (e.g., 0.0185 for 185 bps).
    #[arg(long, default_value_t = 0.0185)]
    atm_vol: f64,

    /// The base bid-ask spread in basis points (bps) for at-the-money volatility.
    #[arg(long, default_value_t = 10)]
    spread_bps: u32,
}

#[derive(clap::Args)]
struct MomentsArgs {
    /// Path to the input CSV file.
    #[arg(long, value_name = "FILE")]
    file: PathBuf,

    /// The initial forward swap rate (e.g., 0.03 for 3%).
    #[arg(long)]
    f0: f64,

    /// The index of the maturity to analyze (0-based).
    #[arg(long)]
    maturity_idx: usize,
}

#[derive(clap::Args)]
struct JohnsonArgs {
    /// Path to the input CSV file.
    #[arg(long, value_name = "FILE")]
    file: PathBuf,

    /// The initial forward swap rate (e.g., 0.03 for 3%).
    #[arg(long)]
    f0: f64,

    /// The index of the maturity to analyze (0-based).
    #[arg(long)]
    maturity_idx: usize,

    /// The number of nodes (m) for the willow tree.
    #[arg(long, default_value_t = 50)]
    nodes: usize,

    /// The node spacing parameter (gamma).
    #[arg(long, default_value_t = 0.6)]
    gamma: f64,
}

#[derive(Serialize)]
struct MomentsOutput {
    mean: f64,
    variance: f64,
    skewness: f64,
    kurtosis: f64,
    johnson_params: serde_json::Value,
}

fn run_generate_data(args: &GenerateDataArgs) -> Result<(), Box<dyn Error>> {
    let maturities: Vec<f64> = args
        .maturities
        .split(',')
        .map(|s| s.trim().parse())
        .collect::<Result<_, _>>()?;

    let f0 = args.f0;
    if args.num_strikes < 2 {
        return Err("Number of strikes must be at least 2.".into());
    }
    let strike_step = (args.strike_range_bps as f64 / 10000.0) / (args.num_strikes - 1) as f64;
    let min_strike = f0 - (args.strike_range_bps as f64 / 20000.0);
    let strikes: Vec<f64> = (0..args.num_strikes)
        .map(|i| min_strike + i as f64 * strike_step)
        .collect();

    let mut wtr = Writer::from_path(&args.output)?;
    wtr.write_record(&[
        "maturity",
        "strike",
        "annuity",
        "bid_vol",
        "mid_vol",
        "ask_vol",
    ])?;

    for &maturity in &maturities {
        // Simple synthetic annuity: T * 0.96 (just an approximation for realistic values)
        let annuity = maturity * 0.96;

        for &strike in &strikes {
            // Create a simple quadratic volatility smile
            let moneyness = (strike - f0).abs();
            let smile_factor = 1.0 + 15.0 * moneyness.powi(2);
            let mid_vol = args.atm_vol * smile_factor;

            // Create a dynamic spread that widens away from ATM
            let spread_factor = 1.0 + 10.0 * moneyness;
            let spread = (args.spread_bps as f64 / 10000.0) * spread_factor;

            let bid_vol = mid_vol - spread / 2.0;
            let ask_vol = mid_vol + spread / 2.0;

            wtr.write_record(&[
                maturity.to_string(),
                strike.to_string(),
                annuity.to_string(),
                bid_vol.to_string(),
                mid_vol.to_string(),
                ask_vol.to_string(),
            ])?;
        }
    }

    wtr.flush()?;
    println!(
        "Successfully generated synthetic data at {:?}",
        args.output
    );
    Ok(())
}

use tracing_subscriber;

fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    match &cli.command {
        Commands::Build(args) => {
            println!("Building Implied Willow Tree...");
            let alpha = args.alpha.unwrap_or(100.0 * args.f0);
            println!("Using alpha (smoothness): {}", alpha);

            let tree = build_tree_from_csv(&args.file, args.f0, args.nodes, args.gamma, alpha)?;

            save_tree_to_file(&tree, &args.output)?;
            println!("Successfully built and saved model to {:?}", args.output);
        }
        Commands::Price(args) => {
            println!(
                "Loading model and pricing swaption for maturity {} and strike {}...",
                args.maturity, args.strike
            );
            let tree = load_tree_from_file(&args.model)?;

            let price =
                price_european_swaption(&tree, args.strike, args.maturity, args.annuity)?;

            if args.output_vols {
                println!("Converting price to normal volatility...");
                let vol = convert_prices_to_vols(
                    price.payer_price,
                    price.receiver_price,
                    args.strike,
                    args.maturity,
                    args.f0,
                    args.annuity,
                )?;

                println!("{:<20} | {:<20}", "Payer Vol", "Receiver Vol");
                println!("{:-<22}|{:-<22}", "", "");
                println!(
                    "{:<20.8} | {:<20.8}",
                    vol.payer_vol, vol.receiver_vol
                );
            } else {
                println!("{:<20} | {:<20}", "Payer Price", "Receiver Price");
                println!("{:-<22}|{:-<22}", "", "");
                println!(
                    "{:<20.8} | {:<20.8}",
                    price.payer_price, price.receiver_price
                );
            }
        }
        Commands::GenerateData(args) => {
            println!("Generating synthetic dataset...");
            run_generate_data(args)?;
        }
        Commands::Moments(args) => {
            let quote_data = parse_quotes_data(&args.file)?;
            let prices = convert_vols_to_prices(
                &quote_data.mid_vols,
                &quote_data.strikes,
                &quote_data.maturities,
                args.f0,
                &quote_data.annuity_factors,
            )?;
            let moments = implied_moments_normal(
                &prices.payer_prices,
                &prices.receiver_prices,
                &quote_data.strikes,
                args.f0,
                &quote_data.annuity_factors,
            )?;

            if args.maturity_idx >= moments.mean.len() {
                return Err(format!(
                    "Maturity index {} is out of bounds.",
                    args.maturity_idx
                )
                .into());
            }

            let mean = moments.mean[args.maturity_idx];
            let variance = moments.variance[args.maturity_idx];
            let skewness = moments.skewness[args.maturity_idx];
            let kurtosis = moments.kurtosis[args.maturity_idx];

            let johnson_fit = hhh(mean, variance.sqrt(), skewness, kurtosis)?;
            let johnson_params = serde_json::to_value(format!("{:?}", johnson_fit))?;

            let output = MomentsOutput {
                mean,
                variance,
                skewness,
                kurtosis,
                johnson_params,
            };

            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        Commands::Johnson(args) => {
            let quote_data = parse_quotes_data(&args.file)?;
            let prices = convert_vols_to_prices(
                &quote_data.mid_vols,
                &quote_data.strikes,
                &quote_data.maturities,
                args.f0,
                &quote_data.annuity_factors,
            )?;
            let moments = implied_moments_normal(
                &prices.payer_prices,
                &prices.receiver_prices,
                &quote_data.strikes,
                args.f0,
                &quote_data.annuity_factors,
            )?;

            if args.maturity_idx >= moments.mean.len() {
                return Err(format!(
                    "Maturity index {} is out of bounds.",
                    args.maturity_idx
                )
                .into());
            }

            // Create the G matrix for the specific maturity
            let g = ndarray::array![
                [moments.mean[args.maturity_idx]],
                [moments.variance[args.maturity_idx]],
                [moments.skewness[args.maturity_idx]],
                [moments.kurtosis[args.maturity_idx]],
            ];

            let result = wt_nodes_from_johnson_curve(&g, args.nodes, args.gamma)?;

            // We need the initial `z` quantiles for comparison, so we recalculate them here.
            // This logic is duplicated from `wt_nodes_from_johnson_curve` but is necessary
            // as the function doesn't return `z`.
            let mut q = vec![0.0; args.nodes];
            for k in 0..=(args.nodes / 2) {
                let val = ((k as f64 + 0.5).powf(args.gamma)) / args.nodes as f64;
                q[k] = val;
                if k < args.nodes {
                    q[args.nodes - 1 - k] = val;
                }
            }
            let q_sum: f64 = q.iter().sum();
            q.iter_mut().for_each(|x| *x /= q_sum);
            let mut z = ndarray::Array1::zeros(args.nodes);
            let normal = statrs::distribution::Normal::new(0.0, 1.0).unwrap();
            z[0] = normal.inverse_cdf(q[0] / 2.0);
            let mut cumulative_q = 0.0;
            for k in 1..args.nodes {
                cumulative_q += q[k - 1];
                let tmp = cumulative_q + q[k] / 2.0;
                z[k] = normal.inverse_cdf(tmp);
            }

            #[derive(Serialize)]
            struct JohnsonOutput {
                initial_quantiles: Vec<f64>,
                transformed_nodes: Vec<f64>,
            }

            let output = JohnsonOutput {
                initial_quantiles: z.to_vec(),
                transformed_nodes: result.willow.into_raw_vec(),
            };

            println!("{}", serde_json::to_string_pretty(&output)?);
        }
    }

    Ok(())
}
