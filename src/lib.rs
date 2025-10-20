//! A Rust library for fitting Johnson distributions to statistical moments.
//!
//! This library provides a safe, idiomatic Rust implementation of the Hill, Hill, and Holder (1976)
//! algorithm, originally implemented in FORTRAN and C. It can determine the appropriate
//! distribution family (Normal, Lognormal, Johnson SU, or Johnson SB) for a given set of
//! the first four moments and calculate the distribution's parameters.
//!
//! The library also includes an FFI wrapper around the original C code and a comprehensive
//! test suite that validates the Rust rewrite against the C implementation to ensure
//! numerical equivalence.

// Silence warnings for unused variables in the C FFI struct, as they are named for clarity.
#![allow(dead_code)]

use thiserror::Error;

/// A dedicated error type for the `mom` function.
#[derive(Error, Debug, PartialEq, Clone, Copy)]
#[error("{0}")]
pub struct MomError(&'static str);

/// The primary error types that can occur during the fitting process.
#[derive(Error, Debug, PartialEq, Clone)]
pub enum HhhError {
    /// Input standard deviation was negative.
    #[error("Invalid standard deviation: {0}")]
    InvalidStdDev(String),
    /// The given moments fall in an impossible region of the (B1, B2) plane.
    #[error("Impossible moments: {0}")]
    ImpossibleMoments(String),
    /// The iterative fitting algorithm for the SB distribution failed to converge.
    #[error("SB fit failed to converge: {0}")]
    SbFitFailed(String),
    /// An internal error occurred within the SU fitting algorithm.
    #[error("SU fit failed: {0}")]
    SuFitError(String),
    /// An internal error occurred within the moment calculation for the SB fitter.
    #[error("Moment calculation failed: {0}")]
    MomFailed(#[from] MomError),
}

/// A struct holding the parameters for a Johnson SU (unbounded) distribution.
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct JohnsonSuParams {
    pub gamma: f64,
    pub delta: f64,
    pub lambda: f64,
    pub xi: f64,
}

/// A struct holding the parameters for a Johnson SB (bounded) distribution.
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct JohnsonSbParams {
    pub gamma: f64,
    pub delta: f64,
    pub lambda: f64,
    pub xi: f64,
}

/// An enum representing the different families of distributions that can be fitted.
///
/// This is the primary return type of the `hhh` function, providing a type-safe way
/// to handle the results of the algorithm.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum JohnsonDistribution {
    /// Normal Distribution.
    Normal { gamma: f64, delta: f64, lambda: f64 },
    /// Lognormal (SL) Distribution.
    Lognormal {
        gamma: f64,
        delta: f64,
        lambda: f64,
        xi: f64,
    },
    /// Johnson SU (unbounded) Distribution.
    Su(JohnsonSuParams),
    /// Johnson SB (bounded) Distribution.
    Sb(JohnsonSbParams),
    /// A special case of the SB distribution on the boundary line.
    St { xi: f64, lambda: f64, delta: f64 },
    /// The trivial case of a distribution with zero variance.
    Constant { value: f64 },
}

/// The main entry point to the Hill, Hill, and Holder algorithm.
///
/// Takes the first four statistical moments and returns the best-fitting
/// Johnson or Normal distribution.
///
/// # Arguments
///
/// * `mean` - The mean of the target distribution.
/// * `std_dev` - The standard deviation of the target distribution.
/// * `skew` - The skewness of the target distribution.
/// * `kurtosis` - The kurtosis of the target distribution.
///
/// # Returns
///
/// A `Result` containing either a `JohnsonDistribution` enum on success or an
/// `HhhError` on failure.
pub fn hhh(
    mean: f64,
    std_dev: f64,
    skew: f64,
    kurtosis: f64,
) -> Result<JohnsonDistribution, HhhError> {
    rust_impl::hhh(mean, std_dev, skew, kurtosis)
}

/// Contains the pure Rust re-implementation of the `f_hhh.c` logic.
pub mod rust_impl {
    use super::*;
    use tracing::{debug, error, instrument, trace};

    const TOLERANCE: f64 = 1e-8;
    const SB_FIT_TOLERANCE: f64 = 1e-4;
    const SB_FIT_LIMIT: i32 = 50;
    const SU_FIT_ITERATION_LIMIT: u32 = 50;
    const MOM_LIMIT: i32 = 500;

    /// The main dispatcher function. See top-level `hhh` documentation.
    #[instrument(level = "debug")]
    pub fn hhh(
        mean: f64,
        std_dev: f64,
        skew: f64,
        kurtosis: f64,
    ) -> Result<JohnsonDistribution, HhhError> {
        if std_dev < 0.0 {
            return Err(HhhError::InvalidStdDev(format!(
                "Standard deviation cannot be negative: {std_dev}"
            )));
        }
        if std_dev.abs() < TOLERANCE {
            trace!("Standard deviation is near zero, returning Constant distribution");
            return Ok(JohnsonDistribution::Constant { value: mean });
        }

        let b1 = skew * skew;
        let b2 = kurtosis;
        trace!(skew_sq = b1, kurtosis = b2, "Calculated initial moments");

        // 1. Check against the boundary line
        if b2 < b1 + 1.0 - TOLERANCE {
            return Err(HhhError::ImpossibleMoments(format!(
                "Kurtosis {b2} is in an impossible region for skew^2 {b1}"
            )));
        } else if (b2 - (b1 + 1.0)).abs() < TOLERANCE {
            trace!("Moments lie on the ST boundary line");
            let mut y = 0.5 + 0.5 * (1.0 - 4.0 / (b1 + 4.0)).sqrt();
            if skew > 0.0 {
                y = 1.0 - y;
            }
            let x = std_dev / (y * (1.0 - y)).sqrt();
            let xi = mean - y * x;
            return Ok(JohnsonDistribution::St {
                xi,
                lambda: xi + x,
                delta: y,
            });
        }

        // 2. Check for Normal distribution
        if skew.abs() < TOLERANCE && (b2 - 3.0).abs() < TOLERANCE {
            trace!("Moments match Normal distribution");
            return Ok(JohnsonDistribution::Normal {
                delta: 1.0 / std_dev,
                gamma: -mean / std_dev,
                lambda: 1.0,
            });
        }

        // 3. Check against the Lognormal line
        trace!("Checking against lognormal line");
        let x = 0.5 * b1 + 1.0;
        let y = skew.abs() * (0.25 * b1 + 1.0).sqrt();
        let u = (x + y).cbrt();
        let w = u + 1.0 / u - 1.0;
        let lognormal_kurtosis = w * w * (3.0 + w * (2.0 + w)) - 3.0;
        trace!(
            lognormal_kurtosis,
            "Calculated theoretical lognormal kurtosis"
        );

        if (lognormal_kurtosis - (b2 - 3.0)).abs() < TOLERANCE {
            trace!("Moments match Lognormal distribution");
            let lambda = skew.signum();
            let delta = 1.0 / w.ln().sqrt();
            let gamma = 0.5 * delta * (w * (w - 1.0) / (std_dev * std_dev)).ln();
            let xi = mean - lambda * ((0.5 / delta - gamma) / delta).exp();
            return Ok(JohnsonDistribution::Lognormal {
                gamma,
                delta,
                lambda,
                xi,
            });
        }

        // 4. Decide between SU and SB
        // This logic now exactly matches the C code's comparison of
        // excess kurtosis (`lognormal_kurtosis`) to full kurtosis (`b2`).
        if lognormal_kurtosis > b2 {
            debug!("Choosing SB distribution");
            // Above the Lognormal line -> SB distribution
            sb_fit(mean, std_dev, skew, b2, TOLERANCE).map(JohnsonDistribution::Sb)
        } else {
            debug!("Choosing SU distribution");
            // Below the Lognormal line -> SU distribution
            su_fit(mean, std_dev, skew, b2, TOLERANCE)
        }
    }

    /// Fits an SU distribution.
    #[instrument(level = "debug", skip(tolerance))]
    fn su_fit(
        mean: f64,
        std_dev: f64,
        skew: f64,
        kurtosis: f64,
        tolerance: f64,
    ) -> Result<JohnsonDistribution, HhhError> {
        let b1 = skew * skew;
        let b3 = kurtosis - 3.0;

        let initial_w_arg = 2.0 * kurtosis - 2.8 * b1 - 2.0;
        if initial_w_arg <= 1.0 {
            let err_msg = "Invalid moments for SU distribution (sqrt of non-positive).";
            error!(initial_w_arg, err_msg);
            return Err(HhhError::SuFitError(err_msg.to_string()));
        }
        let mut w = (initial_w_arg.sqrt() - 1.0).sqrt();
        trace!(initial_w = w, "Calculated initial w");

        let mut y = 0.0;

        if skew.abs() > tolerance {
            let mut converged = false;
            for i in 0..SU_FIT_ITERATION_LIMIT {
                trace!(iteration = i, w, "Starting SU fit iteration");
                let w1 = w + 1.0;
                let wm1 = w - 1.0;
                let z_intermediate = w1 * b3;
                let v = w * (6.0 + w * (3.0 + w));
                let a = 8.0 * (wm1 * (3.0 + w * (7.0 + v)) - z_intermediate);
                let b = 16.0 * (wm1 * (6.0 + v) - b3);

                let discriminant = a * a
                    - 2.0
                        * b
                        * (wm1 * (3.0 + w * (9.0 + w * (10.0 + v))) - 2.0 * w1 * z_intermediate);
                if discriminant < 0.0 {
                    let err_msg = "Negative discriminant in SU fitting iteration.";
                    error!(discriminant, err_msg);
                    return Err(HhhError::SuFitError(err_msg.to_string()));
                }

                y = (discriminant.sqrt() - a) / b;

                let calculated_b1 = y * wm1 * (4.0 * (w + 2.0) * y + 3.0 * w1 * w1).powi(2)
                    / (2.0 * (2.0 * y + w1).powi(3));
                trace!(calculated_b1, "Calculated b1 in iteration");

                let v = w * w;
                let w_update_arg = 1.0
                    - 2.0
                        * (1.5 - kurtosis
                            + (b1 * (kurtosis - 1.5 - v * (1.0 + 0.5 * v))) / calculated_b1);
                if w_update_arg <= 0.0 {
                    let err_msg = "Invalid argument for sqrt in w update.";
                    error!(w_update_arg, err_msg);
                    return Err(HhhError::SuFitError(err_msg.to_string()));
                }

                w = (w_update_arg.sqrt() - 1.0).sqrt();
                trace!(new_w = w, "Updated w");

                if (b1 - calculated_b1).abs() <= tolerance {
                    trace!("SU fit converged");
                    converged = true;
                    break;
                }
            }

            if !converged {
                let err_msg = "SU fit failed to converge within the iteration limit.";
                error!(limit = SU_FIT_ITERATION_LIMIT, err_msg);
                return Err(HhhError::SuFitError(err_msg.to_string()));
            }

            y /= w;
            y = (y.sqrt() + (y + 1.0).sqrt()).ln();
            if skew > 0.0 {
                y = -y;
            }
        }

        let delta = (1.0 / w.ln()).sqrt();
        let gamma = y * delta;
        let exp_y = y.exp();
        let z = exp_y * exp_y;
        let lambda_denom_sq = 0.5 * (w - 1.0) * (0.5 * w * (z + 1.0 / z) + 1.0);

        if lambda_denom_sq <= 0.0 {
            let err_msg = "Non-positive variance in final parameter calculation.";
            error!(lambda_denom_sq, err_msg);
            return Err(HhhError::SuFitError(err_msg.to_string()));
        }
        let lambda = std_dev / lambda_denom_sq.sqrt();
        let xi = (0.5 * w.sqrt() * (exp_y - (-y).exp())) * lambda + mean;

        trace!(gamma, delta, lambda, xi, "Calculated final SU parameters");
        Ok(JohnsonDistribution::Su(JohnsonSuParams {
            gamma,
            delta,
            lambda,
            xi,
        }))
    }

    /// Fits an SB distribution.
    #[instrument(level = "debug", skip(_tolerance))]
    fn sb_fit(
        mean: f64,
        std_dev: f64,
        skew: f64,
        kurtosis: f64,
        _tolerance: f64,
    ) -> Result<JohnsonSbParams, HhhError> {
        let rb1 = skew.abs();
        let b1 = rb1 * rb1;
        let is_negative_skew = skew < 0.0;

        // Phase 1: Initial Guess
        trace!("Calculating initial guess for g and d");
        let (mut g, mut d) = {
            let e = b1 + 1.0;
            let x = 0.5 * b1 + 1.0;
            let y = rb1 * (0.25 * b1 + 1.0).sqrt();
            let u = (x + y).cbrt();
            let w = u + 1.0 / u - 1.0;
            let f = w * w * (3.0 + w * (2.0 + w)) - 3.0;
            let e = (kurtosis - e) / (f - e);

            let d_guess = if rb1.abs() > TOLERANCE {
                let d_temp = 1.0 / (w.ln()).sqrt();
                let f_temp = if d_temp < 0.64 {
                    1.25 * d_temp
                } else {
                    2.0 - 8.5245 / (d_temp * (d_temp * (d_temp - 2.163) + 11.346))
                };
                let f_final = e * f_temp + 1.0;
                if f_final < 1.8 {
                    0.8 * (f_final - 1.0)
                } else {
                    (0.626 * f_final - 0.408) * (3.0 - f_final).powf(-0.479)
                }
            } else {
                let f_temp = e * 2.0 + 1.0;
                if f_temp < 1.8 {
                    0.8 * (f_temp - 1.0)
                } else {
                    (0.626 * f_temp - 0.408) * (3.0 - f_temp).powf(-0.479)
                }
            };

            let g_guess = if b1 < SB_FIT_TOLERANCE {
                0.0
            } else if d_guess <= 1.0 {
                (0.7466 * d_guess.powf(1.7973) + 0.5955) * b1.powf(0.485)
            } else {
                let (u_g, y_g) = if d_guess <= 2.5 {
                    (0.0623, 0.4043)
                } else {
                    (0.0124, 0.5291)
                };
                b1.powf(u_g * d_guess + y_g) * (0.9281 + d_guess * (1.0614 * d_guess - 0.7077))
            };
            (g_guess, d_guess)
        };
        debug!(initial_g = g, initial_d = d, "Initial guess complete");

        // Phase 2: Main Iteration Loop
        for i in 0..SB_FIT_LIMIT {
            trace!(iteration = i, g, d, "Starting SB fit iteration");
            let hmu = mom(g, d)?;

            let s = hmu[0] * hmu[0];
            let h2 = hmu[1] - s;
            if h2 <= 0.0 {
                let err_msg = format!("Non-positive variance ({h2}) encountered during fitting.");
                error!(h2, "SB fit failed");
                return Err(HhhError::SbFitFailed(err_msg));
            }
            let t = h2.sqrt();
            let h2a = t * h2;
            let h2b = h2 * h2;
            let h3 = hmu[2] - hmu[0] * (3.0 * hmu[1] - 2.0 * s);
            let rbet = h3 / h2a;
            let h4 = hmu[3] - hmu[0] * (4.0 * hmu[2] - hmu[0] * (6.0 * hmu[1] - 3.0 * s));
            let bet2 = h4 / h2b;
            trace!(
                calculated_skew = rbet,
                calculated_kurtosis = bet2,
                "Calculated moments in iteration"
            );

            let w = g * d;
            let u = d * d;

            let mut deriv = [0.0; 4];

            for j in 0..2 {
                let mut dd = [0.0; 4];
                for k in 0..4 {
                    let tk = (k + 1) as f64;
                    let s_deriv = if j == 0 {
                        hmu[k + 1] - hmu[k]
                    } else {
                        ((w - tk) * (hmu[k] - hmu[k + 1]) + (tk + 1.0) * (hmu[k + 1] - hmu[k + 2]))
                            / u
                    };
                    dd[k] = tk * s_deriv / d;
                }
                let t_deriv = 2.0 * hmu[0] * dd[0];
                let s_deriv2 = hmu[0] * dd[1];
                let y_deriv = dd[1] - t_deriv;
                let deriv_j = (dd[2]
                    - 3.0 * (s_deriv2 + hmu[1] * dd[0] - t_deriv * hmu[0])
                    - 1.5 * h3 * y_deriv / h2)
                    / h2a;
                let deriv_j2 = (dd[3] - 4.0 * (dd[2] * hmu[0] + dd[0] * hmu[2])
                    + 6.0 * (hmu[1] * t_deriv + hmu[0] * (s_deriv2 - t_deriv * hmu[0]))
                    - 2.0 * h4 * y_deriv / h2)
                    / h2b;
                deriv[j] = deriv_j;
                deriv[j + 2] = deriv_j2;
            }
            trace!(?deriv, "Calculated derivatives");

            let t_inv = 1.0 / (deriv[0] * deriv[3] - deriv[1] * deriv[2]);
            let update_g = (deriv[3] * (rbet - rb1) - deriv[1] * (bet2 - kurtosis)) * t_inv;
            let update_d = (deriv[0] * (bet2 - kurtosis) - deriv[2] * (rbet - rb1)) * t_inv;
            trace!(update_g, update_d, "Calculated updates for g and d");

            g -= update_g;
            if b1.abs() < TOLERANCE || g < 0.0 {
                g = 0.0;
            }
            d -= update_d;

            if update_g.abs() < SB_FIT_TOLERANCE && update_d.abs() < SB_FIT_TOLERANCE {
                debug!(iterations = i, "SB fit converged");
                let h2_final = hmu[1] - hmu[0] * hmu[0];
                let lambda = std_dev / h2_final.sqrt();
                let final_gamma = if is_negative_skew { -g } else { g };
                let final_hmu1 = if is_negative_skew {
                    1.0 - hmu[0]
                } else {
                    hmu[0]
                };
                let xi = mean - lambda * final_hmu1;

                trace!(
                    gamma = final_gamma,
                    delta = d,
                    lambda,
                    xi,
                    "Calculated final SB parameters"
                );
                return Ok(JohnsonSbParams {
                    gamma: final_gamma,
                    delta: d,
                    lambda,
                    xi,
                });
            }
        }

        let err_msg = format!("SB fit failed to converge after {SB_FIT_LIMIT} iterations.");
        error!(limit = SB_FIT_LIMIT, "SB fit failed");
        Err(HhhError::SbFitFailed(err_msg))
    }

    /// Calculates the first six raw moments for an SB distribution using the original HHH algorithm.
    #[instrument(level = "trace")]
    pub fn mom_original(g: f64, d: f64) -> Result<[f64; 6], MomError> {
        let zz = 1.0e-5;
        let vv = 1.0e-8;

        let mut a = [0.0; 6];
        let mut c = [0.0; 6];

        let w = g / d;
        if w > 80.0 {
            return Err(MomError("Input 'g/d' is too large, may cause overflow."));
        }

        let e = w.exp() + 1.0;
        let r = std::f64::consts::SQRT_2 / d; // rttwo from C
        let mut h = if d < 3.0 { 0.25 * d } else { 0.75 };

        for k in 1..=MOM_LIMIT {
            trace!(iteration = k, h, "Starting outer mom loop");
            if k > 1 {
                c.copy_from_slice(&a);
                h *= 0.5;
            }

            let mut t = w;
            let mut u = t;
            let mut y = h * h;
            let x = 2.0 * y;
            a[0] = 1.0 / e;
            for i in 1..6 {
                a[i] = a[i - 1] / e;
            }
            let mut v = y;
            let f = r * h;

            let mut inner_converged = false;
            for m in 1..=MOM_LIMIT {
                let b = a;
                u -= f;
                let z = if u > -23.7 { u.exp() + 1.0 } else { 1.0 };
                t += f;
                let (mut l, s) = if t > 23.7 {
                    (true, 0.0)
                } else {
                    (false, t.exp() + 1.0)
                };

                let mut p = (-v).exp();
                let mut q = p;

                'series_loop: for aa in a.iter_mut() {
                    p /= z;
                    let ab = *aa;
                    *aa += p;
                    if *aa == ab {
                        break 'series_loop;
                    }
                    if !l {
                        q /= s;
                        let ab_q = *aa;
                        *aa += q;
                        if *aa == ab_q {
                            l = true;
                        }
                    }
                }

                y += x;
                v += y;
                let mut all_vv_converged = true;
                for i in 0..6 {
                    if a[i] == 0.0 {
                        error!("Moment calculation resulted in zero.");
                        return Err(MomError("Moment calculation resulted in zero."));
                    }
                    if ((a[i] - b[i]) / a[i]).abs() > vv {
                        all_vv_converged = false;
                        break;
                    }
                }
                if all_vv_converged {
                    trace!(
                        inner_iterations = m,
                        "Inner mom loop (series summation) converged"
                    );
                    inner_converged = true;
                    break;
                }
            }

            if !inner_converged {
                error!(
                    limit = MOM_LIMIT,
                    "Inner mom loop (series summation) failed to converge."
                );
                return Err(MomError(
                    "Inner loop (series summation) failed to converge.",
                ));
            }

            let v_final = 0.5641895835 * h; // rrtpi from C
            for val in a.iter_mut() {
                *val *= v_final;
            }

            if k > 1 {
                let mut outer_converged = true;
                for i in 0..6 {
                    if a[i] == 0.0 {
                        error!("Moment calculation resulted in zero during outer loop.");
                        return Err(MomError(
                            "Moment calculation resulted in zero during outer loop.",
                        ));
                    }
                    if ((a[i] - c[i]) / a[i]).abs() > zz {
                        outer_converged = false;
                        break;
                    }
                }
                if outer_converged {
                    debug!(outer_iterations = k, "Outer mom loop converged");
                    return Ok(a);
                }
            }
        }
        error!(limit = MOM_LIMIT, "Outer mom loop failed to converge.");
        Err(MomError("Outer loop failed to converge."))
    }

    use gauss_quad::GaussHermite;
    use std::sync::OnceLock;

    const QUADRATURE_POINTS: usize = 64;
    static HERMITE_RULE: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();

    /// Calculates the first six raw moments for an SB distribution using Gauss-Hermite quadrature.
    #[instrument(level = "trace")]
    pub fn mom_gauss_hermite(g: f64, d: f64) -> Result<[f64; 6], MomError> {
        let (nodes, weights) = HERMITE_RULE.get_or_init(|| {
            let rule = GaussHermite::new(QUADRATURE_POINTS).unwrap();
            (
                rule.nodes().copied().collect(),
                rule.weights().copied().collect(),
            )
        });

        let mut moments = [0.0; 6];

        let sqrt2 = std::f64::consts::SQRT_2;

        for i in 0..QUADRATURE_POINTS {
            let x = nodes[i];
            let w = weights[i];
            let z = sqrt2 * x;

            let integrand_base = 1.0 / (1.0 + (-(z - g) / d).exp());
            let mut val = 1.0;

            for moment in moments.iter_mut() {
                val *= integrand_base; // val is now integrand_base^(k+1)
                *moment += w * val;
            }
        }

        let scale = 1.0 / std::f64::consts::PI.sqrt();
        for moment in moments.iter_mut() {
            *moment *= scale;
        }

        Ok(moments)
    }

    /// Calculates the first six raw moments for an SB distribution.
    fn mom(g: f64, d: f64) -> Result<[f64; 6], MomError> {
        mom_gauss_hermite(g, d)
    }
}

/// Contains the FFI bindings and a safe wrapper for the original C code.
pub mod c_ffi {
    use super::*;

    // This struct is used to receive the output from the C function.
    // It's defined here to match the C function's expectations.
    #[repr(C)]
    #[derive(Default)]
    struct HhhCOutput {
        itype: f64,
        gamma: f64,
        delta: f64,
        xlam: f64,
        xi: f64,
        ifault: f64,
    }

    // FFI declarations for the C functions.
    #[link(name = "f_hhh", kind = "static")]
    unsafe extern "C" {
        fn hhh(
            xbar: f64,
            sd: f64,
            rb1: f64,
            bb2: f64,
            itype: *mut f64,
            gamma: *mut f64,
            delta: *mut f64,
            xlam: *mut f64,
            xi: *mut f64,
            ifault: *mut f64,
        );
    }

    /// A safe Rust wrapper around the `unsafe` C `hhh` function.
    ///
    /// This function handles the pointer logic and converts the C-style output
    /// into an idiomatic Rust `Result` type, making it easy and safe to use
    /// from other Rust code, especially in the test harness.
    pub fn hhh_c_wrapper(
        mean: f64,
        std_dev: f64,
        skew: f64,
        kurtosis: f64,
    ) -> Result<JohnsonDistribution, HhhError> {
        let mut out = HhhCOutput::default();

        unsafe {
            hhh(
                mean,
                std_dev,
                skew,
                kurtosis,
                &mut out.itype,
                &mut out.gamma,
                &mut out.delta,
                &mut out.xlam,
                &mut out.xi,
                &mut out.ifault,
            );
        }

        if out.ifault != 0.0 {
            return match out.ifault as i32 {
                1 => Err(HhhError::InvalidStdDev("C code failed".to_string())),
                2 => Err(HhhError::ImpossibleMoments("C code failed".to_string())),
                3 => Err(HhhError::SbFitFailed("C code failed".to_string())),
                _ => panic!("Unknown C error code"),
            };
        }

        match out.itype as i32 {
            1 => Ok(JohnsonDistribution::Lognormal {
                gamma: out.gamma,
                delta: out.delta,
                lambda: out.xlam,
                xi: out.xi,
            }),
            2 => Ok(JohnsonDistribution::Su(JohnsonSuParams {
                gamma: out.gamma,
                delta: out.delta,
                lambda: out.xlam,
                xi: out.xi,
            })),
            3 => Ok(JohnsonDistribution::Sb(JohnsonSbParams {
                gamma: out.gamma,
                delta: out.delta,
                lambda: out.xlam,
                xi: out.xi,
            })),
            4 => Ok(JohnsonDistribution::Normal {
                gamma: out.gamma,
                delta: out.delta,
                lambda: out.xlam,
            }),
            5 => {
                if std_dev.abs() < 1e-9 {
                    Ok(JohnsonDistribution::Constant { value: mean })
                } else {
                    Ok(JohnsonDistribution::St {
                        xi: out.xi,
                        lambda: out.xlam,
                        delta: out.delta,
                    })
                }
            }
            _ => panic!("Unknown C itype code"),
        }
    }
}

/// The test harness for comparing the Rust and C implementations.
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Once;

    const FLOAT_TOLERANCE: f64 = 1e-6;
    const SU_FLOAT_TOLERANCE: f64 = 1e-4;

    static TRACING_INIT: Once = Once::new();

    fn init_tracing() {
        TRACING_INIT.call_once(|| {
            tracing_subscriber::fmt()
                .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
                .with_test_writer()
                .init();
        });
    }

    fn compare_results(
        rust_res: Result<JohnsonDistribution, HhhError>,
        c_res: Result<JohnsonDistribution, HhhError>,
    ) {
        println!("Rust result: {rust_res:?}");
        println!("C    result: {c_res:?}");

        assert_eq!(rust_res.is_ok(), c_res.is_ok());
        // We don't compare the string content, just the error type
        assert_eq!(
            std::mem::discriminant(&rust_res),
            std::mem::discriminant(&c_res)
        );

        if let (Ok(rust_dist), Ok(c_dist)) = (rust_res, c_res) {
            match (rust_dist, c_dist) {
                (
                    JohnsonDistribution::Normal {
                        gamma,
                        delta,
                        lambda,
                    },
                    JohnsonDistribution::Normal {
                        gamma: c_gamma,
                        delta: c_delta,
                        lambda: c_lambda,
                    },
                ) => {
                    assert!((gamma - c_gamma).abs() < FLOAT_TOLERANCE);
                    assert!((delta - c_delta).abs() < FLOAT_TOLERANCE);
                    assert!((lambda - c_lambda).abs() < FLOAT_TOLERANCE);
                }
                (
                    JohnsonDistribution::Lognormal {
                        gamma,
                        delta,
                        lambda,
                        xi,
                    },
                    JohnsonDistribution::Lognormal {
                        gamma: c_gamma,
                        delta: c_delta,
                        lambda: c_lambda,
                        xi: c_xi,
                    },
                ) => {
                    assert!((gamma - c_gamma).abs() < FLOAT_TOLERANCE);
                    assert!((delta - c_delta).abs() < FLOAT_TOLERANCE);
                    assert!((lambda - c_lambda).abs() < FLOAT_TOLERANCE);
                    assert!((xi - c_xi).abs() < FLOAT_TOLERANCE);
                }
                (JohnsonDistribution::Su(p), JohnsonDistribution::Su(cp)) => {
                    assert!((p.gamma - cp.gamma).abs() < SU_FLOAT_TOLERANCE);
                    assert!((p.delta - cp.delta).abs() < SU_FLOAT_TOLERANCE);
                    assert!((p.lambda - cp.lambda).abs() < SU_FLOAT_TOLERANCE);
                    assert!((p.xi - cp.xi).abs() < SU_FLOAT_TOLERANCE);
                }
                (JohnsonDistribution::Sb(p), JohnsonDistribution::Sb(cp)) => {
                    assert!((p.gamma - cp.gamma).abs() < FLOAT_TOLERANCE);
                    assert!((p.delta - cp.delta).abs() < FLOAT_TOLERANCE);
                    assert!((p.lambda - cp.lambda).abs() < FLOAT_TOLERANCE);
                    assert!((p.xi - cp.xi).abs() < FLOAT_TOLERANCE);
                }
                (
                    JohnsonDistribution::Constant { value },
                    JohnsonDistribution::Constant { value: c_value },
                ) => {
                    assert!((value - c_value).abs() < FLOAT_TOLERANCE);
                }
                _ => {
                    unreachable!("Mismatched distribution types between Rust and C implementations")
                }
            }
        }
    }

    #[test]
    fn test_normal_case() {
        init_tracing();
        let (mean, std_dev, skew, kurtosis) = (0.0, 1.0, 0.0, 3.0);
        let rust_res = rust_impl::hhh(mean, std_dev, skew, kurtosis);
        let c_res = c_ffi::hhh_c_wrapper(mean, std_dev, skew, kurtosis);
        compare_results(rust_res, c_res);
    }

    #[test]
    fn test_su_case() {
        init_tracing();
        let (mean, std_dev, skew, kurtosis) = (0.0, 1.0, 1.0, 5.0);
        let rust_res = rust_impl::hhh(mean, std_dev, skew, kurtosis);
        let c_res = c_ffi::hhh_c_wrapper(mean, std_dev, skew, kurtosis);
        compare_results(rust_res, c_res);
    }

    #[test]
    fn test_sb_case() {
        init_tracing();
        let (mean, std_dev, skew, kurtosis) = (0.5, 0.2, 0.1, 2.5);
        let rust_res = rust_impl::hhh(mean, std_dev, skew, kurtosis);
        let c_res = c_ffi::hhh_c_wrapper(mean, std_dev, skew, kurtosis);
        compare_results(rust_res, c_res);
    }

    #[test]
    fn test_lognormal_case() {
        init_tracing();
        // These moments are characteristic of a lognormal distribution
        let (mean, std_dev, skew, kurtosis) = (1.6487, 2.136, 6.1848, 113.936);
        let rust_res = rust_impl::hhh(mean, std_dev, skew, kurtosis);
        let c_res = c_ffi::hhh_c_wrapper(mean, std_dev, skew, kurtosis);
        compare_results(rust_res, c_res);
    }

    #[test]
    fn test_impossible_moments() {
        init_tracing();
        let (mean, std_dev, skew, kurtosis) = (0.0, 1.0, 1.0, 1.5);
        let rust_res = rust_impl::hhh(mean, std_dev, skew, kurtosis);
        let c_res = c_ffi::hhh_c_wrapper(mean, std_dev, skew, kurtosis);
        assert!(matches!(rust_res, Err(HhhError::ImpossibleMoments(_))));
        assert!(matches!(c_res, Err(HhhError::ImpossibleMoments(_))));
    }

    #[test]
    fn test_constant_value() {
        init_tracing();
        let (mean, std_dev, skew, kurtosis) = (10.0, 0.0, 0.0, 0.0);
        let rust_res = rust_impl::hhh(mean, std_dev, skew, kurtosis);
        let c_res = c_ffi::hhh_c_wrapper(mean, std_dev, skew, kurtosis);
        compare_results(rust_res, c_res);
    }

    #[test]
    fn test_mom_implementations() {
        init_tracing();
        let test_cases = [
            (0.24185868, 1.57464412), // Original test case from sb_fit
            (0.0, 1.0),               // Symmetrical case
            (1.0, 2.0),               // Higher skew, higher delta
            (0.1, 0.5),               // Low skew, low delta
            (0.5, 5.0),               // Moderate skew, high delta
        ];

        for (g, d) in test_cases {
            println!("Testing mom implementations with g={g}, d={d}");
            let original_res = rust_impl::mom_original(g, d).unwrap();
            let gauss_res = rust_impl::mom_gauss_hermite(g, d).unwrap();

            println!("Original: {original_res:?}");
            println!("Gauss-H:  {gauss_res:?}");

            for i in 0..6 {
                assert!(
                    (original_res[i] - gauss_res[i]).abs() < 1e-7,
                    "Moment {} differs for g={}, d={}: Original={}, Gauss-H={}",
                    i,
                    g,
                    d,
                    original_res[i],
                    gauss_res[i]
                );
            }
        }
    }
}
