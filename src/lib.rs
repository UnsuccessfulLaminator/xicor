//! This crate provides a reasonably efficient implementation of Sourav
//! Chatterjee's xi-correlation coefficient, based on
//! [the original paper](https://arxiv.org/pdf/1909.10140).
//!
//! Chatterjee's xi provides a measure of one variable's dependence on another
//! in a much more general sense than, for example, Pearson's correlation
//! coefficient. Suppose we have some sequence of random `x` values uniformly
//! distributed from zero to tau. For each one, we compute `y = sin(x)`.
//! Pearson's correlation coefficient will be roughly zero for this data, as it
//! measures _linear_ dependence. On the other hand, Chatterjee's xi will be
//! close to 1, representing that `y` is strongly a function of `x`, regardless
//! of what function that may be.
//!
//! ## Highlights
//!
//! - Extremely simple to use (just call [`xicor()`], [`xicorf()`], etc, with
//!   two slices containing the data)
//! - Generic over `Ord`, as xi does not require calculations on the elements
//!   themselves, only the ability to compare them. In principle even strings
//!   could be correlated in this manner (lexicographically), for example.
//! - Quite fast. In release mode on a 12-year-old machine (Dell M4700),
//!   [`xicorf`] was able to process 1,000,000 pairs in 0.33 seconds. Profiling
//!   revealed that 80% of this calculation lay in the standard library's
//!   sorting routines.
//!
//! ## Progress
//!
//! Only calculation of the xi coefficient itself has been implemented so far.
//! The paper also gives a method for finding p-values of the distribution of
//! xi (given certain requirements), and ideally this will also be implemented.

#[cfg(test)]
mod tests;
mod xicor;

pub use xicor::*;
