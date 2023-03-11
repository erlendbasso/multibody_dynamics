
/*!
# multibody_dynamics 
This crate is a multibody dynamics library providing algorithms for computing forward/inverse dynamics and the mass matrix of a multi-body system.


*/

#![cfg_attr(feature = "no_std", no_std)]

#[cfg(feature = "no_std")]
extern crate core as std;

#[allow(non_snake_case)]
#[allow(unused_imports)]




pub mod multibody;
mod math_functions;