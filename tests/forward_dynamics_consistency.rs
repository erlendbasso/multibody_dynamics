#![allow(deprecated)]

use approx::assert_relative_eq;
use multibody_dynamics::multibody::*;
use nalgebra as na;

#[test]
fn forward_dynamics_matches_inverse_when_no_forces() {
    let offsets = vec![
        na::Isometry3::identity(),
        na::Isometry3::identity(),
        na::Isometry3::identity(),
    ];
    let mut joint_types = vec![JointType::Revolute(Axis::Z); 3];
    joint_types[0] = JointType::SixDOF;
    let parent = vec![0u16, 1u16, 2u16];
    let masses = vec![1.0, 1.0, 1.0];
    let r_cg = vec![na::Vector3::zeros(); 3];
    let inertia = vec![na::Matrix3::identity(); 3];
    let mb: MultiBody<3, 8> = MultiBody::new(
        offsets,
        None,
        None,
        Some(inertia),
        joint_types,
        parent,
        na::Vector3::new(0.0, 0.0, -9.81),
        Some(r_cg),
        None,
        Some(masses),
        None,
        None,
    )
    .unwrap();

    let base = na::Isometry3::identity();
    let joint_angles = na::SVector::<f64, 2>::from_vec(vec![0.2, -0.1]);
    let conf = mb.minimal_to_homogeneous_configuration(&base, &joint_angles);
    let mu = na::SVector::<f64, 8>::repeat(0.3); // some velocities

    let rigid_body_forces_func2 =
        &|_x: &[na::Isometry3<f64>], _y: &[na::SVector<f64, 6>]| -> na::SMatrix<f64, 6, 3> {
            na::SMatrix::<f64, 6, 3>::zeros()
        };
    let rigid_body_forces_func1 =
        &|_x: &[na::SVector<f64, 6>], _y: &[na::SVector<f64, 6>]| -> na::SMatrix<f64, 6, 3> {
            na::SMatrix::<f64, 6, 3>::zeros()
        };
    let thruster_forces = vec![na::SVector::<f64, 6>::zeros(); 3];
    let eta = na::SVector::<f64, 8>::zeros();
    let lin_vel_current = na::Vector3::zeros();
    let lin_accel_current = na::Vector3::zeros();

    let accel = mb.forward_dynamics_ab(
        &conf,
        &mu,
        rigid_body_forces_func2,
        &thruster_forces,
        &eta,
        &lin_vel_current,
        &lin_accel_current,
    );
    let sigma_prime = na::SVector::<f64, 8>::zeros();
    let c_vec =
        mb.generalized_newton_euler(&conf, &mu, &mu, &sigma_prime, rigid_body_forces_func1, &eta);
    let mass_mat = mb.compute_mass_matrix(&conf);
    let accel2 = -mass_mat.try_inverse().unwrap() * c_vec;
    assert_relative_eq!(accel, accel2, epsilon = 1e-9);
}
