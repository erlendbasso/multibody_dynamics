#![allow(deprecated)]

use approx::assert_relative_eq;
use multibody_dynamics::multibody::*;
use nalgebra as na;

#[test]
fn pure_prismatic_chain_jacobian_linear_only() {
    // Three prismatic joints along X, Y, Z.
    let offsets = vec![
        na::Isometry3::identity(),
        na::Isometry3::identity(),
        na::Isometry3::identity(),
    ];
    let joint_types = vec![
        JointType::Prismatic(Axis::X),
        JointType::Prismatic(Axis::Y),
        JointType::Prismatic(Axis::Z),
    ];
    let parent = vec![0u16, 1u16, 2u16];
    // Provide mass matrices directly (identity mass + zero coupling) for simplicity.
    let mass_mats = vec![na::SMatrix::<f64, 6, 6>::identity(); 3];
    let mb: MultiBody<3, 3> = MultiBody::new(
        offsets,
        Some(mass_mats),
        None,
        None,
        joint_types,
        parent,
        na::Vector3::zeros(),
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();

    let joint_vars = na::SVector::<f64, 3>::from_vec(vec![0.0, 0.0, 0.0]);
    let six_dof_vars: Vec<na::Isometry3<f64>> = Vec::new();
    let conf = mb.minimal_to_homogeneous_configuration(&six_dof_vars, &joint_vars);
    let jac_body2 = mb.compute_jacobian(&conf, 2); // last body index=2

    // For prismatic along X,Y,Z the spatial Jacobian columns should be e_x, e_y, e_z in linear part.
    for (i, axis) in [na::Vector3::x(), na::Vector3::y(), na::Vector3::z()]
        .iter()
        .enumerate()
    {
        let col = jac_body2.fixed_columns::<1>(i);
        let linear = na::Vector3::new(col[0], col[1], col[2]);
        let angular = na::Vector3::new(col[3], col[4], col[5]);
        assert_relative_eq!(linear, *axis, epsilon = 1e-14);
        assert_relative_eq!(angular, na::Vector3::zeros(), epsilon = 1e-14);
    }
}
