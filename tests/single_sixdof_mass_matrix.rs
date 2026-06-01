#![allow(deprecated)]

use approx::assert_relative_eq;
use multibody_dynamics::math_functions::skew;
use multibody_dynamics::multibody::*;
use nalgebra as na;

#[test]
fn single_sixdof_body_mass_matrix_matches_input() {
    // One rigid body, SixDOF joint.
    let offsets = vec![na::Isometry3::identity()];
    let joint_types = vec![JointType::SixDOF];
    let parent = vec![0u16];
    let mass = 5.0;
    let r_cg = na::Vector3::new(0.2, -0.1, 0.05);
    // Simple inertia about COM (diagonal) then shifted to body frame by parallel axis in constructor logic.
    let inertia_com = na::Matrix3::new(0.3, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.2);
    // Provide inertia matrices directly; constructor will assemble 6x6.
    let mb: MultiBody<1, 6> = MultiBody::new(
        offsets,
        None, // mass matrices -> force constructor path
        None, // added mass
        Some(vec![inertia_com]),
        joint_types,
        parent,
        na::Vector3::new(0.0, 0.0, -9.81),
        Some(vec![r_cg]), // r_com
        None,             // r_cob
        Some(vec![mass]), // masses
        None,             // volumes
        None,             // rho
    )
    .unwrap();

    let base = na::Isometry3::identity();
    let scalar_joint_vars = na::SVector::<f64, 0>::zeros();
    let conf = mb.minimal_to_homogeneous_configuration(&base, &scalar_joint_vars);
    let m_mat = mb.compute_mass_matrix(&conf);

    // Manually build expected matrix: top-left mI, top-right -m [r]^x, bottom-left transpose, bottom-right inertia (already parallel-axis corrected inside new()).
    let mut expected = na::SMatrix::<f64, 6, 6>::zeros();
    expected
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(mass * na::Matrix3::identity()));
    expected
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-mass * skew(&r_cg)));
    expected
        .fixed_view_mut::<3, 3>(3, 0)
        .copy_from(&(mass * skew(&r_cg)));
    expected
        .fixed_view_mut::<3, 3>(3, 3)
        .copy_from(&inertia_com);

    assert_relative_eq!(m_mat, expected, epsilon = 1e-12);
}
