#![allow(deprecated)]

use approx::assert_relative_eq;
use multibody_dynamics::multibody::*;
use nalgebra as na;

#[test]
fn per_body_vs_batch_jacobian() {
    // Small 3 body chain: SixDOF + two revolute Z
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
    // Minimal inertias for convenience
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
    let joint_angles = na::SVector::<f64, 2>::from_vec(vec![0.3, -0.7]);
    let conf = mb.minimal_to_homogeneous_configuration(&base, &joint_angles);

    let j_all = mb.compute_jacobians(&conf);
    for (body, jac_all_body) in j_all.iter().enumerate().take(3) {
        let j_single = mb.compute_jacobian(&conf, body);
        assert_relative_eq!(*jac_all_body, j_single, epsilon = 1e-12);
    }
}
