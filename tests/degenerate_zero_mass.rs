#![allow(deprecated)]

use multibody_dynamics::math_functions::skew;
use multibody_dynamics::multibody::*;
use nalgebra as na;

#[test]
fn zero_mass_body_contributes_nothing() {
    // Two-body chain, second body zero mass & inertia.
    let offsets = vec![na::Isometry3::identity(), na::Isometry3::identity()];
    let joint_types = vec![JointType::Revolute(Axis::Z), JointType::Revolute(Axis::Z)];
    let parent = vec![0u16, 1u16];
    let mass1 = 2.0;
    let mass2 = 0.0; // zero mass edge case
    let r1 = na::Vector3::new(0.0, 0.0, 0.0);
    let r2 = na::Vector3::zeros();
    let i1 = na::Matrix3::identity() * 0.05 - mass1 * skew(&r1) * skew(&r1);
    let i2 = na::Matrix3::zeros();
    let inertia = vec![i1, i2];
    let masses = vec![mass1, mass2];
    let r_com = vec![r1, r2];
    let mb: MultiBody<2, 2> = MultiBody::new(
        offsets,
        None,
        None,
        Some(inertia),
        joint_types,
        parent,
        na::Vector3::zeros(),
        Some(r_com),
        None,
        Some(masses),
        None,
        None,
    )
    .unwrap();

    let q = na::SVector::<f64, 2>::zeros();
    let six_dof_vars: Vec<na::Isometry3<f64>> = Vec::new();
    let conf = mb.minimal_to_homogeneous_configuration(&six_dof_vars, &q);
    let m_mat = mb.compute_mass_matrix(&conf);

    // Expect lower-right element (second joint) near zero due to zero mass/inertia body.
    assert!(m_mat[(1, 1)] < 1e-12);
}

#[test]
fn try_forward_dynamics_reports_singular_scalar_joint_inertia() {
    let mb: MultiBody<1, 1> = MultiBody::new(
        vec![na::Isometry3::identity()],
        Some(vec![na::SMatrix::<f64, 6, 6>::zeros()]),
        None,
        None,
        vec![JointType::Revolute(Axis::Z)],
        vec![0],
        na::Vector3::zeros(),
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();

    let q = na::SVector::<f64, 1>::zeros();
    let six_dof_vars: Vec<na::Isometry3<f64>> = Vec::new();
    let conf = mb.minimal_to_homogeneous_configuration(&six_dof_vars, &q);
    let mu = na::SVector::<f64, 1>::zeros();
    let rigid_body_forces =
        |_: &[na::Isometry3<f64>], _: &[na::SVector<f64, 6>]| na::SMatrix::<f64, 6, 1>::zeros();
    let thruster_forces = vec![na::SVector::<f64, 6>::zeros()];
    let eta = na::SVector::<f64, 1>::zeros();
    let zero3 = na::Vector3::zeros();

    assert_eq!(
        mb.try_forward_dynamics_ab(
            &conf,
            &mu,
            rigid_body_forces,
            &thruster_forces,
            &eta,
            &zero3,
            &zero3,
        )
        .unwrap_err(),
        "scalar joint matrix inversion failed"
    );
}

#[test]
fn try_forward_dynamics_reports_singular_sixdof_joint_inertia() {
    let mb: MultiBody<1, 6> = MultiBody::new(
        vec![na::Isometry3::identity()],
        Some(vec![na::SMatrix::<f64, 6, 6>::zeros()]),
        None,
        None,
        vec![JointType::SixDOF],
        vec![0],
        na::Vector3::zeros(),
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();

    let conf = vec![na::Isometry3::identity()];
    let mu = na::SVector::<f64, 6>::zeros();
    let rigid_body_forces =
        |_: &[na::Isometry3<f64>], _: &[na::SVector<f64, 6>]| na::SMatrix::<f64, 6, 1>::zeros();
    let thruster_forces = vec![na::SVector::<f64, 6>::zeros()];
    let eta = na::SVector::<f64, 6>::zeros();
    let zero3 = na::Vector3::zeros();

    assert_eq!(
        mb.try_forward_dynamics_ab(
            &conf,
            &mu,
            rigid_body_forces,
            &thruster_forces,
            &eta,
            &zero3,
            &zero3,
        )
        .unwrap_err(),
        "6x6 joint matrix inversion failed"
    );
}

#[test]
#[should_panic(expected = "scalar joint matrix inversion failed")]
fn forward_dynamics_rejects_singular_scalar_joint_inertia() {
    let mb: MultiBody<1, 1> = MultiBody::new(
        vec![na::Isometry3::identity()],
        Some(vec![na::SMatrix::<f64, 6, 6>::zeros()]),
        None,
        None,
        vec![JointType::Revolute(Axis::Z)],
        vec![0],
        na::Vector3::zeros(),
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();

    let q = na::SVector::<f64, 1>::zeros();
    let six_dof_vars: Vec<na::Isometry3<f64>> = Vec::new();
    let conf = mb.minimal_to_homogeneous_configuration(&six_dof_vars, &q);
    let mu = na::SVector::<f64, 1>::zeros();
    let rigid_body_forces =
        |_: &[na::Isometry3<f64>], _: &[na::SVector<f64, 6>]| na::SMatrix::<f64, 6, 1>::zeros();
    let thruster_forces = vec![na::SVector::<f64, 6>::zeros()];
    let eta = na::SVector::<f64, 1>::zeros();
    let zero3 = na::Vector3::zeros();

    mb.forward_dynamics_ab(
        &conf,
        &mu,
        rigid_body_forces,
        &thruster_forces,
        &eta,
        &zero3,
        &zero3,
    );
}
