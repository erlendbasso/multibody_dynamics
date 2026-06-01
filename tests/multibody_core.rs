#![allow(deprecated)]

use approx::assert_relative_eq;
use core::f64::consts::PI;
use multibody_dynamics::math_functions::skew;
use multibody_dynamics::multibody::*;
use na::{Translation3, Vector2};
use nalgebra as na;

type Matrix3 = na::Matrix3<f64>;
type Matrix6 = na::SMatrix<f64, 6, 6>;
type Vector3 = na::Vector3<f64>;
type Vector6 = na::SVector<f64, 6>;

fn comp_mass_matrix(m: f64, r: &Vector3, inertia_mat: &Matrix3) -> Matrix6 {
    let mut mass_matrix = Matrix6::zeros();
    mass_matrix
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(m * Matrix3::identity()));
    mass_matrix
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-m * skew(r)));
    mass_matrix
        .fixed_view_mut::<3, 3>(3, 0)
        .copy_from(&(m * skew(r)));
    mass_matrix
        .fixed_view_mut::<3, 3>(3, 3)
        .copy_from(inertia_mat);
    mass_matrix
}

#[test]
fn gen_newton_euler_test() {
    let l_1 = 1.0;
    let l_2 = 1.0;
    let m_1 = 1.0;
    let m_2 = 1.0;
    let r_cg1 = Vector3::new(l_1 / 2.0, 0.0, 0.0);
    let r_cg2 = Vector3::new(l_2 / 2.0, 0.0, 0.0);

    let p_01 = Translation3::new(0.0, 0.0, 0.0);
    let q_01 = na::UnitQuaternion::identity();
    let p_12 = Translation3::new(l_1, 0.0, 0.0);
    let q_12 = na::UnitQuaternion::identity();
    let offset_matrices = vec![
        na::Isometry3::from_parts(p_01, q_01),
        na::Isometry3::from_parts(p_12, q_12),
    ];

    let i_1 = Matrix3::identity() - m_1 * skew(&r_cg1) * skew(&r_cg1);
    let i_2 = Matrix3::identity() - m_2 * skew(&r_cg2) * skew(&r_cg2);
    let inertia_mats = vec![i_1, i_2];

    let joint_types = vec![JointType::Revolute(Axis::Z), JointType::Revolute(Axis::Z)];
    let parent = vec![0, 1];
    let masses = vec![m_1, m_2];
    let mut r_cg = vec![Vector3::zeros(); 2];
    r_cg[0] = r_cg1;
    r_cg[1] = r_cg2;

    let multibody: MultiBody<2, 2> = MultiBody::new(
        offset_matrices,
        None,
        None,
        Some(inertia_mats),
        joint_types,
        parent,
        Vector3::new(0.0, 0.0, 9.81),
        Some(r_cg),
        None,
        Some(masses),
        None,
        None,
    )
    .unwrap();

    let conf = vec![
        na::Isometry3::identity(),
        na::Isometry3::from_parts(
            Translation3::identity(),
            na::UnitQuaternion::from_euler_angles(0.0, 0.0, PI / 2.0),
        ),
    ];

    let mu = Vector2::new(0.0, 1.0);
    let sigma_prime = Vector2::new(0.0, 0.0);
    let eta = na::SVector::<f64, 2>::zeros();
    let rigid_body_forces_func = &|_x: &[Vector6], _y: &[Vector6]| -> na::SMatrix<f64, 6, 2> {
        na::SMatrix::<f64, 6, 2>::zeros()
    };
    let zeta = multibody.generalized_newton_euler(
        &conf,
        &mu,
        &mu,
        &sigma_prime,
        rigid_body_forces_func,
        &eta,
    );
    assert_relative_eq!(zeta, Vector2::new(-0.5, 0.0), epsilon = 1e-5);
}

#[test]
fn jacobian_and_forward_dynamics_smoke() {
    // Reduced test ensuring jacobians and forward dynamics stay coherent.
    let c1 = na::Isometry3::<f64>::identity();
    let q12 = na::UnitQuaternion::from_axis_angle(&Vector3::x_axis(), -PI / 2.0);
    let q21 = na::UnitQuaternion::from_axis_angle(&Vector3::x_axis(), PI / 2.0);
    let l1 = 0.62;
    let l2 = 0.10;
    let c2 = na::Isometry3::from_parts(
        Translation3::new(l1, 0.0, 0.0),
        na::UnitQuaternion::identity(),
    );
    let c3 = na::Isometry3::from_parts(Translation3::new(l2, 0.0, 0.0), q12);
    let c4 = na::Isometry3::from_parts(Translation3::new(l1, 0.0, 0.0), q21);
    let offset_matrices = vec![c1, c2, c3, c4, c3, c4, c3, c4, c3];
    let mut joint_types = vec![JointType::Revolute(Axis::Z); 9];
    joint_types[0] = JointType::SixDOF;
    let parent: Vec<u16> = (0..9_u16).collect();
    let r_cg1 = Vector3::new(l1 / 2.0, 0.0, 0.0);
    let r_cg2 = Vector3::new(l2 / 2.0, 0.0, 0.0);
    let m1 = PI * 0.09 * 0.09 * l1 * 1000.0;
    let m2 = PI * 0.09 * 0.09 * l2 * 1000.0;
    let inertia_mat1 = Matrix3::new(
        0.5 * m1 * 0.09 * 0.09,
        0.0,
        0.0,
        0.0,
        (1.0 / 12.0) * m1 * (3.0 * 0.09 * 0.09 + l1 * l1),
        0.0,
        0.0,
        0.0,
        (1.0 / 12.0) * m1 * (3.0 * 0.09 * 0.09 + l1 * l1),
    ) - m1 * skew(&r_cg1) * skew(&r_cg1);
    let inertia_mat2 = Matrix3::new(
        0.5 * m2 * 0.09 * 0.09,
        0.0,
        0.0,
        0.0,
        (1.0 / 12.0) * m2 * (3.0 * 0.09 * 0.09 + l2 * l2),
        0.0,
        0.0,
        0.0,
        (1.0 / 12.0) * m2 * (3.0 * 0.09 * 0.09 + l2 * l2),
    ) - m2 * skew(&r_cg2) * skew(&r_cg2);
    let m1_mat = comp_mass_matrix(m1, &r_cg1, &inertia_mat1);
    let m2_mat = comp_mass_matrix(m2, &r_cg2, &inertia_mat2);
    let mass_matrices = vec![
        m1_mat, m2_mat, m1_mat, m2_mat, m1_mat, m2_mat, m1_mat, m2_mat, m1_mat,
    ];
    // Provide mass and center of gravity so hydrostatic force (used internally) has data.
    let masses = vec![m1, m2, m1, m2, m1, m2, m1, m2, m1];
    let r_cg_vec = vec![
        r_cg1, r_cg2, r_cg1, r_cg2, r_cg1, r_cg2, r_cg1, r_cg2, r_cg1,
    ];
    let multibody: MultiBody<9, 14> = MultiBody::new(
        offset_matrices,
        Some(mass_matrices),
        None,
        None,
        joint_types,
        parent,
        Vector3::new(0.0, 0.0, 9.81),
        Some(r_cg_vec),
        None,
        Some(masses),
        None,
        None,
    )
    .unwrap();

    let configuration_base = na::Isometry3::identity();
    let joint_angles = na::SVector::<f64, 8>::from_vec(vec![
        PI / 4.0,
        PI / 3.0,
        PI / 5.0,
        PI / 7.0,
        PI / 15.0,
        PI / 10.0,
        PI / 4.0,
        PI / 2.5,
    ]);
    let conf = multibody.minimal_to_homogeneous_configuration(&configuration_base, &joint_angles);

    let jac = multibody.compute_jacobian(&conf, 7);
    let jacs = multibody.compute_jacobians(&conf);
    assert_relative_eq!(jacs[7].column(6), jac.column(6), epsilon = 1e-8);

    let mu = na::SVector::<f64, 14>::repeat(1.0);
    let djac = multibody.compute_jacobian_derivative(&conf, &mu, 7);
    let djacs = multibody.compute_jacobian_derivatives(&jacs, &conf, &mu);
    for i in 0..6 {
        assert_relative_eq!(djacs[7].column(i), djac.column(i), epsilon = 1e-8);
    }

    // forward dynamics consistency check
    let rigid_body_forces_func2 =
        &|_x: &[na::Isometry3<f64>], _y: &[Vector6]| -> na::SMatrix<f64, 6, 9> {
            na::SMatrix::<f64, 6, 9>::zeros()
        };
    let rigid_body_forces_func1 = &|_x: &[Vector6], _y: &[Vector6]| -> na::SMatrix<f64, 6, 9> {
        na::SMatrix::<f64, 6, 9>::zeros()
    };
    let thruster_forces = vec![Vector6::zeros(); 9];
    let eta = na::SVector::<f64, 14>::zeros();
    let lin_vel_current = Vector3::zeros();
    let lin_accel_current = Vector3::zeros();
    let accel = multibody.forward_dynamics_ab(
        &conf,
        &mu,
        rigid_body_forces_func2,
        &thruster_forces,
        &eta,
        &lin_vel_current,
        &lin_accel_current,
    );
    let sigma_prime = na::SVector::<f64, 14>::zeros();
    let c_vec = multibody.generalized_newton_euler(
        &conf,
        &mu,
        &mu,
        &sigma_prime,
        rigid_body_forces_func1,
        &eta,
    );
    let mass_mat = multibody.compute_mass_matrix(&conf);
    let accel2 = -mass_mat.try_inverse().unwrap() * c_vec;
    assert_relative_eq!(accel, accel2, epsilon = 1e-7);
}
