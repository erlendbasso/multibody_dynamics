#![allow(deprecated)]

use approx::assert_relative_eq;
use multibody_dynamics::math_functions::skew;
use multibody_dynamics::multibody::*;
use nalgebra as na;

fn scalar_link(mass: f64) -> LinkProperties {
    LinkProperties {
        mass: Some(mass),
        r_com: Some(na::Vector3::zeros()),
        inertia3: Some(na::Matrix3::identity()),
        ..LinkProperties::default()
    }
}

fn simple_config<const NUM_BODIES: usize, const NUM_DOFS: usize>(
    joint_types: Vec<JointType>,
    parent: Vec<u16>,
) -> MultiBodyConfig<NUM_BODIES, NUM_DOFS> {
    MultiBodyConfig {
        topology: Topology {
            offset_matrices: vec![na::Isometry3::identity(); NUM_BODIES],
            joint_types,
            parent,
        },
        link_props: Some(vec![scalar_link(1.0); NUM_BODIES]),
        env: Environment::default(),
    }
}

#[test]
fn rejects_wrong_dof_count_before_matrix_indexing() {
    let cfg = simple_config::<1, 1>(vec![JointType::SixDOF], vec![0]);

    let err = MultiBody::<1, 1>::from_config(cfg).err().unwrap();
    assert_eq!(err, "NUM_DOFS must equal sum of joint dimensions");
}

#[test]
fn rejects_zero_body_models() {
    let cfg = MultiBodyConfig::<0, 0> {
        topology: Topology {
            offset_matrices: Vec::new(),
            joint_types: Vec::new(),
            parent: Vec::new(),
        },
        link_props: Some(Vec::new()),
        env: Environment::default(),
    };

    let err = MultiBody::<0, 0>::from_config(cfg).err().unwrap();
    assert_eq!(err, "NUM_BODIES must be greater than zero");
}

#[test]
fn rejects_invalid_parent_topology() {
    let later_parent = simple_config::<2, 2>(vec![JointType::Revolute(Axis::Z); 2], vec![2, 0]);
    assert_eq!(
        MultiBody::<2, 2>::from_config(later_parent).err().unwrap(),
        "parent must reference an earlier body"
    );

    let out_of_range = simple_config::<2, 2>(vec![JointType::Revolute(Axis::Z); 2], vec![0, 3]);
    assert_eq!(
        MultiBody::<2, 2>::from_config(out_of_range).err().unwrap(),
        "parent index out of range"
    );
}

#[test]
fn checked_kinematic_apis_validate_lengths_and_body_ids() {
    let mb = MultiBody::<1, 1>::from_config(simple_config::<1, 1>(
        vec![JointType::Revolute(Axis::Z)],
        vec![0],
    ))
    .unwrap();
    let conf = vec![na::Isometry3::identity()];
    let empty_conf: Vec<na::Isometry3<f64>> = Vec::new();
    let mu = na::SVector::<f64, 1>::zeros();
    let jacs = mb.compute_jacobians(&conf);

    assert_eq!(
        mb.try_compute_mass_matrix(&empty_conf).unwrap_err(),
        "conf length mismatch"
    );
    assert_eq!(
        mb.try_compute_body_configurations(&empty_conf).unwrap_err(),
        "conf length mismatch"
    );
    assert_eq!(
        mb.try_compute_jacobians(&empty_conf).unwrap_err(),
        "conf length mismatch"
    );
    assert_eq!(
        mb.try_compute_jacobian_derivatives(&jacs, &empty_conf, &mu)
            .unwrap_err(),
        "conf length mismatch"
    );

    let empty_jacs: Vec<na::SMatrix<f64, 6, 1>> = Vec::new();
    assert_eq!(
        mb.try_compute_jacobian_derivatives(&empty_jacs, &conf, &mu)
            .unwrap_err(),
        "jacobians length mismatch"
    );
    assert_eq!(
        mb.try_compute_jacobian(&conf, 1).unwrap_err(),
        "body_id out of range"
    );
    assert_eq!(
        mb.try_compute_jacobian_derivative(&conf, &mu, 1)
            .unwrap_err(),
        "body_id out of range"
    );
    assert_eq!(
        mb.try_compute_hydrostatic_force(&na::UnitQuaternion::identity(), &na::Vector3::zeros(), 1)
            .unwrap_err(),
        "body_id out of range"
    );
}

#[test]
fn try_forward_dynamics_validates_slice_lengths() {
    let mb = MultiBody::<1, 1>::from_config(simple_config::<1, 1>(
        vec![JointType::Revolute(Axis::Z)],
        vec![0],
    ))
    .unwrap();
    let rigid_body_forces =
        |_: &[na::Isometry3<f64>], _: &[na::SVector<f64, 6>]| na::SMatrix::<f64, 6, 1>::zeros();
    let conf = vec![na::Isometry3::identity()];
    let empty_conf: Vec<na::Isometry3<f64>> = Vec::new();
    let mu = na::SVector::<f64, 1>::zeros();
    let thruster_forces = vec![na::SVector::<f64, 6>::zeros()];
    let empty_thrusters: Vec<na::SVector<f64, 6>> = Vec::new();
    let eta = na::SVector::<f64, 1>::zeros();
    let zero3 = na::Vector3::zeros();

    assert_eq!(
        mb.try_forward_dynamics_ab(
            &empty_conf,
            &mu,
            rigid_body_forces,
            &thruster_forces,
            &eta,
            &zero3,
            &zero3,
        )
        .unwrap_err(),
        "conf length mismatch"
    );
    assert_eq!(
        mb.try_forward_dynamics_ab(
            &conf,
            &mu,
            rigid_body_forces,
            &empty_thrusters,
            &eta,
            &zero3,
            &zero3,
        )
        .unwrap_err(),
        "thruster_forces length mismatch"
    );
}

#[test]
fn legacy_new_returns_err_for_short_optional_vectors() {
    let result = MultiBody::<2, 2>::new(
        vec![na::Isometry3::identity(); 2],
        Some(vec![na::SMatrix::<f64, 6, 6>::identity()]),
        None,
        None,
        vec![JointType::Revolute(Axis::Z); 2],
        vec![0, 1],
        na::Vector3::zeros(),
        None,
        None,
        None,
        None,
        None,
    );

    assert_eq!(result.err().unwrap(), "mass_matrices length mismatch");
}

#[test]
fn rejects_partial_scalar_mass_properties() {
    let mut cfg = simple_config::<1, 1>(vec![JointType::Revolute(Axis::Z)], vec![0]);
    cfg.link_props = Some(vec![LinkProperties {
        mass: Some(1.0),
        r_com: Some(na::Vector3::zeros()),
        inertia3: None,
        ..LinkProperties::default()
    }]);

    assert_eq!(
        MultiBody::<1, 1>::from_config(cfg).err().unwrap(),
        "mass, r_com, and inertia3 must all be provided together"
    );
}

#[test]
fn legacy_new_preserves_missing_rho_as_zero_density() {
    let mass = 2.0;
    let gravity = na::Vector3::new(0.0, 0.0, -9.81);
    let mb = MultiBody::<1, 1>::new(
        vec![na::Isometry3::identity()],
        None,
        None,
        Some(vec![na::Matrix3::identity()]),
        vec![JointType::Revolute(Axis::Z)],
        vec![0],
        gravity,
        Some(vec![na::Vector3::zeros()]),
        Some(vec![na::Vector3::zeros()]),
        Some(vec![mass]),
        Some(vec![1.0]),
        None,
    )
    .unwrap();

    let force =
        mb.compute_hydrostatic_force(&na::UnitQuaternion::identity(), &na::Vector3::zeros(), 0);

    assert_relative_eq!(force.fixed_rows::<3>(0).into_owned(), mass * gravity);
}

#[test]
fn try_minimal_configuration_validates_input_lengths() {
    let mb = MultiBody::<2, 7>::from_config(simple_config::<2, 7>(
        vec![JointType::SixDOF, JointType::Revolute(Axis::Z)],
        vec![0, 1],
    ))
    .unwrap();

    let no_six_dof_vars: Vec<na::Isometry3<f64>> = Vec::new();
    let scalar = na::SVector::<f64, 1>::zeros();
    assert_eq!(
        mb.try_minimal_to_homogeneous_configuration(&no_six_dof_vars, &scalar)
            .unwrap_err(),
        "six_dof_vars length mismatch"
    );

    let base = na::Isometry3::identity();
    let too_many_scalars = na::SVector::<f64, 2>::zeros();
    assert_eq!(
        mb.try_minimal_to_homogeneous_configuration(&base, &too_many_scalars)
            .unwrap_err(),
        "scalar_joint_vars length mismatch"
    );
}

#[test]
fn mass6_supplies_gravity_mass_properties() {
    let mass = 2.0;
    let r_com = na::Vector3::new(0.1, -0.2, 0.3);
    let mut mass6 = na::SMatrix::<f64, 6, 6>::zeros();
    mass6
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(mass * na::Matrix3::identity()));
    mass6
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-mass * skew(&r_com)));
    mass6
        .fixed_view_mut::<3, 3>(3, 0)
        .copy_from(&(mass * skew(&r_com)));
    mass6
        .fixed_view_mut::<3, 3>(3, 3)
        .copy_from(&na::Matrix3::identity());

    let mb = MultiBody::<1, 6>::from_config(MultiBodyConfig {
        topology: Topology {
            offset_matrices: vec![na::Isometry3::identity()],
            joint_types: vec![JointType::SixDOF],
            parent: vec![0],
        },
        link_props: Some(vec![LinkProperties {
            mass6: Some(mass6),
            ..LinkProperties::default()
        }]),
        env: Environment {
            gravity: na::Vector3::new(0.0, 0.0, -9.81),
            rho: 1025.0,
        },
    })
    .unwrap();

    let force =
        mb.compute_hydrostatic_force(&na::UnitQuaternion::identity(), &na::Vector3::zeros(), 0);
    let expected_linear = mass * na::Vector3::new(0.0, 0.0, -9.81);
    let expected_rotational = mass * skew(&r_com) * na::Vector3::new(0.0, 0.0, -9.81);

    assert_relative_eq!(force.fixed_rows::<3>(0).into_owned(), expected_linear);
    assert_relative_eq!(force.fixed_rows::<3>(3).into_owned(), expected_rotational);
}

#[test]
fn try_regressor_rejects_wrong_joint_shape() {
    let mb = MultiBody::<1, 1>::from_config(simple_config::<1, 1>(
        vec![JointType::Revolute(Axis::Z)],
        vec![0],
    ))
    .unwrap();
    let body = |_: &na::Isometry3<f64>,
                _: &na::SVector<f64, 6>,
                _: &na::SVector<f64, 6>,
                _: &na::SVector<f64, 6>|
     -> na::SMatrix<f64, 6, 2> { na::SMatrix::<f64, 6, 2>::zeros() };
    let joint = |_: &na::Isometry3<f64>,
                 _: JointKinArg,
                 _: JointKinArg,
                 _: JointKinArg|
     -> JointRegressorOut<2> {
        JointRegressorOut::Matrix(na::SMatrix::<f64, 6, 2>::zeros())
    };
    let body_ref: &BodyRegressorFn<2> = &body;
    let joint_ref: &JointRegressorFn<2> = &joint;
    let conf = vec![na::Isometry3::identity()];
    let mu = na::SVector::<f64, 1>::zeros();

    assert_eq!(
        mb.try_compute_regressor_matrix([body_ref], [joint_ref], &conf, &mu, &mu, &mu)
            .unwrap_err(),
        "scalar joint regressor returned Matrix"
    );
}

#[test]
fn try_regressor_validates_configuration_length() {
    let mb = MultiBody::<1, 1>::from_config(simple_config::<1, 1>(
        vec![JointType::Revolute(Axis::Z)],
        vec![0],
    ))
    .unwrap();
    let body = |_: &na::Isometry3<f64>,
                _: &na::SVector<f64, 6>,
                _: &na::SVector<f64, 6>,
                _: &na::SVector<f64, 6>|
     -> na::SMatrix<f64, 6, 2> { na::SMatrix::<f64, 6, 2>::zeros() };
    let joint = |_: &na::Isometry3<f64>,
                 _: JointKinArg,
                 _: JointKinArg,
                 _: JointKinArg|
     -> JointRegressorOut<2> {
        JointRegressorOut::Row(na::SMatrix::<f64, 1, 2>::zeros())
    };
    let body_ref: &BodyRegressorFn<2> = &body;
    let joint_ref: &JointRegressorFn<2> = &joint;
    let conf: Vec<na::Isometry3<f64>> = Vec::new();
    let mu = na::SVector::<f64, 1>::zeros();

    assert_eq!(
        mb.try_compute_regressor_matrix([body_ref], [joint_ref], &conf, &mu, &mu, &mu)
            .unwrap_err(),
        "conf length mismatch"
    );
}
