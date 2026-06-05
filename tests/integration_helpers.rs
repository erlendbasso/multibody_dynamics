use approx::assert_relative_eq;
use multibody_dynamics::math_functions::exp_se3;
use multibody_dynamics::multibody::*;
use nalgebra as na;

type Vector3 = na::Vector3<f64>;
type Vector6 = na::SVector<f64, 6>;

fn scalar_link() -> LinkProperties {
    LinkProperties {
        mass: Some(1.0),
        r_com: Some(Vector3::zeros()),
        inertia3: Some(na::Matrix3::identity()),
        ..LinkProperties::default()
    }
}

fn one_body_model<const NUM_DOFS: usize>(joint_type: JointType) -> MultiBody<1, NUM_DOFS> {
    MultiBody::from_config(MultiBodyConfig {
        topology: Topology {
            offset_matrices: vec![na::Isometry3::identity()],
            joint_types: vec![joint_type],
            parent: vec![0],
        },
        link_props: Some(vec![scalar_link()]),
        env: Environment::default(),
    })
    .unwrap()
}

#[test]
fn semi_implicit_euler_advances_revolute_joint() {
    let model = one_body_model::<1>(JointType::Revolute(Axis::Z));
    let state = DynamicsState::<1, 1> {
        conf: vec![na::Isometry3::identity()],
        mu: na::SVector::<f64, 1>::from_element(1.25),
    };
    let rigid_body_forces = |_: &[na::Isometry3<f64>], _: &[Vector6]| -> na::SMatrix<f64, 6, 1> {
        na::SMatrix::<f64, 6, 1>::zeros()
    };
    let thruster_forces = vec![Vector6::zeros()];
    let eta = na::SVector::<f64, 1>::zeros();
    let zero3 = Vector3::zeros();
    let input = DynamicsStepInput {
        rigid_body_forces: &rigid_body_forces,
        thruster_forces: &thruster_forces,
        eta: &eta,
        lin_vel_current: &zero3,
        lin_accel_current: &zero3,
    };

    let stationary = DynamicsState::<1, 1> {
        conf: vec![na::Isometry3::identity()],
        mu: na::SVector::<f64, 1>::zeros(),
    };
    let stationary_next = model.step_dynamics(
        &stationary,
        input,
        IntegrationOptions {
            dt: 0.2,
            method: IntegrationMethod::SemiImplicitEuler,
        },
    );
    assert_relative_eq!(
        stationary_next.conf[0]
            .rotation
            .angle_to(&na::UnitQuaternion::identity()),
        0.0,
        epsilon = 1e-12
    );

    let next = model.step_dynamics(
        &state,
        input,
        IntegrationOptions {
            dt: 0.2,
            method: IntegrationMethod::SemiImplicitEuler,
        },
    );
    let expected_rotation = na::UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 1.25 * 0.2);

    assert_relative_eq!(
        next.conf[0].rotation.angle_to(&expected_rotation),
        0.0,
        epsilon = 1e-12
    );
    assert_relative_eq!(next.mu, state.mu, epsilon = 1e-12);
}

#[test]
fn semi_implicit_euler_advances_prismatic_joint() {
    let model = one_body_model::<1>(JointType::Prismatic(Axis::X));
    let state = DynamicsState::<1, 1> {
        conf: vec![na::Isometry3::identity()],
        mu: na::SVector::<f64, 1>::from_element(2.0),
    };
    let rigid_body_forces = |_: &[na::Isometry3<f64>], _: &[Vector6]| -> na::SMatrix<f64, 6, 1> {
        na::SMatrix::<f64, 6, 1>::zeros()
    };
    let thruster_forces = vec![Vector6::zeros()];
    let eta = na::SVector::<f64, 1>::zeros();
    let zero3 = Vector3::zeros();
    let input = DynamicsStepInput {
        rigid_body_forces: &rigid_body_forces,
        thruster_forces: &thruster_forces,
        eta: &eta,
        lin_vel_current: &zero3,
        lin_accel_current: &zero3,
    };

    let next = model.step_dynamics(
        &state,
        input,
        IntegrationOptions {
            dt: 0.25,
            method: IntegrationMethod::SemiImplicitEuler,
        },
    );

    assert_relative_eq!(next.conf[0].translation.vector, Vector3::new(0.5, 0.0, 0.0));
    assert_relative_eq!(next.mu, state.mu, epsilon = 1e-12);
}

#[test]
fn six_dof_step_advances_body_frame_linear_and_angular_velocity() {
    let model = one_body_model::<6>(JointType::SixDOF);
    let rigid_body_forces = |_: &[na::Isometry3<f64>], _: &[Vector6]| -> na::SMatrix<f64, 6, 1> {
        na::SMatrix::<f64, 6, 1>::zeros()
    };
    let thruster_forces = vec![Vector6::zeros()];
    let eta = na::SVector::<f64, 6>::zeros();
    let zero3 = Vector3::zeros();
    let input = DynamicsStepInput {
        rigid_body_forces: &rigid_body_forces,
        thruster_forces: &thruster_forces,
        eta: &eta,
        lin_vel_current: &zero3,
        lin_accel_current: &zero3,
    };

    let mut linear_mu = na::SVector::<f64, 6>::zeros();
    linear_mu[0] = 1.0;
    let linear_state = DynamicsState::<1, 6> {
        conf: vec![na::Isometry3::identity()],
        mu: linear_mu,
    };
    let linear_next = model.step_dynamics(
        &linear_state,
        input,
        IntegrationOptions {
            dt: 0.25,
            method: IntegrationMethod::Rk4,
        },
    );
    assert_relative_eq!(
        linear_next.conf[0].translation.vector,
        Vector3::new(0.25, 0.0, 0.0),
        epsilon = 1e-12
    );
    assert_relative_eq!(
        linear_next.conf[0]
            .rotation
            .angle_to(&na::UnitQuaternion::identity()),
        0.0,
        epsilon = 1e-12
    );

    let mut angular_mu = na::SVector::<f64, 6>::zeros();
    angular_mu[5] = 2.0;
    let angular_state = DynamicsState::<1, 6> {
        conf: vec![na::Isometry3::identity()],
        mu: angular_mu,
    };
    let angular_next = model.step_dynamics(
        &angular_state,
        input,
        IntegrationOptions {
            dt: 0.25,
            method: IntegrationMethod::Rk4,
        },
    );
    let expected_rotation = na::UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 2.0 * 0.25);
    assert_relative_eq!(
        angular_next.conf[0].rotation.angle_to(&expected_rotation),
        0.0,
        epsilon = 1e-12
    );
    assert_relative_eq!(
        angular_next.conf[0].rotation.quaternion().norm(),
        1.0,
        epsilon = 1e-12
    );
}

#[test]
fn euler_velocity_update_matches_forward_dynamics() {
    let model = one_body_model::<1>(JointType::Revolute(Axis::Z));
    let state = DynamicsState::<1, 1> {
        conf: vec![na::Isometry3::identity()],
        mu: na::SVector::<f64, 1>::zeros(),
    };
    let rigid_body_forces = |_: &[na::Isometry3<f64>], _: &[Vector6]| -> na::SMatrix<f64, 6, 1> {
        na::SMatrix::<f64, 6, 1>::zeros()
    };
    let thruster_forces = vec![Vector6::zeros()];
    let eta = na::SVector::<f64, 1>::from_element(2.0);
    let zero3 = Vector3::zeros();
    let input = DynamicsStepInput {
        rigid_body_forces: &rigid_body_forces,
        thruster_forces: &thruster_forces,
        eta: &eta,
        lin_vel_current: &zero3,
        lin_accel_current: &zero3,
    };
    let acceleration = model.forward_dynamics_ab(
        &state.conf,
        &state.mu,
        rigid_body_forces,
        &thruster_forces,
        &eta,
        &zero3,
        &zero3,
    );
    let dt = 0.125;

    let next = model.step_dynamics(
        &state,
        input,
        IntegrationOptions {
            dt,
            method: IntegrationMethod::SemiImplicitEuler,
        },
    );

    assert_relative_eq!(next.mu, state.mu + dt * acceleration, epsilon = 1e-12);
}

#[test]
fn rk4_matches_constant_velocity_scalar_motion() {
    let model = one_body_model::<1>(JointType::Revolute(Axis::Z));
    let state = DynamicsState::<1, 1> {
        conf: vec![na::Isometry3::identity()],
        mu: na::SVector::<f64, 1>::from_element(0.75),
    };
    let rigid_body_forces = |_: &[na::Isometry3<f64>], _: &[Vector6]| -> na::SMatrix<f64, 6, 1> {
        na::SMatrix::<f64, 6, 1>::zeros()
    };
    let thruster_forces = vec![Vector6::zeros()];
    let eta = na::SVector::<f64, 1>::zeros();
    let zero3 = Vector3::zeros();
    let input = DynamicsStepInput {
        rigid_body_forces: &rigid_body_forces,
        thruster_forces: &thruster_forces,
        eta: &eta,
        lin_vel_current: &zero3,
        lin_accel_current: &zero3,
    };

    let next = model.step_dynamics(
        &state,
        input,
        IntegrationOptions {
            dt: 0.4,
            method: IntegrationMethod::Rk4,
        },
    );
    let expected_rotation = na::UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.75 * 0.4);

    assert_relative_eq!(
        next.conf[0].rotation.angle_to(&expected_rotation),
        0.0,
        epsilon = 1e-12
    );
    assert_relative_eq!(next.mu, state.mu, epsilon = 1e-12);
}

#[test]
fn rk4_six_dof_composes_noncommuting_stage_twists() {
    let model = one_body_model::<6>(JointType::SixDOF);
    let rigid_body_forces = |_: &[na::Isometry3<f64>], _: &[Vector6]| -> na::SMatrix<f64, 6, 1> {
        na::SMatrix::<f64, 6, 1>::zeros()
    };
    let thruster_forces = vec![Vector6::zeros()];
    let mut eta = na::SVector::<f64, 6>::zeros();
    eta[4] = 3.0;
    let zero3 = Vector3::zeros();
    let input = DynamicsStepInput {
        rigid_body_forces: &rigid_body_forces,
        thruster_forces: &thruster_forces,
        eta: &eta,
        lin_vel_current: &zero3,
        lin_accel_current: &zero3,
    };
    let mut initial_mu = na::SVector::<f64, 6>::zeros();
    initial_mu[3] = 1.5;
    initial_mu[5] = 0.7;
    let state = DynamicsState::<1, 6> {
        conf: vec![na::Isometry3::identity()],
        mu: initial_mu,
    };
    let dt = 0.4;

    let k1_mu = model.forward_dynamics_ab(
        &state.conf,
        &state.mu,
        rigid_body_forces,
        &thruster_forces,
        &eta,
        &zero3,
        &zero3,
    );
    let k1_conf_velocity = state.mu;
    let state2 = DynamicsState::<1, 6> {
        conf: vec![state.conf[0] * exp_se3(&(k1_conf_velocity * (0.5 * dt)))],
        mu: state.mu + 0.5 * dt * k1_mu,
    };
    let k2_mu = model.forward_dynamics_ab(
        &state2.conf,
        &state2.mu,
        rigid_body_forces,
        &thruster_forces,
        &eta,
        &zero3,
        &zero3,
    );
    let k2_conf_velocity = state2.mu;
    let state3 = DynamicsState::<1, 6> {
        conf: vec![state.conf[0] * exp_se3(&(k2_conf_velocity * (0.5 * dt)))],
        mu: state.mu + 0.5 * dt * k2_mu,
    };
    let k3_mu = model.forward_dynamics_ab(
        &state3.conf,
        &state3.mu,
        rigid_body_forces,
        &thruster_forces,
        &eta,
        &zero3,
        &zero3,
    );
    let k3_conf_velocity = state3.mu;
    let state4 = DynamicsState::<1, 6> {
        conf: vec![state.conf[0] * exp_se3(&(k3_conf_velocity * dt))],
        mu: state.mu + dt * k3_mu,
    };
    let _k4_mu = model.forward_dynamics_ab(
        &state4.conf,
        &state4.mu,
        rigid_body_forces,
        &thruster_forces,
        &eta,
        &zero3,
        &zero3,
    );
    let k4_conf_velocity = state4.mu;

    let expected_delta = exp_se3(&(k1_conf_velocity * (dt / 6.0)))
        * exp_se3(&(k2_conf_velocity * (dt / 3.0)))
        * exp_se3(&(k3_conf_velocity * (dt / 3.0)))
        * exp_se3(&(k4_conf_velocity * (dt / 6.0)));
    let averaged_velocity =
        (k1_conf_velocity + 2.0 * k2_conf_velocity + 2.0 * k3_conf_velocity + k4_conf_velocity)
            / 6.0;
    let averaged_delta = exp_se3(&(averaged_velocity * dt));

    let next = model.step_dynamics(
        &state,
        input,
        IntegrationOptions {
            dt,
            method: IntegrationMethod::Rk4,
        },
    );

    assert!(
        expected_delta.rotation.angle_to(&averaged_delta.rotation) > 1e-4,
        "test setup must exercise non-commuting stage rotations"
    );
    assert_relative_eq!(
        next.conf[0].rotation.angle_to(&expected_delta.rotation),
        0.0,
        epsilon = 1e-12
    );
    assert_relative_eq!(
        next.conf[0].translation.vector,
        expected_delta.translation.vector,
        epsilon = 1e-12
    );
}

#[test]
fn try_step_dynamics_validates_inputs() {
    let model = one_body_model::<1>(JointType::Revolute(Axis::Z));
    let rigid_body_forces = |_: &[na::Isometry3<f64>], _: &[Vector6]| -> na::SMatrix<f64, 6, 1> {
        na::SMatrix::<f64, 6, 1>::zeros()
    };
    let thruster_forces = vec![Vector6::zeros()];
    let eta = na::SVector::<f64, 1>::zeros();
    let zero3 = Vector3::zeros();
    let input = DynamicsStepInput {
        rigid_body_forces: &rigid_body_forces,
        thruster_forces: &thruster_forces,
        eta: &eta,
        lin_vel_current: &zero3,
        lin_accel_current: &zero3,
    };
    let valid_state = DynamicsState::<1, 1> {
        conf: vec![na::Isometry3::identity()],
        mu: na::SVector::<f64, 1>::zeros(),
    };

    assert_eq!(
        model
            .try_step_dynamics(
                &valid_state,
                input,
                IntegrationOptions {
                    dt: f64::NAN,
                    method: IntegrationMethod::SemiImplicitEuler,
                },
            )
            .unwrap_err(),
        "dt must be finite and non-negative"
    );

    let bad_conf_state = DynamicsState::<1, 1> {
        conf: Vec::new(),
        mu: na::SVector::<f64, 1>::zeros(),
    };
    assert_eq!(
        model
            .try_step_dynamics(
                &bad_conf_state,
                input,
                IntegrationOptions {
                    dt: 0.1,
                    method: IntegrationMethod::SemiImplicitEuler,
                },
            )
            .unwrap_err(),
        "conf length mismatch"
    );

    let empty_thrusters: Vec<Vector6> = Vec::new();
    let bad_input = DynamicsStepInput {
        rigid_body_forces: &rigid_body_forces,
        thruster_forces: &empty_thrusters,
        eta: &eta,
        lin_vel_current: &zero3,
        lin_accel_current: &zero3,
    };
    assert_eq!(
        model
            .try_step_dynamics(
                &valid_state,
                bad_input,
                IntegrationOptions {
                    dt: 0.1,
                    method: IntegrationMethod::SemiImplicitEuler,
                },
            )
            .unwrap_err(),
        "thruster_forces length mismatch"
    );
}
