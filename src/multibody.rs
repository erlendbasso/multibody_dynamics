extern crate nalgebra as na;
use crate::math_functions::*;
use na::{
    Isometry3, Matrix1, Matrix3, Matrix4, Matrix6, Quaternion, SMatrix, SVector, Translation3,
    UnitQuaternion, Vector1, Vector3, Vector6, U6,
};

// use num::{One, Zero};

#[derive(Clone, Debug)]
pub enum Axis {
    X,
    Y,
    Z,
}

#[derive(Clone, Debug)]
pub enum JointType {
    Revolute(Axis),
    Prismatic(Axis),
    SixDOF,
}

fn joint_dim(joint_type: &JointType) -> usize {
    match joint_type {
        JointType::Revolute(_) | JointType::Prismatic(_) => 1,
        JointType::SixDOF => 6,
    }
}

fn mass_properties_from_mass6(mass6: &Matrix6<f64>) -> (f64, Vector3<f64>) {
    let mass = mass6.fixed_view::<3, 3>(0, 0).trace() / 3.0;
    let r_com = if mass.abs() > f64::EPSILON {
        let skew_r = -mass6.fixed_view::<3, 3>(0, 3).into_owned() / mass;
        Vector3::new(skew_r[(2, 1)], skew_r[(0, 2)], skew_r[(1, 0)])
    } else {
        Vector3::zeros()
    };
    (mass, r_com)
}

/// System topology: kinematics-only data
#[derive(Clone, Debug)]
pub struct Topology {
    pub offset_matrices: Vec<Isometry3<f64>>, // h_i offset transforms
    pub joint_types: Vec<JointType>,
    pub parent: Vec<u16>,
}

/// Per-link physical properties (mass/inertia and hydro-related values)
#[derive(Clone, Debug, Default)]
pub struct LinkProperties {
    /// Optional full 6x6 mass matrix (if provided, it’s the base to use)
    pub mass6: Option<Matrix6<f64>>,
    /// Optional added mass 6x6 matrix to be added to base mass
    pub added_mass6: Option<Matrix6<f64>>,
    /// Optional scalar mass, center of mass, and 3x3 inertia (used if mass6 is None)
    pub mass: Option<f64>,
    pub r_com: Option<Vector3<f64>>,
    pub inertia3: Option<Matrix3<f64>>,
    /// Hydro-related properties
    pub volume: Option<f64>,
    pub r_cob: Option<Vector3<f64>>,
}

/// Environment parameters
#[derive(Clone, Debug)]
pub struct Environment {
    pub gravity: Vector3<f64>,
    pub rho: f64,
}

impl Default for Environment {
    fn default() -> Self {
        Environment {
            gravity: Vector3::zeros(),
            rho: 1025.0,
        }
    }
}

/// All inputs needed to construct a MultiBody in a single object
#[derive(Clone, Debug)]
pub struct MultiBodyConfig<const NUM_BODIES: usize, const NUM_DOFS: usize> {
    pub topology: Topology,
    /// Optional per-link properties; defaults to zeroed properties when None
    pub link_props: Option<Vec<LinkProperties>>,
    pub env: Environment,
}

// Intentionally no builder: prefer a single config parameter for simplicity

/// Callback type for a per-body regressor W_i.
///
/// Arguments:
/// - pose_world: world→body_i transform g_i (absolute link pose), i.e. compute_body_configurations(conf)[i].
/// - nu: body velocity, expressed in the body frame.
/// - nu_bar: desired body velocity, expressed in the body frame.
/// - alpha_bar: body acceleration, expressed in the body frame.
///
/// Returns:
/// - 6×NUM_PARAMS regressor W_i expressed in the body i frame.
pub type BodyRegressorFn<'a, const NUM_PARAMS: usize> = dyn Fn(
        &Isometry3<f64>, // pose_world = g_i
        &Vector6<f64>,   // nu
        &Vector6<f64>,   // nu_bar
        &Vector6<f64>,   // alpha_bar
    ) -> SMatrix<f64, 6, NUM_PARAMS>
    + 'a;

#[derive(Clone, Debug)]
pub enum JointKinArg {
    Scalar(f64),
    SixDOF(Vector6<f64>),
}

/// Return type for joint regressors: 1xP for scalar joints or 6xP for 6-DoF joints.
#[derive(Clone, Debug)]
pub enum JointRegressorOut<const NUM_PARAMS: usize> {
    Row(SMatrix<f64, 1, NUM_PARAMS>),
    Matrix(SMatrix<f64, 6, NUM_PARAMS>),
}

pub type JointRegressorFn<'a, const NUM_PARAMS: usize> = dyn Fn(
        &Isometry3<f64>, // joint_pose_local = conf[i]
        JointKinArg,     // mu
        JointKinArg,     // mu_bar
        JointKinArg,     // sigma_bar
    ) -> JointRegressorOut<NUM_PARAMS>
    + 'a;

/// Callback type for spatial forces applied to each body during forward dynamics.
///
/// The first slice contains the relative body transforms used by the articulated-body
/// recursion: `h[i] = offset_matrices[i] * conf[i]`. Each `h[i]` transforms body `i`
/// coordinates into its parent coordinates, or into the inertial root coordinates for
/// root bodies. Equivalently, `Ad_inv(&h[i])` maps parent-frame spatial vectors into
/// body `i`'s frame. These are not accumulated world poses.
///
/// The second slice contains `nu[i]`, the spatial velocity of body `i` expressed in
/// body `i`'s frame and ordered `[linear; angular]`. The returned matrix column `i`
/// is the external spatial force applied to body `i`, expressed in body `i`'s frame
/// and ordered `[force; torque]`.
pub type RigidBodyForcesFn<'a, const NUM_BODIES: usize> =
    dyn Fn(&[Isometry3<f64>], &[Vector6<f64>]) -> SMatrix<f64, 6, NUM_BODIES> + 'a;

/// Allows overloading of functions for both a single 6DOF configuration and for a vector of 6DOF configurations, which is required when there are more than one 6DOF joint in the multibody system.
pub trait IntoHomogeneousConfigurationVec {
    fn into(&self) -> Vec<Isometry3<f64>>;
}

impl IntoHomogeneousConfigurationVec for Isometry3<f64> {
    fn into(&self) -> Vec<Isometry3<f64>> {
        vec![*self]
    }
}

impl IntoHomogeneousConfigurationVec for Vec<Isometry3<f64>> {
    fn into(&self) -> Vec<Isometry3<f64>> {
        self.clone()
    }
}

pub struct MultiBody<const NUM_BODIES: usize, const NUM_DOFS: usize> {
    offset_matrices: Vec<Isometry3<f64>>,
    mass_matrices: Vec<Matrix6<f64>>,
    joint_types: Vec<JointType>,
    parent: Vec<u16>,
    // For each body j, ancestors[j] contains its strict ancestors in root->...->parent order.
    ancestors: Vec<Vec<usize>>,
    Phi: SMatrix<f64, 6, NUM_DOFS>,
    joint_dims: SVector<usize, NUM_BODIES>,
    joint_size_offsets: Vec<usize>,
    gravity: Vector3<f64>,
    r_com: Option<Vec<Vector3<f64>>>,
    mass: Option<Vec<f64>>,
    r_cob: Option<Vec<Vector3<f64>>>,
    volume: Option<Vec<f64>>,
    rho: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct DynamicsState<const NUM_BODIES: usize, const NUM_DOFS: usize> {
    /// Per-joint homogeneous configurations in topology order.
    pub conf: Vec<Isometry3<f64>>,
    /// Generalized velocity vector.
    pub mu: SVector<f64, NUM_DOFS>,
}

#[derive(Clone, Copy)]
pub struct DynamicsStepInput<'a, const NUM_BODIES: usize, const NUM_DOFS: usize> {
    /// Callback returning body-frame spatial forces for each body.
    pub rigid_body_forces: &'a RigidBodyForcesFn<'a, NUM_BODIES>,
    /// Per-body spatial thruster forces.
    pub thruster_forces: &'a [Vector6<f64>],
    /// Generalized effort input.
    pub eta: &'a SVector<f64, NUM_DOFS>,
    /// Ambient/current linear velocity used by the hydrodynamic forward-dynamics terms.
    pub lin_vel_current: &'a Vector3<f64>,
    /// Ambient/current linear acceleration used by the hydrodynamic forward-dynamics terms.
    pub lin_accel_current: &'a Vector3<f64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IntegrationMethod {
    /// Update velocity first, then advance configuration with the new velocity.
    SemiImplicitEuler,
    /// Fourth-order Runge-Kutta velocity update.
    ///
    /// Scalar joint configurations use the RK4 weighted velocity. `SixDOF` joint poses use ordered
    /// stage exponential composition for their body-frame twists.
    Rk4,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntegrationOptions {
    /// Integration timestep in seconds.
    pub dt: f64,
    /// Integration scheme to use for this step.
    pub method: IntegrationMethod,
}

#[derive(Clone, Debug)]
pub struct ForwardDynamicsWorkspace<const NUM_BODIES: usize> {
    h: Vec<Isometry3<f64>>,
    Ad_h_inv_cache: Vec<Matrix6<f64>>,
    nu: Vec<Vector6<f64>>,
    alpha: Vec<Vector6<f64>>,
    a_e: Vec<Vector3<f64>>,
    b: Vec<Vector6<f64>>,
    M_a: Vec<Matrix6<f64>>,
    v_inv_scalar: Vec<f64>,
    v_inv_matrix: Vec<Matrix6<f64>>,
    U_scalar: Vec<Vector6<f64>>,
    U_matrix: Vec<Matrix6<f64>>,
    u_scalar: Vec<f64>,
    u_matrix: Vec<Vector6<f64>>,
    joint_is_sixdof: Vec<bool>,
}

impl<const NUM_BODIES: usize> ForwardDynamicsWorkspace<NUM_BODIES> {
    pub fn new() -> Self {
        Self {
            h: vec![Isometry3::<f64>::identity(); NUM_BODIES],
            Ad_h_inv_cache: vec![Matrix6::<f64>::zeros(); NUM_BODIES],
            nu: vec![Vector6::<f64>::zeros(); NUM_BODIES],
            alpha: vec![Vector6::<f64>::zeros(); NUM_BODIES],
            a_e: vec![Vector3::<f64>::zeros(); NUM_BODIES],
            b: vec![Vector6::<f64>::zeros(); NUM_BODIES],
            M_a: vec![Matrix6::<f64>::zeros(); NUM_BODIES],
            v_inv_scalar: vec![0.0; NUM_BODIES],
            v_inv_matrix: vec![Matrix6::<f64>::zeros(); NUM_BODIES],
            U_scalar: vec![Vector6::<f64>::zeros(); NUM_BODIES],
            U_matrix: vec![Matrix6::<f64>::zeros(); NUM_BODIES],
            u_scalar: vec![0.0; NUM_BODIES],
            u_matrix: vec![Vector6::<f64>::zeros(); NUM_BODIES],
            joint_is_sixdof: vec![false; NUM_BODIES],
        }
    }
}

impl<const NUM_BODIES: usize> Default for ForwardDynamicsWorkspace<NUM_BODIES> {
    fn default() -> Self {
        Self::new()
    }
}

// impl<T: na::RealField  + na::ClosedAdd + na::ClosedMul + na::ClosedDiv + Copy, const NUM_BODIES: usize, const NUM_DOFS: usize> MultiBody<T, NUM_BODIES, NUM_DOFS> {
impl<const NUM_BODIES: usize, const NUM_DOFS: usize> TryFrom<MultiBodyConfig<NUM_BODIES, NUM_DOFS>>
    for MultiBody<NUM_BODIES, NUM_DOFS>
{
    type Error = &'static str;

    fn try_from(cfg: MultiBodyConfig<NUM_BODIES, NUM_DOFS>) -> Result<Self, Self::Error> {
        // Validate basic sizes
        let nb = NUM_BODIES;
        if nb == 0 {
            return Err("NUM_BODIES must be greater than zero");
        }
        if cfg.topology.offset_matrices.len() != nb {
            return Err("offset_matrices length mismatch");
        }
        if cfg.topology.joint_types.len() != nb {
            return Err("joint_types length mismatch");
        }
        if cfg.topology.parent.len() != nb {
            return Err("parent length mismatch");
        }
        for (i, parent_entry) in cfg.topology.parent.iter().enumerate() {
            if *parent_entry == 0 {
                continue;
            }
            let parent_idx = usize::from(*parent_entry - 1);
            if parent_idx >= nb {
                return Err("parent index out of range");
            }
            if parent_idx >= i {
                return Err("parent must reference an earlier body");
            }
        }
        let expected_dofs: usize = cfg.topology.joint_types.iter().map(joint_dim).sum();
        if expected_dofs != NUM_DOFS {
            return Err("NUM_DOFS must equal sum of joint dimensions");
        }

        let MultiBodyConfig {
            topology,
            link_props,
            env,
        } = cfg;
        let Topology {
            offset_matrices,
            joint_types,
            parent,
        } = topology;

        let mut joint_dims = SVector::<usize, NUM_BODIES>::zeros();
        let mut Phi = SMatrix::<f64, 6, NUM_DOFS>::zeros();
        let mut joint_size_offsets = 0;
        let mut joint_offset_vec = vec![0; NUM_BODIES];

        for i in 0..NUM_BODIES {
            joint_offset_vec[i] = joint_size_offsets;

            match &joint_types[i] {
                JointType::Revolute(axis) => {
                    let Phi_i = match axis {
                        Axis::X => Vector6::new(0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
                        Axis::Y => Vector6::new(0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                        Axis::Z => Vector6::new(0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
                    };
                    Phi.fixed_view_mut::<6, 1>(0, i + joint_size_offsets)
                        .copy_from(&Phi_i);
                    joint_dims[i] = 1;
                }
                JointType::Prismatic(axis) => {
                    let Phi_i = match axis {
                        Axis::X => Vector6::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                        Axis::Y => Vector6::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
                        Axis::Z => Vector6::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                    };
                    Phi.fixed_view_mut::<6, 1>(0, i + joint_size_offsets)
                        .copy_from(&Phi_i);
                    joint_dims[i] = 1;
                }
                JointType::SixDOF => {
                    Phi.fixed_view_mut::<6, 6>(0, i + joint_size_offsets)
                        .copy_from(&Matrix6::identity());
                    joint_dims[i] = 6;

                    joint_size_offsets += joint_dims[i] - 1;
                }
            }
        }

        let link_props = link_props.unwrap_or_else(|| vec![LinkProperties::default(); NUM_BODIES]);
        if link_props.len() != nb {
            return Err("link_props length mismatch");
        }
        // Build mass matrices and scalar properties from per-link properties
        let mut mass_matrices = vec![Matrix6::zeros(); NUM_BODIES];
        let mut mass_vec = vec![0.0f64; NUM_BODIES];
        let mut r_com_vec = vec![Vector3::zeros(); NUM_BODIES];
        let mut volume_vec = vec![0.0f64; NUM_BODIES];
        let mut r_cob_vec = vec![Vector3::zeros(); NUM_BODIES];
        for i in 0..NUM_BODIES {
            let lp = &link_props[i];
            // base mass
            let mut mm = if let Some(m6) = lp.mass6 {
                let (mass_from_m6, r_com_from_m6) = mass_properties_from_mass6(&m6);
                mass_vec[i] = lp.mass.unwrap_or(mass_from_m6);
                r_com_vec[i] = lp.r_com.unwrap_or(r_com_from_m6);
                m6
            } else {
                match (lp.mass, lp.r_com, lp.inertia3) {
                    (Some(m), Some(rc), Some(inertia3)) => {
                        mass_vec[i] = m;
                        r_com_vec[i] = rc;
                        let mut mass_matrix = Matrix6::zeros();
                        mass_matrix
                            .fixed_view_mut::<3, 3>(0, 0)
                            .copy_from(&(m * Matrix3::identity()));
                        mass_matrix
                            .fixed_view_mut::<3, 3>(0, 3)
                            .copy_from(&(-m * skew(&rc)));
                        mass_matrix
                            .fixed_view_mut::<3, 3>(3, 0)
                            .copy_from(&(m * skew(&rc)));
                        mass_matrix
                            .fixed_view_mut::<3, 3>(3, 3)
                            .copy_from(&inertia3);
                        mass_matrix
                    }
                    (None, None, None) => Matrix6::zeros(),
                    _ => return Err("mass, r_com, and inertia3 must all be provided together"),
                }
            };
            if let Some(am) = lp.added_mass6 {
                mm += am;
            }
            mass_matrices[i] = mm;
            // hydro props
            if let Some(v) = lp.volume {
                volume_vec[i] = v;
            }
            if let Some(rcb) = lp.r_cob {
                r_cob_vec[i] = rcb;
            }
        }

        let parent_vec = parent; // rename for clarity
        let ancestors_build = {
            let mut anc: Vec<Vec<usize>> = vec![Vec::new(); NUM_BODIES];
            for j in 0..NUM_BODIES {
                let mut p = (parent_vec[j] as i32) - 1;
                while p >= 0 {
                    anc[j].push(p as usize);
                    p = (parent_vec[p as usize] as i32) - 1;
                }
                anc[j].reverse();
            }
            anc
        };
        // Pull hydro fields from link_props and env
        let volume = Some(volume_vec);
        let r_cob = Some(r_cob_vec);
        let rho = Some(env.rho);

        Ok(MultiBody {
            offset_matrices,
            mass_matrices,
            joint_types,
            parent: parent_vec.clone(),
            ancestors: ancestors_build,
            Phi,
            joint_dims,
            joint_size_offsets: joint_offset_vec,
            gravity: env.gravity,
            r_com: Some(r_com_vec),
            mass: Some(mass_vec),
            r_cob,
            volume,
            rho,
        })
    }
}

impl<const NUM_BODIES: usize, const NUM_DOFS: usize> MultiBody<NUM_BODIES, NUM_DOFS> {
    /// Preferred constructor: provide a single configuration object
    pub fn from_config(cfg: MultiBodyConfig<NUM_BODIES, NUM_DOFS>) -> Result<Self, &'static str> {
        Self::try_from(cfg)
    }
    #[deprecated(note = "Use MultiBody::from_config(...) or TryFrom<MultiBodyConfig>")]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        offset_matrices: Vec<Isometry3<f64>>,
        mass_matrices: Option<Vec<Matrix6<f64>>>,
        added_mass: Option<Vec<Matrix6<f64>>>,
        inertia_matrices: Option<Vec<Matrix3<f64>>>,
        joint_types: Vec<JointType>,
        parent: Vec<u16>,
        gravity: Vector3<f64>,
        r_com: Option<Vec<Vector3<f64>>>,
        r_cob: Option<Vec<Vector3<f64>>>,
        mass: Option<Vec<f64>>,
        volume: Option<Vec<f64>>,
        rho: Option<f64>,
    ) -> Result<MultiBody<NUM_BODIES, NUM_DOFS>, &'static str> {
        if let Some(values) = &mass_matrices {
            if values.len() != NUM_BODIES {
                return Err("mass_matrices length mismatch");
            }
        }
        if let Some(values) = &added_mass {
            if values.len() != NUM_BODIES {
                return Err("added_mass length mismatch");
            }
        }
        if let Some(values) = &inertia_matrices {
            if values.len() != NUM_BODIES {
                return Err("inertia_matrices length mismatch");
            }
        }
        if let Some(values) = &r_com {
            if values.len() != NUM_BODIES {
                return Err("r_com length mismatch");
            }
        }
        if let Some(values) = &r_cob {
            if values.len() != NUM_BODIES {
                return Err("r_cob length mismatch");
            }
        }
        if let Some(values) = &mass {
            if values.len() != NUM_BODIES {
                return Err("mass length mismatch");
            }
        }
        if let Some(values) = &volume {
            if values.len() != NUM_BODIES {
                return Err("volume length mismatch");
            }
        }
        // Build config from legacy arguments and delegate
        let topology = Topology {
            offset_matrices,
            joint_types,
            parent,
        };
        // Build per-link properties from legacy args
        let mut link_props = Vec::with_capacity(NUM_BODIES);
        for i in 0..NUM_BODIES {
            let lp = LinkProperties {
                mass6: mass_matrices.as_ref().map(|v| v[i]),
                added_mass6: added_mass.as_ref().map(|v| v[i]),
                mass: mass.as_ref().map(|v| v[i]),
                r_com: r_com.as_ref().map(|v| v[i]),
                inertia3: inertia_matrices.as_ref().map(|v| v[i]),
                volume: volume.as_ref().map(|v| v[i]),
                r_cob: r_cob.as_ref().map(|v| v[i]),
            };
            link_props.push(lp);
        }
        let env = Environment {
            gravity,
            rho: rho.unwrap_or(1025.0),
        };
        let cfg = MultiBodyConfig::<NUM_BODIES, NUM_DOFS> {
            topology,
            link_props: Some(link_props),
            env,
        };
        let mut multibody = MultiBody::<NUM_BODIES, NUM_DOFS>::try_from(cfg)?;
        multibody.rho = rho;
        Ok(multibody)
    }

    /// Converts a set of minimal coordinates to a set of homogenous coordinates.
    pub fn minimal_to_homogeneous_configuration<Configuration, const D: usize>(
        &self,
        six_dof_vars: &Configuration,
        scalar_joint_vars: &SVector<f64, D>,
    ) -> Vec<Isometry3<f64>>
    where
        Configuration: IntoHomogeneousConfigurationVec,
    {
        self.try_minimal_to_homogeneous_configuration(six_dof_vars, scalar_joint_vars)
            .expect("configuration coordinate count does not match topology")
    }

    /// Checked variant of [`minimal_to_homogeneous_configuration`].
    pub fn try_minimal_to_homogeneous_configuration<Configuration, const D: usize>(
        &self,
        six_dof_vars: &Configuration,
        scalar_joint_vars: &SVector<f64, D>,
    ) -> Result<Vec<Isometry3<f64>>, &'static str>
    where
        Configuration: IntoHomogeneousConfigurationVec,
    {
        let six_dof_vars = six_dof_vars.into();
        let expected_six_dof = self
            .joint_types
            .iter()
            .filter(|joint_type| matches!(joint_type, JointType::SixDOF))
            .count();
        if six_dof_vars.len() != expected_six_dof {
            return Err("six_dof_vars length mismatch");
        }
        let expected_scalar_dofs = self
            .joint_types
            .iter()
            .filter(|joint_type| !matches!(joint_type, JointType::SixDOF))
            .map(joint_dim)
            .sum::<usize>();
        if D != expected_scalar_dofs {
            return Err("scalar_joint_vars length mismatch");
        }
        let mut j = 0;
        let mut k = 0;

        let mut conf: Vec<Isometry3<f64>> = vec![Isometry3::identity(); NUM_BODIES];

        for (i, conf_i) in conf.iter_mut().enumerate().take(NUM_BODIES) {
            match &self.joint_types[i] {
                JointType::Revolute(axis) => {
                    let mut temp = Isometry3::identity();
                    temp.rotation = match axis {
                        Axis::X => UnitQuaternion::from_axis_angle(
                            &Vector3::x_axis(),
                            scalar_joint_vars[j],
                        ),
                        Axis::Y => UnitQuaternion::from_axis_angle(
                            &Vector3::y_axis(),
                            scalar_joint_vars[j],
                        ),
                        Axis::Z => UnitQuaternion::from_axis_angle(
                            &Vector3::z_axis(),
                            scalar_joint_vars[j],
                        ),
                    };

                    *conf_i = temp;
                    j += 1;
                }
                JointType::Prismatic(axis) => {
                    let mut temp = Isometry3::identity();
                    temp.translation = match axis {
                        Axis::X => Translation3::new(scalar_joint_vars[j], 0.0, 0.0),
                        Axis::Y => Translation3::new(0.0, scalar_joint_vars[j], 0.0),
                        Axis::Z => Translation3::new(0.0, 0.0, scalar_joint_vars[j]),
                    };

                    *conf_i = temp;
                    j += 1;
                }
                JointType::SixDOF => {
                    *conf_i = six_dof_vars[k];
                    k += 1;
                }
            }
        }
        Ok(conf)
    }

    /// Advances the dynamics state by one timestep.
    ///
    /// This is the panic-on-error wrapper around [`try_step_dynamics`].
    pub fn step_dynamics(
        &self,
        state: &DynamicsState<NUM_BODIES, NUM_DOFS>,
        input: DynamicsStepInput<'_, NUM_BODIES, NUM_DOFS>,
        options: IntegrationOptions,
    ) -> DynamicsState<NUM_BODIES, NUM_DOFS> {
        match self.try_step_dynamics(state, input, options) {
            Ok(state) => state,
            Err(err) => panic!("{}", err),
        }
    }

    /// Checked variant of [`step_dynamics`].
    pub fn try_step_dynamics(
        &self,
        state: &DynamicsState<NUM_BODIES, NUM_DOFS>,
        input: DynamicsStepInput<'_, NUM_BODIES, NUM_DOFS>,
        options: IntegrationOptions,
    ) -> Result<DynamicsState<NUM_BODIES, NUM_DOFS>, &'static str> {
        self.validate_dynamics_step(state, input, options)?;
        let mut workspace = ForwardDynamicsWorkspace::<NUM_BODIES>::new();

        let next_state = match options.method {
            IntegrationMethod::SemiImplicitEuler => {
                self.step_dynamics_euler(state, input, options.dt, &mut workspace)?
            }
            IntegrationMethod::Rk4 => {
                self.step_dynamics_rk4(state, input, options.dt, &mut workspace)?
            }
        };

        Ok(next_state)
    }

    fn validate_dynamics_step(
        &self,
        state: &DynamicsState<NUM_BODIES, NUM_DOFS>,
        input: DynamicsStepInput<'_, NUM_BODIES, NUM_DOFS>,
        options: IntegrationOptions,
    ) -> Result<(), &'static str> {
        if !options.dt.is_finite() || options.dt < 0.0 {
            return Err("dt must be finite and non-negative");
        }
        if state.conf.len() != NUM_BODIES {
            return Err("conf length mismatch");
        }
        if input.thruster_forces.len() != NUM_BODIES {
            return Err("thruster_forces length mismatch");
        }
        Ok(())
    }

    fn validate_configuration(&self, conf: &[Isometry3<f64>]) -> Result<(), &'static str> {
        if conf.len() != NUM_BODIES {
            return Err("conf length mismatch");
        }
        Ok(())
    }

    fn validate_body_id(&self, body_id: usize) -> Result<(), &'static str> {
        if body_id >= NUM_BODIES {
            return Err("body_id out of range");
        }
        Ok(())
    }

    fn validate_jacobians(&self, jacs: &[SMatrix<f64, 6, NUM_DOFS>]) -> Result<(), &'static str> {
        if jacs.len() != NUM_BODIES {
            return Err("jacobians length mismatch");
        }
        Ok(())
    }

    fn step_dynamics_euler(
        &self,
        state: &DynamicsState<NUM_BODIES, NUM_DOFS>,
        input: DynamicsStepInput<'_, NUM_BODIES, NUM_DOFS>,
        dt: f64,
        workspace: &mut ForwardDynamicsWorkspace<NUM_BODIES>,
    ) -> Result<DynamicsState<NUM_BODIES, NUM_DOFS>, &'static str> {
        let acceleration = self.dynamics_acceleration(state, input, workspace)?;
        let mu = state.mu + dt * acceleration;
        let conf = self.advance_configuration(&state.conf, &mu, dt);

        Ok(DynamicsState { conf, mu })
    }

    fn step_dynamics_rk4(
        &self,
        state: &DynamicsState<NUM_BODIES, NUM_DOFS>,
        input: DynamicsStepInput<'_, NUM_BODIES, NUM_DOFS>,
        dt: f64,
        workspace: &mut ForwardDynamicsWorkspace<NUM_BODIES>,
    ) -> Result<DynamicsState<NUM_BODIES, NUM_DOFS>, &'static str> {
        let k1_mu = self.dynamics_acceleration(state, input, workspace)?;
        let k1_conf_velocity = state.mu;

        let state2 = DynamicsState {
            conf: self.advance_configuration(&state.conf, &k1_conf_velocity, 0.5 * dt),
            mu: state.mu + 0.5 * dt * k1_mu,
        };
        let k2_mu = self.dynamics_acceleration(&state2, input, workspace)?;
        let k2_conf_velocity = state2.mu;

        let state3 = DynamicsState {
            conf: self.advance_configuration(&state.conf, &k2_conf_velocity, 0.5 * dt),
            mu: state.mu + 0.5 * dt * k2_mu,
        };
        let k3_mu = self.dynamics_acceleration(&state3, input, workspace)?;
        let k3_conf_velocity = state3.mu;

        let state4 = DynamicsState {
            conf: self.advance_configuration(&state.conf, &k3_conf_velocity, dt),
            mu: state.mu + dt * k3_mu,
        };
        let k4_mu = self.dynamics_acceleration(&state4, input, workspace)?;
        let k4_conf_velocity = state4.mu;

        let mu = state.mu + (dt / 6.0) * (k1_mu + 2.0 * k2_mu + 2.0 * k3_mu + k4_mu);
        let conf = self.advance_configuration_rk4(
            &state.conf,
            &k1_conf_velocity,
            &k2_conf_velocity,
            &k3_conf_velocity,
            &k4_conf_velocity,
            dt,
        );

        Ok(DynamicsState { conf, mu })
    }

    fn dynamics_acceleration(
        &self,
        state: &DynamicsState<NUM_BODIES, NUM_DOFS>,
        input: DynamicsStepInput<'_, NUM_BODIES, NUM_DOFS>,
        workspace: &mut ForwardDynamicsWorkspace<NUM_BODIES>,
    ) -> Result<SVector<f64, NUM_DOFS>, &'static str> {
        self.try_forward_dynamics_ab_with_workspace(
            &state.conf,
            &state.mu,
            input.rigid_body_forces,
            input.thruster_forces,
            input.eta,
            input.lin_vel_current,
            input.lin_accel_current,
            workspace,
        )
    }

    fn advance_configuration(
        &self,
        conf: &[Isometry3<f64>],
        mu: &SVector<f64, NUM_DOFS>,
        dt: f64,
    ) -> Vec<Isometry3<f64>> {
        let mut next_conf = Vec::with_capacity(NUM_BODIES);

        for (i, conf_i) in conf.iter().enumerate().take(NUM_BODIES) {
            next_conf.push(*conf_i * self.joint_delta(i, mu, dt));
        }

        next_conf
    }

    fn advance_configuration_rk4(
        &self,
        conf: &[Isometry3<f64>],
        k1: &SVector<f64, NUM_DOFS>,
        k2: &SVector<f64, NUM_DOFS>,
        k3: &SVector<f64, NUM_DOFS>,
        k4: &SVector<f64, NUM_DOFS>,
        dt: f64,
    ) -> Vec<Isometry3<f64>> {
        let scalar_velocity = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        let mut next_conf = Vec::with_capacity(NUM_BODIES);

        for (i, conf_i) in conf.iter().enumerate().take(NUM_BODIES) {
            match &self.joint_types[i] {
                JointType::Revolute(_) | JointType::Prismatic(_) => {
                    next_conf.push(*conf_i * self.joint_delta(i, &scalar_velocity, dt));
                }
                JointType::SixDOF => {
                    let idx = i + self.joint_size_offsets[i];
                    // Body-frame SE(3) twists from different RK stages live in their staged
                    // frames, so compose ordered stage exponentials instead of averaging twists.
                    let delta = exp_se3(&(Self::six_dof_twist(k1, idx) * (dt / 6.0)))
                        * exp_se3(&(Self::six_dof_twist(k2, idx) * (dt / 3.0)))
                        * exp_se3(&(Self::six_dof_twist(k3, idx) * (dt / 3.0)))
                        * exp_se3(&(Self::six_dof_twist(k4, idx) * (dt / 6.0)));
                    next_conf.push(*conf_i * delta);
                }
            }
        }

        next_conf
    }

    fn joint_delta(&self, body_id: usize, mu: &SVector<f64, NUM_DOFS>, dt: f64) -> Isometry3<f64> {
        let idx = body_id + self.joint_size_offsets[body_id];
        match &self.joint_types[body_id] {
            JointType::Revolute(axis) => {
                let angle = mu[idx] * dt;
                let rotation = match axis {
                    Axis::X => UnitQuaternion::from_axis_angle(&Vector3::x_axis(), angle),
                    Axis::Y => UnitQuaternion::from_axis_angle(&Vector3::y_axis(), angle),
                    Axis::Z => UnitQuaternion::from_axis_angle(&Vector3::z_axis(), angle),
                };
                Isometry3::from_parts(Translation3::identity(), rotation)
            }
            JointType::Prismatic(axis) => {
                let distance = mu[idx] * dt;
                let translation = match axis {
                    Axis::X => Translation3::new(distance, 0.0, 0.0),
                    Axis::Y => Translation3::new(0.0, distance, 0.0),
                    Axis::Z => Translation3::new(0.0, 0.0, distance),
                };
                Isometry3::from_parts(translation, UnitQuaternion::identity())
            }
            JointType::SixDOF => exp_se3(&(Self::six_dof_twist(mu, idx) * dt)),
        }
    }

    fn six_dof_twist(mu: &SVector<f64, NUM_DOFS>, idx: usize) -> Vector6<f64> {
        Vector6::from_column_slice(mu.rows(idx, 6).as_slice())
    }

    pub fn generalized_newton_euler(
        &self,
        conf: &[Isometry3<f64>],
        mu: &SVector<f64, NUM_DOFS>,
        mu_prime: &SVector<f64, NUM_DOFS>,
        sigma_prime: &SVector<f64, NUM_DOFS>,
        rigid_body_forces: impl Fn(&[Vector6<f64>], &[Vector6<f64>]) -> SMatrix<f64, 6, NUM_BODIES>,
        eta: &SVector<f64, NUM_DOFS>,
    ) -> SVector<f64, NUM_DOFS> {
        let mut w: Vec<Vector6<f64>> = vec![Vector6::zeros(); NUM_BODIES];
        let mut zeta = SVector::<f64, NUM_DOFS>::zeros();
        let mut h = vec![Isometry3::<f64>::identity(); NUM_BODIES];
        let mut alpha = vec![Vector6::<f64>::zeros(); NUM_BODIES];
        let mut nu = vec![Vector6::<f64>::zeros(); NUM_BODIES];
        let mut nu_prime = vec![Vector6::<f64>::zeros(); NUM_BODIES];
        // Cache Ad(h_i^{-1}) for reuse (avoids repeated inverse computations)
        let mut Ad_h_inv_cache = vec![Matrix6::zeros(); NUM_BODIES];

        let lambda = |x: usize| -> i32 { self.parent[x] as i32 - 1 };

        for i in 0..NUM_BODIES {
            let idx = i + self.joint_size_offsets[i];
            h[i] = self.offset_matrices[i] * conf[i];
            Ad_h_inv_cache[i] = Ad_inv(&h[i]);

            let Phi_i = self.Phi.columns(idx, self.joint_dims[i]);
            let mu_i = mu.rows(idx, self.joint_dims[i]);
            let mu_prime_i = mu_prime.rows(idx, self.joint_dims[i]);
            let sigma_prime_i = sigma_prime.rows(idx, self.joint_dims[i]);
            // Cache repeated products
            // Joint spatial velocity and acceleration in body i coordinates.
            let v_i = Phi_i * mu_i;
            let vdot_i = Phi_i * mu_prime_i;
            let ad_v_i = ad_se3(&v_i);
            let ad_vdot_i = ad_se3(&vdot_i);

            if lambda(i) < 0 {
                nu[i] = v_i; // v_i is Copy (SVector)
                nu_prime[i] = vdot_i; // vdot_i is Copy

                match self.joint_types[i] {
                    JointType::Revolute(_) | JointType::Prismatic(_) => {
                        alpha[i] = ad_vdot_i * v_i + Phi_i * sigma_prime_i;
                    }
                    JointType::SixDOF => {
                        alpha[i] = ad_vdot_i * v_i
                            + Phi_i
                                * (sigma_prime_i
                                    + ad_se3(&mu_i.fixed_rows::<6>(0).into()) * mu_prime_i);
                    }
                }
            } else {
                let Ad_h_inv = Ad_h_inv_cache[i];

                nu[i] = Ad_h_inv * nu[lambda(i) as usize] + v_i;
                nu_prime[i] = Ad_h_inv * nu_prime[lambda(i) as usize] + vdot_i;

                // Reuse cached ad_v_i and ad_vdot_i, avoid recomputing Phi_i * mu_i
                alpha[i] = Ad_h_inv * alpha[lambda(i) as usize]
                    + Phi_i * sigma_prime_i
                    + 0.5 * ad_v_i * vdot_i
                    - 0.5 * ad_v_i * nu_prime[i]
                    + 0.5 * ad_se3(&nu[i]) * vdot_i;

                alpha[i] += match self.joint_types[i] {
                    JointType::Revolute(_) | JointType::Prismatic(_) => Vector6::zeros(),
                    JointType::SixDOF => {
                        let mu_i = mu_i.fixed_rows::<6>(0).into();
                        Phi_i * ad_se3(&mu_i) * mu_prime_i
                    }
                }
            }
            let quat = UnitQuaternion::from_quaternion(*h[i].rotation.quaternion());
            w[i] = self.mass_matrices[i] * alpha[i]
                - 1.0 / 2.0 * ad_se3(&nu[i]).transpose() * self.mass_matrices[i] * nu_prime[i]
                - 1.0 / 2.0 * ad_se3(&nu_prime[i]).transpose() * self.mass_matrices[i] * nu[i]
                - self.compute_hydrostatic_force(&quat, &Vector3::zeros(), i);
        }

        let rigid_body_forces = rigid_body_forces(&nu, &nu_prime);

        // backward step
        for i in (0..NUM_BODIES).rev() {
            let idx = i + self.joint_size_offsets[i];
            let Phi_i = self.Phi.columns(idx, self.joint_dims[i]);
            let eta_i = eta.rows(idx, self.joint_dims[i]);

            w[i] += rigid_body_forces.column(i);

            let zeta_i = Phi_i.transpose() * w[i] - eta_i;
            zeta.rows_mut(idx, self.joint_dims[i]).copy_from(&zeta_i);

            if lambda(i) >= 0 {
                w[lambda(i) as usize] =
                    w[lambda(i) as usize] + Ad_h_inv_cache[i].transpose() * w[i];
            }
        }
        zeta
    }

    /// Computes the mass matrix of the multibody system using the composite rigid body algorithm (CRB).
    ///
    /// This is the panic-on-error wrapper around [`try_compute_mass_matrix`].
    pub fn compute_mass_matrix(&self, conf: &[Isometry3<f64>]) -> SMatrix<f64, NUM_DOFS, NUM_DOFS> {
        self.try_compute_mass_matrix(conf)
            .unwrap_or_else(|err| panic!("{}", err))
    }

    /// Checked variant of [`compute_mass_matrix`].
    pub fn try_compute_mass_matrix(
        &self,
        conf: &[Isometry3<f64>],
    ) -> Result<SMatrix<f64, NUM_DOFS, NUM_DOFS>, &'static str> {
        self.validate_configuration(conf)?;

        let mut M_c = self.mass_matrices.clone();
        let mut M_o = SMatrix::<f64, NUM_DOFS, NUM_DOFS>::zeros();
        let mut h = vec![Isometry3::<f64>::identity(); NUM_BODIES];
        let mut Ad_h_inv_cache = vec![Matrix6::zeros(); NUM_BODIES];

        for i in 0..NUM_BODIES {
            h[i] = self.offset_matrices[i] * conf[i];
            Ad_h_inv_cache[i] = Ad_inv(&h[i]);
        }

        for i in (0..NUM_BODIES).rev() {
            let lambda_i = self.parent[i] as i32 - 1;
            let Ad_h_i_inv = Ad_h_inv_cache[i];
            if lambda_i >= 0 {
                M_c[lambda_i as usize] =
                    M_c[lambda_i as usize] + Ad_h_i_inv.transpose() * M_c[i] * Ad_h_i_inv;
            }
            let idx = i + self.joint_size_offsets[i];
            // Distinguish scalar vs 6DOF for stack-friendly computation
            if self.joint_dims[i] == 1 {
                let phi_col = self.Phi.fixed_view::<6, 1>(0, idx);
                let X = M_c[i] * phi_col;
                let mass_scalar = phi_col.transpose() * X;
                M_o[(idx, idx)] = mass_scalar[(0, 0)];

                let mut j = i;
                let lambda = |x: usize| -> i32 { self.parent[x] as i32 - 1 };
                let mut X_prop = X;
                while lambda(j) >= 0 {
                    X_prop = Ad_h_inv_cache[j].transpose() * X_prop;
                    j = lambda(j) as usize;
                    let idx_j = j + self.joint_size_offsets[j];
                    if self.joint_dims[j] == 1 {
                        let phi_j = self.Phi.fixed_view::<6, 1>(0, idx_j);
                        let temp = X_prop.transpose() * phi_j;
                        M_o[(idx, idx_j)] = temp[(0, 0)];
                        M_o[(idx_j, idx)] = temp[(0, 0)];
                    } else {
                        // 6DOF parent
                        let phi_j = self.Phi.fixed_view::<6, 6>(0, idx_j);
                        let temp = X_prop.transpose() * phi_j;
                        M_o.view_mut((idx, idx_j), (1, 6)).copy_from(&temp);
                        M_o.view_mut((idx_j, idx), (6, 1))
                            .copy_from(&temp.transpose());
                    }
                }
            } else {
                // 6DOF
                let phi_block = self.Phi.fixed_view::<6, 6>(0, idx);
                let mut X = M_c[i] * phi_block; // 6x6
                let self_block = phi_block.transpose() * X;
                M_o.view_mut((idx, idx), (6, 6)).copy_from(&self_block);

                let mut j = i;
                let lambda = |x: usize| -> i32 { self.parent[x] as i32 - 1 };
                while lambda(j) >= 0 {
                    X = Ad_h_inv_cache[j].transpose() * X;
                    j = lambda(j) as usize;
                    let idx_j = j + self.joint_size_offsets[j];
                    if self.joint_dims[j] == 1 {
                        let phi_j = self.Phi.fixed_view::<6, 1>(0, idx_j);
                        let temp = X.transpose() * phi_j; // 6x1
                        M_o.view_mut((idx, idx_j), (6, 1)).copy_from(&temp);
                        M_o.view_mut((idx_j, idx), (1, 6))
                            .copy_from(&temp.transpose());
                    } else {
                        // 6DOF
                        let phi_j = self.Phi.fixed_view::<6, 6>(0, idx_j);
                        let temp = X.transpose() * phi_j; // 6x6
                        M_o.view_mut((idx, idx_j), (6, 6)).copy_from(&temp);
                        M_o.view_mut((idx_j, idx), (6, 6))
                            .copy_from(&temp.transpose());
                    }
                }
            }
        }
        Ok(M_o)
    }

    /// Computes the forward dynamics using the articulated body algorithm (AB).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_dynamics_ab(
        &self,
        conf: &[Isometry3<f64>],
        mu: &SVector<f64, NUM_DOFS>,
        // damping_func: impl Fn(&Vector6<f64>, &Vector6<f64>, usize) -> Vector6<f64>,
        rigid_body_forces_func: impl Fn(
            &[Isometry3<f64>],
            &[Vector6<f64>],
        ) -> SMatrix<f64, 6, NUM_BODIES>,
        thruster_forces: &[Vector6<f64>],
        eta: &SVector<f64, NUM_DOFS>,
        lin_vel_current: &Vector3<f64>,
        lin_accel_current: &Vector3<f64>,
    ) -> SVector<f64, NUM_DOFS> {
        self.try_forward_dynamics_ab(
            conf,
            mu,
            rigid_body_forces_func,
            thruster_forces,
            eta,
            lin_vel_current,
            lin_accel_current,
        )
        .unwrap_or_else(|err| panic!("{}", err))
    }

    /// Checked variant of [`forward_dynamics_ab`].
    #[allow(clippy::too_many_arguments)]
    pub fn try_forward_dynamics_ab(
        &self,
        conf: &[Isometry3<f64>],
        mu: &SVector<f64, NUM_DOFS>,
        rigid_body_forces_func: impl Fn(
            &[Isometry3<f64>],
            &[Vector6<f64>],
        ) -> SMatrix<f64, 6, NUM_BODIES>,
        thruster_forces: &[Vector6<f64>],
        eta: &SVector<f64, NUM_DOFS>,
        lin_vel_current: &Vector3<f64>,
        lin_accel_current: &Vector3<f64>,
    ) -> Result<SVector<f64, NUM_DOFS>, &'static str> {
        let mut workspace = ForwardDynamicsWorkspace::<NUM_BODIES>::new();
        self.try_forward_dynamics_ab_with_workspace(
            conf,
            mu,
            rigid_body_forces_func,
            thruster_forces,
            eta,
            lin_vel_current,
            lin_accel_current,
            &mut workspace,
        )
    }

    /// Computes forward dynamics using caller-owned scratch storage.
    ///
    /// This is the allocation-reuse variant of [`forward_dynamics_ab`]. Reusing a workspace across
    /// calls avoids repeatedly allocating the articulated-body algorithm's temporary buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_dynamics_ab_with_workspace(
        &self,
        conf: &[Isometry3<f64>],
        mu: &SVector<f64, NUM_DOFS>,
        rigid_body_forces_func: impl Fn(
            &[Isometry3<f64>],
            &[Vector6<f64>],
        ) -> SMatrix<f64, 6, NUM_BODIES>,
        thruster_forces: &[Vector6<f64>],
        eta: &SVector<f64, NUM_DOFS>,
        lin_vel_current: &Vector3<f64>,
        lin_accel_current: &Vector3<f64>,
        workspace: &mut ForwardDynamicsWorkspace<NUM_BODIES>,
    ) -> SVector<f64, NUM_DOFS> {
        self.try_forward_dynamics_ab_with_workspace(
            conf,
            mu,
            rigid_body_forces_func,
            thruster_forces,
            eta,
            lin_vel_current,
            lin_accel_current,
            workspace,
        )
        .unwrap_or_else(|err| panic!("{}", err))
    }

    /// Checked variant of [`forward_dynamics_ab_with_workspace`].
    #[allow(clippy::too_many_arguments)]
    pub fn try_forward_dynamics_ab_with_workspace(
        &self,
        conf: &[Isometry3<f64>],
        mu: &SVector<f64, NUM_DOFS>,
        rigid_body_forces_func: impl Fn(
            &[Isometry3<f64>],
            &[Vector6<f64>],
        ) -> SMatrix<f64, 6, NUM_BODIES>,
        thruster_forces: &[Vector6<f64>],
        eta: &SVector<f64, NUM_DOFS>,
        lin_vel_current: &Vector3<f64>,
        lin_accel_current: &Vector3<f64>,
        workspace: &mut ForwardDynamicsWorkspace<NUM_BODIES>,
    ) -> Result<SVector<f64, NUM_DOFS>, &'static str> {
        self.validate_configuration(conf)?;
        if thruster_forces.len() != NUM_BODIES {
            return Err("thruster_forces length mismatch");
        }

        let h = &mut workspace.h;
        let Ad_h_inv_cache = &mut workspace.Ad_h_inv_cache;
        let nu = &mut workspace.nu;
        let alpha = &mut workspace.alpha;
        let a_e = &mut workspace.a_e;
        let b = &mut workspace.b;
        let M_a = &mut workspace.M_a;
        let v_inv_scalar = &mut workspace.v_inv_scalar;
        let v_inv_matrix = &mut workspace.v_inv_matrix;
        let U_scalar = &mut workspace.U_scalar;
        let U_matrix = &mut workspace.U_matrix;
        let u_scalar = &mut workspace.u_scalar;
        let u_matrix = &mut workspace.u_matrix;
        let joint_is_sixdof = &mut workspace.joint_is_sixdof;

        let mut sigma = SVector::<f64, NUM_DOFS>::zeros();

        let mut nu_0 = Vector6::<f64>::zeros();
        nu_0.fixed_view_mut::<3, 1>(0, 0)
            .copy_from(&(-lin_vel_current));

        let a_e0 = self.gravity - lin_accel_current;
        a_e[0] = a_e0;

        M_a.copy_from_slice(&self.mass_matrices);

        let lambda = |x: usize| -> i32 { self.parent[x] as i32 - 1 };

        for i in 0..NUM_BODIES {
            let idx = i + self.joint_size_offsets[i];
            h[i] = self.offset_matrices[i] * conf[i];
            Ad_h_inv_cache[i] = Ad_inv(&h[i]);

            let joint_velocity = if self.joint_dims[i] == 1 {
                self.Phi.fixed_view::<6, 1>(0, idx) * mu[idx]
            } else {
                self.Phi.fixed_view::<6, 6>(0, idx) * mu.fixed_rows::<6>(idx)
            };

            if lambda(i) == -1 {
                nu[i] = Ad_h_inv_cache[i] * nu_0 + joint_velocity;
                a_e[i] = h[i].rotation.inverse() * a_e0;
            } else {
                nu[i] = Ad_h_inv_cache[i] * nu[lambda(i) as usize] + joint_velocity;
                a_e[i] = h[i].rotation.inverse() * a_e[lambda(i) as usize];
            }
            b[i] = -ad_se3(&nu[i]).transpose() * M_a[i] * nu[i]
                // - damping_func(&nu[i], &nu[i], i)
                - self.compute_hydrostatic_force(&h[i].rotation, lin_accel_current, i)
                - thruster_forces[i];
        }

        let rigid_body_forces = rigid_body_forces_func(&h[..], &nu[..]);

        for i in (0..NUM_BODIES).rev() {
            let idx = i + self.joint_size_offsets[i];
            b[i] += -rigid_body_forces.column(i);

            if self.joint_dims[i] == 1 {
                // Scalar joint path.
                let phi_col = self.Phi.fixed_view::<6, 1>(0, idx);
                let U_i_col = M_a[i] * phi_col; // 6x1
                let V_i_scalar = phi_col.transpose() * U_i_col; // 1x1
                let u_i_scalar = eta[idx] - (phi_col.transpose() * b[i])[0];
                let v_scalar = V_i_scalar[(0, 0)];
                if !v_scalar.is_finite() || v_scalar.abs() <= f64::EPSILON {
                    return Err("scalar joint matrix inversion failed");
                }
                let inv_scalar = 1.0 / v_scalar;
                let v_i = phi_col * mu[idx];

                if lambda(i) >= 0 {
                    // Rank-1 update: M_bar = M_a - U U^T / V
                    let outer = U_i_col * U_i_col.transpose();
                    let M_bar = M_a[i] - outer * inv_scalar;
                    let b_bar =
                        b[i] + M_bar * ad_se3(&nu[i]) * v_i + U_i_col * (inv_scalar * u_i_scalar);
                    let Ad_h_i_inv = Ad_h_inv_cache[i];
                    M_a[lambda(i) as usize] =
                        M_a[lambda(i) as usize] + Ad_h_i_inv.transpose() * M_bar * Ad_h_i_inv;
                    b[lambda(i) as usize] = b[lambda(i) as usize] + Ad_h_i_inv.transpose() * b_bar;
                }
                v_inv_scalar[i] = inv_scalar;
                U_scalar[i] = U_i_col;
                u_scalar[i] = u_i_scalar;
                joint_is_sixdof[i] = false;
            } else {
                // 6DOF joint path.
                let Phi_block = self.Phi.fixed_view::<6, 6>(0, idx);
                let v_i = Phi_block * mu.fixed_rows::<6>(idx);

                let U_i_block = M_a[i] * Phi_block; // 6x6
                let V_i_block = Phi_block.transpose() * U_i_block; // 6x6

                let eta_block = eta.fixed_rows::<6>(idx).into_owned();
                let u_i_block = eta_block - Phi_block.transpose() * b[i]; // 6x1
                let V_i_inv_block = V_i_block
                    .try_inverse()
                    .ok_or("6x6 joint matrix inversion failed")?;

                if lambda(i) >= 0 {
                    let M_bar = M_a[i] - U_i_block * V_i_inv_block * U_i_block.transpose();
                    let b_bar =
                        b[i] + M_bar * ad_se3(&nu[i]) * v_i + U_i_block * V_i_inv_block * u_i_block;
                    let Ad_h_i_inv = Ad_h_inv_cache[i];
                    M_a[lambda(i) as usize] =
                        M_a[lambda(i) as usize] + Ad_h_i_inv.transpose() * M_bar * Ad_h_i_inv;
                    b[lambda(i) as usize] = b[lambda(i) as usize] + Ad_h_i_inv.transpose() * b_bar;
                }
                v_inv_matrix[i] = V_i_inv_block;
                U_matrix[i] = U_i_block;
                u_matrix[i] = u_i_block;
                joint_is_sixdof[i] = true;
            }
        }

        let mut alpha_0 = Vector6::<f64>::zeros();
        alpha_0
            .fixed_view_mut::<3, 1>(0, 0)
            .copy_from(&(-lin_accel_current));

        for i in 0..NUM_BODIES {
            let idx = i + self.joint_size_offsets[i];
            let v_i = if self.joint_dims[i] == 6 {
                self.Phi.fixed_view::<6, 6>(0, idx) * mu.fixed_rows::<6>(idx)
            } else {
                self.Phi.fixed_view::<6, 1>(0, idx) * mu[idx]
            };

            let Ad_h_i_inv = Ad_h_inv_cache[i];

            let alpha_bar: SVector<f64, 6> = if lambda(i) == -1 {
                Ad_h_i_inv * alpha_0 + ad_se3(&nu[i]) * v_i
            } else {
                Ad_h_i_inv * alpha[lambda(i) as usize] + ad_se3(&nu[i]) * v_i
            };
            if joint_is_sixdof[i] {
                // 6DOF variant
                let temp = v_inv_matrix[i] * (u_matrix[i] - U_matrix[i].transpose() * alpha_bar);
                sigma.rows_mut(idx, 6).copy_from(&temp);
                alpha[i] = alpha_bar + self.Phi.fixed_view::<6, 6>(0, idx) * temp;
            } else {
                // Scalar variant lives in scalar stacks in same order.
                let correction = (U_scalar[i].transpose() * alpha_bar)[0];
                let temp_scalar = v_inv_scalar[i] * (u_scalar[i] - correction);
                // Write scalar result
                sigma[(idx, 0)] = temp_scalar;
                alpha[i] = alpha_bar + self.Phi.fixed_view::<6, 1>(0, idx) * temp_scalar;
            }
        }

        Ok(sigma)
    }

    pub fn compute_hydrostatic_force(
        &self,
        quat: &UnitQuaternion<f64>,
        current_accel: &Vector3<f64>,
        body_id: usize,
    ) -> Vector6<f64> {
        self.try_compute_hydrostatic_force(quat, current_accel, body_id)
            .unwrap_or_else(|err| panic!("{}", err))
    }

    /// Checked variant of [`compute_hydrostatic_force`].
    pub fn try_compute_hydrostatic_force(
        &self,
        quat: &UnitQuaternion<f64>,
        current_accel: &Vector3<f64>,
        body_id: usize,
    ) -> Result<Vector6<f64>, &'static str> {
        self.validate_body_id(body_id)?;

        let mut hydrostatic_force = Vector6::<f64>::zeros();

        let Rot = quat.to_rotation_matrix();
        let rho = self.rho.unwrap_or(0.0);
        let volume = match &self.volume {
            Some(volume) => volume[body_id],
            None => 0.0,
        };
        let r_cob = match &self.r_cob {
            Some(r_cob) => r_cob[body_id],
            None => Vector3::<f64>::zeros(),
        };

        let mass = self.mass.as_ref().unwrap()[body_id];
        let r_com = self.r_com.as_ref().unwrap()[body_id];

        let linear =
            (mass - rho * volume) * Rot.matrix().transpose() * (self.gravity - current_accel);
        let rotational = (mass * skew(&r_com) - rho * volume * skew(&r_cob))
            * Rot.matrix().transpose()
            * (self.gravity - current_accel);

        hydrostatic_force
            .fixed_view_mut::<3, 1>(0, 0)
            .copy_from(&linear);
        hydrostatic_force
            .fixed_view_mut::<3, 1>(3, 0)
            .copy_from(&rotational);

        Ok(hydrostatic_force)
    }

    pub fn compute_body_configurations(&self, config: &[Isometry3<f64>]) -> Vec<Isometry3<f64>> {
        self.try_compute_body_configurations(config)
            .unwrap_or_else(|err| panic!("{}", err))
    }

    /// Checked variant of [`compute_body_configurations`].
    pub fn try_compute_body_configurations(
        &self,
        config: &[Isometry3<f64>],
    ) -> Result<Vec<Isometry3<f64>>, &'static str> {
        self.validate_configuration(config)?;

        let mut g = vec![Isometry3::<f64>::identity(); NUM_BODIES];
        let lambda = |x: usize| -> i32 { self.parent[x] as i32 - 1 };

        for i in 0..NUM_BODIES {
            g[i] = self.offset_matrices[i] * config[i];
            if lambda(i) >= 0 {
                g[i] = g[lambda(i) as usize] * g[i];
            }
        }
        Ok(g)
    }

    pub fn compute_jacobians(&self, config: &[Isometry3<f64>]) -> Vec<SMatrix<f64, 6, NUM_DOFS>> {
        self.try_compute_jacobians(config)
            .unwrap_or_else(|err| panic!("{}", err))
    }

    /// Checked variant of [`compute_jacobians`].
    pub fn try_compute_jacobians(
        &self,
        config: &[Isometry3<f64>],
    ) -> Result<Vec<SMatrix<f64, 6, NUM_DOFS>>, &'static str> {
        self.validate_configuration(config)?;

        // O(N) recursive Jacobian construction.
        let mut jacs = vec![SMatrix::<f64, 6, NUM_DOFS>::zeros(); NUM_BODIES];
        let mut h = vec![Isometry3::<f64>::identity(); NUM_BODIES];
        let mut Ad_inv_cache = vec![Matrix6::zeros(); NUM_BODIES];

        for i in 0..NUM_BODIES {
            let idx_i = i + self.joint_size_offsets[i];
            h[i] = self.offset_matrices[i] * config[i];
            Ad_inv_cache[i] = Ad_inv(&h[i]);

            let parent_i = self.parent[i] as i32 - 1;
            if parent_i >= 0 {
                // Propagate parent Jacobian: J_i = Ad(h_i^{-1}) * J_parent
                jacs[i] = Ad_inv_cache[i] * jacs[parent_i as usize];
            } else {
                jacs[i].fill(0.0);
            }
            // Insert this joint's own motion subspace columns (overwriting transformed placeholder)
            let Phi_i = self.Phi.view((0, idx_i), (6, self.joint_dims[i]));
            jacs[i]
                .view_mut((0, idx_i), (6, self.joint_dims[i]))
                .copy_from(&Phi_i);
        }
        Ok(jacs)
    }

    pub fn compute_jacobian_derivatives(
        &self,
        jacs: &[SMatrix<f64, 6, NUM_DOFS>],
        config: &[Isometry3<f64>],
        mu: &SVector<f64, NUM_DOFS>,
    ) -> Vec<SMatrix<f64, 6, NUM_DOFS>> {
        self.try_compute_jacobian_derivatives(jacs, config, mu)
            .unwrap_or_else(|err| panic!("{}", err))
    }

    /// Checked variant of [`compute_jacobian_derivatives`].
    pub fn try_compute_jacobian_derivatives(
        &self,
        jacs: &[SMatrix<f64, 6, NUM_DOFS>],
        config: &[Isometry3<f64>],
        mu: &SVector<f64, NUM_DOFS>,
    ) -> Result<Vec<SMatrix<f64, 6, NUM_DOFS>>, &'static str> {
        self.validate_configuration(config)?;
        self.validate_jacobians(jacs)?;

        let mut jacobian_derivs = vec![SMatrix::<f64, 6, NUM_DOFS>::zeros(); NUM_BODIES];
        // Cache body transforms and adjoints once.
        let mut h = vec![Isometry3::<f64>::identity(); NUM_BODIES];
        let mut Ad_inv_cache = vec![Matrix6::zeros(); NUM_BODIES];
        let mut phi_mu_cache: Vec<Vector6<f64>> = vec![Vector6::zeros(); NUM_BODIES];
        for j in 0..NUM_BODIES {
            h[j] = self.offset_matrices[j] * config[j];
            Ad_inv_cache[j] = Ad_inv(&h[j]);
            let idx_j = j + self.joint_size_offsets[j];
            // Compute Phi_j * mu_j (6x1) manually (joint dim 1 or 6).
            match self.joint_dims[j] {
                1 => {
                    let col = self.Phi.column(idx_j);
                    phi_mu_cache[j] = col * mu[idx_j];
                }
                6 => {
                    // 6DOF block: copy mu segment then multiply by identity (Phi block is I6).
                    let mu_block = mu.rows(idx_j, 6);
                    for r in 0..6 {
                        phi_mu_cache[j][r] = mu_block[r];
                    }
                }
                _ => unreachable!("Unsupported joint dimension"),
            }
        }
        let lambda = |x: usize| -> i32 { self.parent[x] as i32 - 1 };
        // Optimized double loop: hoist ad_se3 computation per j and precompute product with jacs[j].
        for j in 1..NUM_BODIES {
            let parent = lambda(j);
            if parent < 0 {
                continue;
            }
            let Phi_q_mu_j = &phi_mu_cache[j];
            let ad_phi_mu_j = ad_se3(Phi_q_mu_j);
            let ad_phi_mu_j_jac_j = ad_phi_mu_j * jacs[j]; // 6 x NUM_DOFS
            let parent = parent as usize;
            // Iterate only true ancestors i of j
            for &i in &self.ancestors[j] {
                let idx_i = i + self.joint_size_offsets[i];
                let parent_block =
                    jacobian_derivs[parent].view((0, idx_i), (6, self.joint_dims[i]));
                let djac_ji = Ad_inv_cache[j] * parent_block
                    - ad_phi_mu_j_jac_j.view((0, idx_i), (6, self.joint_dims[i]));
                jacobian_derivs[j]
                    .view_mut((0, idx_i), (6, self.joint_dims[i]))
                    .copy_from(&djac_ji);
            }
        }
        Ok(jacobian_derivs)
    }

    pub fn compute_jacobian(
        &self,
        config: &[Isometry3<f64>],
        body_id: usize,
    ) -> SMatrix<f64, 6, NUM_DOFS> {
        self.try_compute_jacobian(config, body_id)
            .unwrap_or_else(|err| panic!("{}", err))
    }

    /// Checked variant of [`compute_jacobian`].
    pub fn try_compute_jacobian(
        &self,
        config: &[Isometry3<f64>],
        body_id: usize,
    ) -> Result<SMatrix<f64, 6, NUM_DOFS>, &'static str> {
        self.validate_configuration(config)?;
        self.validate_body_id(body_id)?;

        let mut jacobian = SMatrix::<f64, 6, NUM_DOFS>::zeros();
        let idx = body_id + self.joint_size_offsets[body_id];
        let Phi_i = self.Phi.view((0, idx), (6, self.joint_dims[body_id]));
        jacobian
            .view_mut((0, idx), (6, self.joint_dims[body_id]))
            .copy_from(&Phi_i);

        // Walk up the chain accumulating transforms; maintain g = h_parent * ... * h_body
        let mut j = body_id;
        let lambda = |x: usize| -> i32 { self.parent[x] as i32 - 1 };
        let mut k = Isometry3::<f64>::identity();
        let mut first = true;
        while lambda(j) >= 0 {
            let h_j = self.offset_matrices[j] * config[j];
            if first {
                k = h_j;
                first = false;
            } else {
                k = h_j * k;
            }
            j = lambda(j) as usize;
            let idx_j = j + self.joint_size_offsets[j];
            let Phi_j = self.Phi.view((0, idx_j), (6, self.joint_dims[j]));
            let Ad_k_inv = Ad_inv(&k); // k^{-1} adjoint
            jacobian
                .view_mut((0, idx_j), (6, self.joint_dims[j]))
                .copy_from(&(Ad_k_inv * Phi_j));
        }
        Ok(jacobian)
    }

    pub fn compute_jacobian_derivative(
        &self,
        config: &[Isometry3<f64>],
        mu: &SVector<f64, NUM_DOFS>,
        body_id: usize,
    ) -> SMatrix<f64, 6, NUM_DOFS> {
        self.try_compute_jacobian_derivative(config, mu, body_id)
            .unwrap_or_else(|err| panic!("{}", err))
    }

    /// Checked variant of [`compute_jacobian_derivative`].
    pub fn try_compute_jacobian_derivative(
        &self,
        config: &[Isometry3<f64>],
        mu: &SVector<f64, NUM_DOFS>,
        body_id: usize,
    ) -> Result<SMatrix<f64, 6, NUM_DOFS>, &'static str> {
        self.validate_configuration(config)?;
        self.validate_body_id(body_id)?;

        let mut jacobian_deriv = SMatrix::<f64, 6, NUM_DOFS>::zeros();
        let mut j = body_id;
        let lambda = |x: usize| -> i32 { self.parent[x] as i32 - 1 };
        let mut Ad_h_inv: SMatrix<f64, 6, 6>;
        let mut nu = SVector::<f64, 6>::zeros();
        let mut h = Isometry3::<f64>::identity();
        while lambda(j) >= 0 {
            let idx_j = j + self.joint_size_offsets[j];
            let Phi_j = self.Phi.view((0, idx_j), (6, self.joint_dims[j]));
            let mu_j = mu.rows(idx_j, self.joint_dims[j]);
            let h_j = self.offset_matrices[j] * config[j];
            if j == body_id {
                nu = SMatrix::<f64, 6, 6>::identity() * Phi_j * mu_j;
                h = h_j;
            } else {
                Ad_h_inv = Ad_inv(&h);
                nu += Ad_h_inv * Phi_j * mu_j;
                h = h_j * h;
            }
            j = lambda(j) as usize;
            let idx_j = j + self.joint_size_offsets[j];
            let Phi_j = self.Phi.view((0, idx_j), (6, self.joint_dims[j]));
            let jac_j = -ad_se3(&nu) * Ad_inv(&h) * Phi_j;
            jacobian_deriv
                .view_mut((0, idx_j), (6, self.joint_dims[j]))
                .copy_from(&jac_j);
        }
        Ok(jacobian_deriv)
    }

    /// Computes the regressor matrix for the multibody system. The function takes in body regressors in each link frame, as well as joint_regressors.
    pub fn compute_regressor_matrix<'a, const NUM_PARAMS: usize>(
        &self,
        body_regressors: [&'a BodyRegressorFn<'a, NUM_PARAMS>; NUM_BODIES],
        joint_regressors: [&'a JointRegressorFn<'a, NUM_PARAMS>; NUM_BODIES],
        conf: &[Isometry3<f64>],
        mu: &SVector<f64, NUM_DOFS>,
        mu_bar: &SVector<f64, NUM_DOFS>,
        sigma_bar: &SVector<f64, NUM_DOFS>,
    ) -> SMatrix<f64, NUM_DOFS, NUM_PARAMS> {
        self.try_compute_regressor_matrix(
            body_regressors,
            joint_regressors,
            conf,
            mu,
            mu_bar,
            sigma_bar,
        )
        .expect("regressor input validation failed")
    }

    /// Checked variant of [`compute_regressor_matrix`].
    pub fn try_compute_regressor_matrix<'a, const NUM_PARAMS: usize>(
        &self,
        body_regressors: [&'a BodyRegressorFn<'a, NUM_PARAMS>; NUM_BODIES],
        joint_regressors: [&'a JointRegressorFn<'a, NUM_PARAMS>; NUM_BODIES],
        conf: &[Isometry3<f64>],
        mu: &SVector<f64, NUM_DOFS>,
        mu_bar: &SVector<f64, NUM_DOFS>,
        sigma_bar: &SVector<f64, NUM_DOFS>,
    ) -> Result<SMatrix<f64, NUM_DOFS, NUM_PARAMS>, &'static str> {
        if conf.len() != NUM_BODIES {
            return Err("conf length mismatch");
        }
        let mut regressor = SMatrix::<f64, NUM_DOFS, NUM_PARAMS>::zeros();
        // Compute the regressor matrix
        let mut W: Vec<SMatrix<f64, 6, NUM_PARAMS>> =
            vec![SMatrix::<f64, 6, NUM_PARAMS>::zeros(); NUM_BODIES];
        let mut h = vec![Isometry3::<f64>::identity(); NUM_BODIES];
        let mut alpha_bar = vec![Vector6::<f64>::zeros(); NUM_BODIES];
        let mut nu = vec![Vector6::<f64>::zeros(); NUM_BODIES];
        let mut nu_bar = vec![Vector6::<f64>::zeros(); NUM_BODIES];
        // Cache Ad(h_i^{-1}) for reuse (avoids repeated inverse computations)
        let mut Ad_h_inv_cache = vec![Matrix6::zeros(); NUM_BODIES];

        let g = self.try_compute_body_configurations(conf)?;

        let lambda = |x: usize| -> i32 { self.parent[x] as i32 - 1 };

        for i in 0..NUM_BODIES {
            let idx = i + self.joint_size_offsets[i];
            h[i] = self.offset_matrices[i] * conf[i];
            Ad_h_inv_cache[i] = Ad_inv(&h[i]);

            let Phi_i = self.Phi.columns(idx, self.joint_dims[i]);
            let mu_i = mu.rows(idx, self.joint_dims[i]);
            let mu_bar_i = mu_bar.rows(idx, self.joint_dims[i]);
            let sigma_bar_i = sigma_bar.rows(idx, self.joint_dims[i]);
            // Cache repeated products
            // Joint spatial velocity and acceleration in body i coordinates.
            let v_i = Phi_i * mu_i;
            let vdot_i = Phi_i * mu_bar_i;
            let ad_v_i = ad_se3(&v_i);
            let ad_vdot_i = ad_se3(&vdot_i);

            if lambda(i) < 0 {
                nu[i] = v_i; // v_i is Copy (SVector)
                nu_bar[i] = vdot_i; // vdot_i is Copy

                alpha_bar[i] = ad_vdot_i * v_i + Phi_i * sigma_bar_i;

                alpha_bar[i] += match self.joint_types[i] {
                    JointType::Revolute(_) | JointType::Prismatic(_) => Vector6::zeros(),
                    JointType::SixDOF => Phi_i * ad_se3(&mu_i.fixed_rows::<6>(0).into()) * mu_bar_i,
                }
            } else {
                let Ad_h_inv = Ad_h_inv_cache[i];

                nu[i] = Ad_h_inv * nu[lambda(i) as usize] + v_i;
                nu_bar[i] = Ad_h_inv * nu_bar[lambda(i) as usize] + vdot_i;

                // Reuse cached ad_v_i and ad_vdot_i, avoid recomputing Phi_i * mu_i
                alpha_bar[i] = Ad_h_inv * alpha_bar[lambda(i) as usize]
                    + Phi_i * sigma_bar_i
                    + 0.5 * ad_v_i * vdot_i
                    - 0.5 * ad_v_i * nu_bar[i]
                    + 0.5 * ad_se3(&nu[i]) * vdot_i;

                alpha_bar[i] += match self.joint_types[i] {
                    JointType::Revolute(_) | JointType::Prismatic(_) => Vector6::zeros(),
                    JointType::SixDOF => {
                        let mu_i = mu_i.fixed_rows::<6>(0).into();
                        Phi_i * ad_se3(&mu_i) * mu_bar_i
                    }
                };
            }
            W[i] = body_regressors[i](&g[i], &nu[i], &nu_bar[i], &alpha_bar[i]);
        }

        // backward step
        for i in (0..NUM_BODIES).rev() {
            let idx = i + self.joint_size_offsets[i];
            let Phi_i = self.Phi.columns(idx, self.joint_dims[i]);
            let mut regressor_i = Phi_i.transpose() * W[i];

            match self.joint_types[i] {
                JointType::Revolute(_) | JointType::Prismatic(_) => {
                    let jr = joint_regressors[i](
                        &conf[i],
                        JointKinArg::Scalar(mu[idx]),
                        JointKinArg::Scalar(mu_bar[idx]),
                        JointKinArg::Scalar(sigma_bar[idx]),
                    );
                    if let JointRegressorOut::Row(row) = jr {
                        regressor_i += row;
                    } else {
                        return Err("scalar joint regressor returned Matrix");
                    }
                }
                JointType::SixDOF => {
                    let mu6 = Vector6::from_column_slice(mu.rows(idx, 6).as_slice());
                    let mu_bar6 = Vector6::from_column_slice(mu_bar.rows(idx, 6).as_slice());
                    let sigma_bar6 = Vector6::from_column_slice(sigma_bar.rows(idx, 6).as_slice());
                    let jr = joint_regressors[i](
                        &conf[i],
                        JointKinArg::SixDOF(mu6),
                        JointKinArg::SixDOF(mu_bar6),
                        JointKinArg::SixDOF(sigma_bar6),
                    );
                    if let JointRegressorOut::Matrix(matrix) = jr {
                        regressor_i += matrix; // 6xP
                    } else {
                        return Err("six_dof joint regressor returned Row");
                    }
                }
            }

            regressor
                .rows_mut(idx, self.joint_dims[i])
                .copy_from(&regressor_i);

            if lambda(i) >= 0 {
                W[lambda(i) as usize] =
                    W[lambda(i) as usize] + Ad_h_inv_cache[i].transpose() * W[i];
            }
        }

        Ok(regressor)
    }
}
