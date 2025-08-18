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
pub type BodyRegressorFn<const NUM_PARAMS: usize> = dyn Fn(
    &Isometry3<f64>, // pose_world = g_i
    &Vector6<f64>,   // nu
    &Vector6<f64>,   // nu_bar
    &Vector6<f64>,   // alpha_bar
) -> SMatrix<f64, 6, NUM_PARAMS>;

/// Callback type for a per-joint regressor (additional joint effects).
///
/// Arguments:
/// - joint_pose_local: parent→joint_i transform (local joint configuration), i.e. conf[i].
/// - nu: body velocity, expressed in the body frame.
/// - nu_bar: desired body velocity, expressed in the body frame.
/// - alpha_bar: body acceleration, expressed in the body frame.
///
/// Returns:
/// - 6×NUM_PARAMS regressor contribution expressed in the body i frame.
pub type JointRegressorFn<const NUM_PARAMS: usize> = dyn Fn(
    &Isometry3<f64>, // joint_pose_local = conf[i]
    &Vector6<f64>,   // nu
    &Vector6<f64>,   // nu_bar
    &Vector6<f64>,   // alpha_bar
) -> SMatrix<f64, 6, NUM_PARAMS>;

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

// impl<T: na::RealField  + na::ClosedAdd + na::ClosedMul + na::ClosedDiv + Copy, const NUM_BODIES: usize, const NUM_DOFS: usize> MultiBody<T, NUM_BODIES, NUM_DOFS> {
impl<const NUM_BODIES: usize, const NUM_DOFS: usize> MultiBody<NUM_BODIES, NUM_DOFS> {
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
        // NOTE: Many parameters; consider refactoring to a builder (MultiBodyBuilder) to reduce
        // clippy::too_many_arguments while preserving clarity. Kept for backward compatibility.
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

        let mass_matrices = match mass_matrices {
            Some(mass_matrices) => mass_matrices,
            None => {
                let mut mass_mats = vec![Matrix6::zeros(); NUM_BODIES];
                for i in 0..NUM_BODIES {
                    let m = mass.as_ref().expect(
                        "Scalar masses should be provided if the mass matrix is not given.",
                    )[i];
                    let r = r_com.as_ref().expect(
                        "The center of gravity must be given if the mass matrix is not given.",
                    )[i];
                    let added_mass_i = match added_mass {
                        Some(ref added_mass) => added_mass[i],
                        None => Matrix6::zeros(),
                    };
                    let inertia_mat = inertia_matrices.as_ref().expect("The 3x3 inertia matrices must be provided if the 6x6 mass matrix is not given.")[i];
                    mass_mats[i] = ({
                        let r = &r;
                        let inertia_mat = &inertia_mat;
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
                    }) + added_mass_i;
                }
                mass_mats
            }
        };

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
        Ok(MultiBody {
            offset_matrices,
            mass_matrices,
            joint_types,
            parent: parent_vec.clone(),
            ancestors: ancestors_build,
            Phi,
            joint_dims,
            joint_size_offsets: joint_offset_vec,
            gravity,
            r_com,
            r_cob,
            mass,
            volume,
            rho,
        })
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
        let six_dof_vars = six_dof_vars.into();
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
        conf
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

    /// Computes the mass matrix of the multibody system using the composite rigid body algorithm (CRB). Assumes that GNE/MNE/AB has been called.
    pub fn compute_mass_matrix(&self, conf: &[Isometry3<f64>]) -> SMatrix<f64, NUM_DOFS, NUM_DOFS> {
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
        M_o
    }

    /// Computes the forward dynamics using the articulated body algorithm (AB).
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
        // TODO: Consider consolidating lesser-used arguments (thruster forces, environment terms)
        // into a context struct to appease clippy::too_many_arguments without harming ergonomics.
        let mut h = vec![Isometry3::<f64>::identity(); NUM_BODIES];
        let mut nu = vec![Vector6::<f64>::zeros(); NUM_BODIES];
        let mut alpha = vec![Vector6::<f64>::zeros(); NUM_BODIES];
        let mut sigma = SVector::<f64, NUM_DOFS>::zeros();

        let mut nu_0 = Vector6::<f64>::zeros();
        nu_0.fixed_view_mut::<3, 1>(0, 0)
            .copy_from(&(-lin_vel_current));

        let mut a_e = vec![Vector3::<f64>::zeros(); NUM_BODIES];
        let mut b = vec![Vector6::<f64>::zeros(); NUM_BODIES];
        let a_e0 = self.gravity - lin_accel_current;
        a_e[0] = a_e0;

        let mut M_a = self.mass_matrices.clone();
        // Preallocate (optimization 1) and fill in reverse order indices.
        // Store per-joint intermediate data without dynamic heap matrices.
        // For scalar joints we use rank-1 representations (f64, Vector6); for 6DOF full Matrix6/Vector6.
        let mut v_inv_scalar: Vec<f64> = vec![0.0; NUM_BODIES];
        let mut v_inv_matrix: Vec<Matrix6<f64>> = vec![Matrix6::<f64>::zeros(); NUM_BODIES];
        let mut U_scalar: Vec<Vector6<f64>> = vec![Vector6::<f64>::zeros(); NUM_BODIES];
        let mut U_matrix: Vec<Matrix6<f64>> = vec![Matrix6::<f64>::zeros(); NUM_BODIES];
        let mut u_scalar: Vec<f64> = vec![0.0; NUM_BODIES];
        let mut u_matrix: Vec<Vector6<f64>> = vec![Vector6::<f64>::zeros(); NUM_BODIES];
        let mut joint_is_sixdof: Vec<bool> = vec![false; NUM_BODIES];

        let lambda = |x: usize| -> i32 { self.parent[x] as i32 - 1 };

        for i in 0..NUM_BODIES {
            let idx = i + self.joint_size_offsets[i];
            h[i] = self.offset_matrices[i] * conf[i];

            let Phi_i = self.Phi.view((0, idx), (6, self.joint_dims[i]));
            let mu_i = mu.rows(idx, self.joint_dims[i]);

            if lambda(i) == -1 {
                nu[i] = Ad(&h[i].inverse()) * nu_0 + Phi_i * mu_i;
                a_e[i] = h[i].rotation.inverse() * a_e0;
            } else {
                nu[i] = Ad(&h[i].inverse()) * nu[lambda(i) as usize] + Phi_i * mu_i;
                a_e[i] = h[i].rotation.inverse() * a_e[lambda(i) as usize];
            }
            let quat = UnitQuaternion::from_quaternion(*h[i].rotation.quaternion());
            b[i] = -ad_se3(&nu[i]).transpose() * M_a[i] * nu[i]
                // - damping_func(&nu[i], &nu[i], i)
                - self.compute_hydrostatic_force(&quat, lin_accel_current, i)
                - thruster_forces[i];
        }

        let rigid_body_forces = rigid_body_forces_func(&h, &nu);

        for i in (0..NUM_BODIES).rev() {
            let idx = i + self.joint_size_offsets[i];
            let Phi_i = self.Phi.view((0, idx), (6, self.joint_dims[i]));
            let mu_i = mu.rows(idx, self.joint_dims[i]);
            b[i] += -rigid_body_forces.column(i);

            if self.joint_dims[i] == 1 {
                // Scalar joint path.
                let phi_col = Phi_i.column(0); // 6x1 (dynamic view)
                let U_i_col = M_a[i] * phi_col; // 6x1
                let V_i_scalar = phi_col.transpose() * U_i_col; // 1x1
                let u_i_scalar = (eta.rows(idx, 1)[0]) - (phi_col.transpose() * b[i])[0];
                let inv_scalar = 1.0 / V_i_scalar[(0, 0)];

                // Build v_i as static 6x1 for downstream use
                let mut v_i = Vector6::<f64>::zeros();
                v_i.copy_from(&(Phi_i * mu_i));

                if lambda(i) >= 0 {
                    // Rank-1 update: M_bar = M_a - U U^T / V
                    let outer = U_i_col * U_i_col.transpose();
                    let M_bar = M_a[i] - outer * inv_scalar;
                    let b_bar =
                        b[i] + M_bar * ad_se3(&nu[i]) * v_i + U_i_col * (inv_scalar * u_i_scalar);
                    let Ad_h_i_inv = Ad(&h[i].inverse());
                    M_a[lambda(i) as usize] =
                        M_a[lambda(i) as usize] + Ad_h_i_inv.transpose() * M_bar * Ad_h_i_inv;
                    b[lambda(i) as usize] = b[lambda(i) as usize] + Ad_h_i_inv.transpose() * b_bar;
                }
                v_inv_scalar[i] = inv_scalar;
                U_scalar[i] = U_i_col;
                u_scalar[i] = u_i_scalar;
                joint_is_sixdof[i] = false;
            } else {
                // 6DOF joint path: convert dynamic views to fixed-size matrices/vectors.
                let mut Phi_block = Matrix6::<f64>::zeros();
                Phi_block.copy_from(&Phi_i);
                let mut v_i = Vector6::<f64>::zeros();
                v_i.copy_from(&(Phi_block * mu_i));

                let U_i_block = M_a[i] * Phi_block; // 6x6
                let V_i_block = Phi_block.transpose() * U_i_block; // 6x6

                let mut eta_block = Vector6::<f64>::zeros();
                eta_block.copy_from(&eta.rows(idx, 6));
                let u_i_block = eta_block - Phi_block.transpose() * b[i]; // 6x1
                let V_i_inv_block = V_i_block
                    .try_inverse()
                    .expect("6x6 joint matrix inversion failed");

                if lambda(i) >= 0 {
                    let M_bar = M_a[i] - U_i_block * V_i_inv_block * U_i_block.transpose();
                    let b_bar =
                        b[i] + M_bar * ad_se3(&nu[i]) * v_i + U_i_block * V_i_inv_block * u_i_block;
                    let Ad_h_i_inv = Ad(&h[i].inverse());
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
            let Phi_i = self.Phi.view((0, idx), (6, self.joint_dims[i]));
            let mu_i = mu.rows(idx, self.joint_dims[i]);
            // Recompute v_i as fixed-size where possible.
            let mut v_i = Vector6::<f64>::zeros();
            if self.joint_dims[i] == 6 {
                let mut Phi_block = Matrix6::<f64>::zeros();
                Phi_block.copy_from(&Phi_i);
                v_i.copy_from(&(Phi_block * mu_i));
            } else {
                // scalar joint: Phi_i * mu_i yields 6x1; copy into Vector6
                v_i.copy_from(&(Phi_i * mu_i));
            }

            let Ad_h_i_inv = Ad(&h[i].inverse());

            let alpha_bar: SVector<f64, 6> = if lambda(i) == -1 {
                Ad_h_i_inv * alpha_0 + ad_se3(&nu[i]) * v_i
            } else {
                Ad_h_i_inv * alpha[lambda(i) as usize] + ad_se3(&nu[i]) * v_i
            };
            if joint_is_sixdof[i] {
                // 6DOF variant
                let temp = v_inv_matrix[i] * (u_matrix[i] - U_matrix[i].transpose() * alpha_bar);
                sigma.rows_mut(idx, 6).copy_from(&temp);
                alpha[i] = alpha_bar + Phi_i * temp;
            } else {
                // Scalar variant lives in scalar stacks in same order.
                let correction = (U_scalar[i].transpose() * alpha_bar)[0];
                let temp_scalar = v_inv_scalar[i] * (u_scalar[i] - correction);
                // Write scalar result
                sigma[(idx, 0)] = temp_scalar;
                alpha[i] = alpha_bar + Phi_i * SVector::<f64, 1>::from_element(temp_scalar);
            }
        }

        sigma
    }

    pub fn compute_hydrostatic_force(
        &self,
        quat: &UnitQuaternion<f64>,
        current_accel: &Vector3<f64>,
        body_id: usize,
    ) -> Vector6<f64> {
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

        hydrostatic_force
    }

    pub fn compute_body_configurations(&self, config: &[Isometry3<f64>]) -> Vec<Isometry3<f64>> {
        let mut g = vec![Isometry3::<f64>::identity(); NUM_BODIES];
        let lambda = |x: usize| -> i32 { self.parent[x] as i32 - 1 };

        for i in 0..NUM_BODIES {
            g[i] = self.offset_matrices[i] * config[i];
            if lambda(i) >= 0 {
                g[i] = g[lambda(i) as usize] * g[i];
            }
        }
        g
    }

    pub fn compute_jacobians(&self, config: &[Isometry3<f64>]) -> Vec<SMatrix<f64, 6, NUM_DOFS>> {
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
        jacs
    }

    pub fn compute_jacobian_derivatives(
        &self,
        jacs: &[SMatrix<f64, 6, NUM_DOFS>],
        config: &[Isometry3<f64>],
        mu: &SVector<f64, NUM_DOFS>,
    ) -> Vec<SMatrix<f64, 6, NUM_DOFS>> {
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
            // body 0 has no parent
            let Phi_q_mu_j = &phi_mu_cache[j];
            let ad_phi_mu_j = ad_se3(Phi_q_mu_j);
            let ad_phi_mu_j_jac_j = ad_phi_mu_j * jacs[j]; // 6 x NUM_DOFS
            let parent = lambda(j) as usize;
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
        jacobian_derivs
    }

    pub fn compute_jacobian(
        &self,
        config: &[Isometry3<f64>],
        body_id: usize,
    ) -> SMatrix<f64, 6, NUM_DOFS> {
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
        jacobian
    }

    pub fn compute_jacobian_derivative(
        &self,
        config: &[Isometry3<f64>],
        mu: &SVector<f64, NUM_DOFS>,
        body_id: usize,
    ) -> SMatrix<f64, 6, NUM_DOFS> {
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
                Ad_h_inv = Ad(&h.inverse());
                nu += Ad_h_inv * Phi_j * mu_j;
                h = h_j * h;
            }
            j = lambda(j) as usize;
            let idx_j = j + self.joint_size_offsets[j];
            let Phi_j = self.Phi.view((0, idx_j), (6, self.joint_dims[j]));
            let jac_j = -ad_se3(&nu) * Ad(&h.inverse()) * Phi_j;
            jacobian_deriv
                .view_mut((0, idx_j), (6, self.joint_dims[j]))
                .copy_from(&jac_j);
        }
        jacobian_deriv
    }

    /// Computes the regressor matrix for the multibody system. The function takes in body regressors in each link frame, as well as joint_regressors.
    pub fn compute_regressor_matrix<const NUM_PARAMS: usize, F>(
        &self,
        body_regressors: [&BodyRegressorFn<NUM_PARAMS>; NUM_BODIES],
        joint_regressors: [&JointRegressorFn<NUM_PARAMS>; NUM_BODIES],
        conf: &[Isometry3<f64>],
        mu: &SVector<f64, NUM_DOFS>,
        mu_prime: &SVector<f64, NUM_DOFS>,
        sigma_prime: &SVector<f64, NUM_DOFS>,
    ) -> SMatrix<f64, NUM_DOFS, NUM_PARAMS> {
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

        let g = self.compute_body_configurations(&conf);

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
                nu_bar[i] = vdot_i; // vdot_i is Copy

                alpha_bar[i] = ad_vdot_i * v_i + Phi_i * sigma_prime_i;

                alpha_bar[i] += match self.joint_types[i] {
                    JointType::Revolute(_) | JointType::Prismatic(_) => Vector6::zeros(),
                    JointType::SixDOF => {
                        Phi_i * ad_se3(&mu_i.fixed_rows::<6>(0).into()) * mu_prime_i
                    }
                }
            } else {
                let Ad_h_inv = Ad_h_inv_cache[i];

                nu[i] = Ad_h_inv * nu[lambda(i) as usize] + v_i;
                nu_bar[i] = Ad_h_inv * nu_bar[lambda(i) as usize] + vdot_i;

                // Reuse cached ad_v_i and ad_vdot_i, avoid recomputing Phi_i * mu_i
                alpha_bar[i] = Ad_h_inv * alpha_bar[lambda(i) as usize]
                    + Phi_i * sigma_prime_i
                    + 0.5 * ad_v_i * vdot_i
                    - 0.5 * ad_v_i * nu_bar[i]
                    + 0.5 * ad_se3(&nu[i]) * vdot_i;

                alpha_bar[i] += match self.joint_types[i] {
                    JointType::Revolute(_) | JointType::Prismatic(_) => Vector6::zeros(),
                    JointType::SixDOF => {
                        let mu_i = mu_i.fixed_rows::<6>(0).into();
                        Phi_i * ad_se3(&mu_i) * mu_prime_i
                    }
                };
            }
            W[i] = body_regressors[i](&g[i], &nu[i], &nu_bar[i], &alpha_bar[i]);
        }

        // backward step
        for i in (0..NUM_BODIES).rev() {
            let idx = i + self.joint_size_offsets[i];
            let Phi_i = self.Phi.columns(idx, self.joint_dims[i]);
            let regressor_i = Phi_i.transpose() * W[i]
                + joint_regressors[i](&conf[i], &nu[i], &nu_bar[i], &alpha_bar[i]);
            regressor
                .rows_mut(idx, self.joint_dims[i])
                .copy_from(&regressor_i);

            if lambda(i) >= 0 {
                W[lambda(i) as usize] =
                    W[lambda(i) as usize] + Ad_h_inv_cache[i].transpose() * W[i];
            }
        }

        regressor
    }
}
