// #![no_std]
#![cfg_attr(feature = "no_std", no_std)]

#[cfg(feature = "no_std")]
extern crate core as std;

extern crate nalgebra as na;
use crate::math_functions::*;
use na::{
    DMatrix, DVector, Dyn, Isometry3, Matrix1, Matrix3, Matrix4, Matrix6, OMatrix, Quaternion,
    SMatrix, SVector, Translation3, UnitQuaternion, Vector1, Vector3, Vector6, U1, U6,
};

// use num::{One, Zero};

#[derive(Clone, Debug)]
pub enum JointType {
    Revolute,
    Prismatic,
    SixDOF,
}

/// Allows overloading of functions for both a single 6DOF configuration and for a vector of 6DOF configurations, which is required when there are more than one 6DOF joint in the multibody system.
pub trait IntoHomogenousConfigurationVec {
    fn into(&self) -> Vec<Isometry3<f64>>;
}

impl IntoHomogenousConfigurationVec for Isometry3<f64> {
    fn into(&self) -> Vec<Isometry3<f64>> {
        vec![*self]
    }
}

impl IntoHomogenousConfigurationVec for Vec<Isometry3<f64>> {
    fn into(&self) -> Vec<Isometry3<f64>> {
        self.clone()
    }
}

pub struct MultiBody<const NUM_BODIES: usize, const NUM_DOFS: usize> {
    offset_matrices: Vec<Isometry3<f64>>,
    mass_matrices: Vec<Matrix6<f64>>,
    joint_types: Vec<JointType>,
    parent: Vec<u16>,
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
        let mut joint_dims = SVector::<usize, NUM_BODIES>::zeros();
        let mut Phi = SMatrix::<f64, 6, NUM_DOFS>::zeros();
        let mut joint_size_offsets = 0;
        let mut joint_offset_vec = vec![0; NUM_BODIES];

        for i in 0..NUM_BODIES {
            joint_offset_vec[i] = joint_size_offsets;

            match joint_types[i] {
                JointType::Revolute => {
                    let Phi_i = Vector6::new(0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
                    Phi.fixed_view_mut::<6, 1>(0, i + joint_size_offsets)
                        .copy_from(&Phi_i);
                    joint_dims[i] = 1;
                }
                JointType::Prismatic => {
                    let Phi_i = Vector6::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
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

        Ok(MultiBody {
            offset_matrices,
            mass_matrices,
            joint_types,
            parent,
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
    pub fn minimal_to_homogenous_configuration<Configuration, const D: usize>(
        &self,
        six_dof_vars: &Configuration,
        scalar_joint_vars: &SVector<f64, D>,
    ) -> Vec<Isometry3<f64>>
    where
        Configuration: IntoHomogenousConfigurationVec,
    {
        let six_dof_vars = six_dof_vars.into();
        let mut j = 0;
        let mut k = 0;

        let mut conf: Vec<Isometry3<f64>> = vec![Isometry3::identity(); NUM_BODIES];

        for i in 1..NUM_BODIES {
            match self.joint_types[i] {
                JointType::Revolute => {
                    let mut temp = Isometry3::identity();
                    temp.rotation =
                        UnitQuaternion::from_axis_angle(&Vector3::z_axis(), scalar_joint_vars[j]);

                    conf[i] = temp;
                    j += 1;
                }
                JointType::Prismatic => {
                    let mut temp = Isometry3::identity();
                    temp.translation = Translation3::new(0.0, 0.0, scalar_joint_vars[j]);

                    conf[i] = temp;
                    j += 1;
                }
                JointType::SixDOF => {
                    conf[i] = six_dof_vars[k];
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

        let lambda = |x: usize| -> i32 { self.parent[x] as i32 - 1 };

        for i in 0..NUM_BODIES {
            let idx = i + self.joint_size_offsets[i];
            h[i] = self.offset_matrices[i] * conf[i];

            let Phi_i = self.Phi.columns(idx, self.joint_dims[i]);
            let mu_i = mu.rows(idx, self.joint_dims[i]);
            let mu_prime_i = mu_prime.rows(idx, self.joint_dims[i]);
            let sigma_prime_i = sigma_prime.rows(idx, self.joint_dims[i]);

            if lambda(i) < 0 {
                nu[i] = Phi_i * mu_i;
                nu_prime[i] = Phi_i * mu_prime_i;

                match self.joint_types[i] {
                    JointType::Revolute | JointType::Prismatic => {
                        alpha[i] = ad_se3(&nu_prime[i]) * Phi_i * mu_i + Phi_i * sigma_prime_i;
                    }
                    JointType::SixDOF => {
                        alpha[i] = ad_se3(&nu_prime[i]) * Phi_i * mu_i
                            + Phi_i
                                * (sigma_prime_i
                                    + ad_se3(&mu_i.fixed_rows::<6>(0).try_into().unwrap())
                                        * mu_prime_i);
                    }
                }
            } else {
                let Ad_h_inv = Ad(&h[i].inverse());

                nu[i] = Ad_h_inv * nu[lambda(i) as usize] + Phi_i * mu_i;

                nu_prime[i] = Ad_h_inv * nu_prime[lambda(i) as usize] + Phi_i * mu_prime_i;

                alpha[i] = Ad_h_inv * alpha[lambda(i) as usize]
                            + ad_se3(&nu_prime[i]) * Phi_i * mu_i
                            + Phi_i * sigma_prime_i;

                alpha[i] += match self.joint_types[i] {
                    JointType::Revolute | JointType::Prismatic => {
                        Vector6::zeros()
                    }
                    JointType::SixDOF => {
                        let mu_i = mu_i.fixed_rows::<6>(0).try_into().unwrap();
                        Phi_i * ad_se3(&mu_i) * mu_prime_i
                    }
                }
            }
            let quat = UnitQuaternion::from_quaternion(*h[i].rotation.quaternion());
            w[i] = self.mass_matrices[i] * alpha[i]
                - ad_se3(&nu_prime[i]).transpose() * self.mass_matrices[i] * nu[i]
                - self.compute_hydrostatic_force(&quat, &Vector3::zeros(), i);
        }

        let rigid_body_forces = rigid_body_forces(&nu, &nu_prime);

        // backward step
        for i in (0..NUM_BODIES).rev() {
            let idx = i + self.joint_size_offsets[i];
            let Phi_i = self.Phi.columns(idx, self.joint_dims[i]);
            let eta_i = eta.rows(idx, self.joint_dims[i]);

            w[i] += rigid_body_forces.column(i);

            let temp = Phi_i.transpose() * w[i] - eta_i;
            zeta.rows_mut(idx, self.joint_dims[i]).copy_from(&temp);

            if lambda(i) >= 0 {
                w[lambda(i) as usize] =
                    w[lambda(i) as usize] + Ad(&h[i].inverse()).transpose() * w[i];
            }
        }
        zeta
    }

    /// Computes the mass matrix of the multibody system using the composite rigid body algorithm (CRB). Assumes that GNE/MNE/AB has been called.
    pub fn compute_mass_matrix(&self, conf: &[Isometry3<f64>]) -> SMatrix<f64, NUM_DOFS, NUM_DOFS> {
        let mut M_c = self.mass_matrices.clone();
        let mut M_o = SMatrix::<f64, NUM_DOFS, NUM_DOFS>::zeros();
        let mut h = vec![Isometry3::<f64>::identity(); NUM_BODIES];

        for i in 0..NUM_BODIES {
            h[i] = self.offset_matrices[i] * conf[i];
        }

        for i in (0..NUM_BODIES).rev() {
            let lambda_i = self.parent[i] as i32 - 1;
            let Ad_h_i_inv = Ad(&h[i].inverse());

            if lambda_i >= 0 {
                M_c[lambda_i as usize] =
                    M_c[lambda_i as usize] + Ad_h_i_inv.transpose() * M_c[i] * Ad_h_i_inv;
            }
            let idx = i + self.joint_size_offsets[i];

            let Phi_i: OMatrix<f64, U6, Dyn> = match self.joint_types[i] {
                JointType::Revolute | JointType::Prismatic => {
                    OMatrix::from_columns(&[self.Phi.fixed_view::<6, 1>(0, idx)])
                }
                JointType::SixDOF => OMatrix::<f64, U6, Dyn>::from_iterator(
                    6,
                    self.Phi.fixed_view::<6, 6>(0, idx).iter().cloned(),
                ),
            };
            let mut X = M_c[i] * &Phi_i;
            M_o.view_mut((idx, idx), (self.joint_dims[i], self.joint_dims[i]))
                .copy_from(&(Phi_i.transpose() * &X));

            let mut j = i;
            let lambda = |x: usize| -> i32 { self.parent[x] as i32 - 1 };

            while lambda(j) >= 0 {
                X = Ad(&h[j].inverse()).transpose() * X;
                j = lambda(j) as usize;

                let Phi_j: OMatrix<f64, U6, Dyn> = match self.joint_types[j] {
                    JointType::Revolute | JointType::Prismatic => OMatrix::from_columns(&[self
                        .Phi
                        .fixed_view::<6, 1>(0, j + self.joint_size_offsets[j])]),
                    JointType::SixDOF => OMatrix::<f64, U6, Dyn>::from_iterator(
                        6,
                        self.Phi
                            .fixed_view::<6, 6>(0, j + self.joint_size_offsets[j])
                            .iter()
                            .cloned(),
                    ),
                };
                let temp = X.transpose() * &Phi_j;

                M_o.view_mut(
                    (idx, j + self.joint_size_offsets[j]),
                    (self.joint_dims[i], self.joint_dims[j]),
                )
                .copy_from(&temp);
                M_o.view_mut(
                    (j + self.joint_size_offsets[j], idx),
                    (self.joint_dims[j], self.joint_dims[i]),
                )
                .copy_from(&temp.transpose());
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
        rigid_body_forces_func: impl Fn(&[Isometry3<f64>], &[Vector6<f64>]) -> SMatrix<f64, 6, NUM_BODIES>,
        thruster_forces: &[Vector6<f64>],
        eta: &SVector<f64, NUM_DOFS>,
        lin_vel_current: &Vector3<f64>,
        lin_accel_current: &Vector3<f64>,
    ) -> SVector<f64, NUM_DOFS> {
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
        let mut V_inv: Vec<DMatrix<f64>> = Vec::new();
        let mut U_vec: Vec<OMatrix<f64, U6, Dyn>> = Vec::new();
        let mut u: Vec<DVector<f64>> = Vec::new();

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
            let U_i = M_a[i] * Phi_i;

            let V_i = Phi_i.transpose() * &U_i;
            b[i] += -rigid_body_forces.column(i);

            let u_i = eta.rows(idx, self.joint_dims[i]) - Phi_i.transpose() * b[i];
            let V_i_inv = V_i.try_inverse().unwrap();

            if lambda(i) >= 0 {
                let mu_i = mu.rows(idx, self.joint_dims[i]);

                let M_bar = M_a[i] - &U_i * &V_i_inv * &U_i.transpose();
                let b_bar = b[i] + M_bar * ad_se3(&nu[i]) * Phi_i * mu_i + &U_i * &V_i_inv * &u_i;

                let Ad_h_i_inv = Ad(&h[i].inverse());
                M_a[lambda(i) as usize] =
                    M_a[lambda(i) as usize] + Ad_h_i_inv.transpose() * M_bar * Ad_h_i_inv;
                b[lambda(i) as usize] = b[lambda(i) as usize] + Ad_h_i_inv.transpose() * b_bar;
            }
            V_inv.insert(0, V_i_inv);
            U_vec.insert(0, U_i);
            u.insert(0, u_i);
        }

        let mut alpha_0 = Vector6::<f64>::zeros();
        alpha_0
            .fixed_view_mut::<3, 1>(0, 0)
            .copy_from(&(-lin_accel_current));

        for i in 0..NUM_BODIES {
            let idx = i + self.joint_size_offsets[i];
            let Phi_i = self.Phi.view((0, idx), (6, self.joint_dims[i]));
            let mu_i = mu.rows(idx, self.joint_dims[i]);

            let Ad_h_i_inv = Ad(&h[i].inverse());

            let alpha_bar: SVector<f64, 6> = if lambda(i) == -1 {
                Ad_h_i_inv * alpha_0 + ad_se3(&nu[i]) * Phi_i * mu_i
            } else {
                Ad_h_i_inv * alpha[lambda(i) as usize] + ad_se3(&nu[i]) * Phi_i * mu_i
            };
            let temp = &V_inv[i] * (&u[i] - &U_vec[i].transpose() * alpha_bar);
            sigma.rows_mut(idx, self.joint_dims[i]).copy_from(&temp);

            alpha[i] = alpha_bar + Phi_i * temp;
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

    pub fn compute_jacobians(&self, config: &[Isometry3<f64>]) -> Vec<SMatrix<f64, 6, NUM_DOFS>> {
        let mut jacs = vec![SMatrix::<f64, 6, NUM_DOFS>::zeros(); NUM_BODIES];
        let mut h = vec![Isometry3::<f64>::identity(); NUM_BODIES];

        for i in 0..NUM_BODIES {
            let idx = i + self.joint_size_offsets[i];

            let Phi_i = self.Phi.view((0, idx), (6, self.joint_dims[i]));
            jacs[i]
                .view_mut((0, idx), (6, self.joint_dims[i]))
                .copy_from(&Phi_i);

            let lambda = |x: usize| -> i32 { self.parent[x] as i32 - 1 };
            for j in i + 1..NUM_BODIES {
                if lambda(j) >= 0 {
                    h[j] = self.offset_matrices[j] * config[j];
                    let temp = Ad(&h[j].inverse())
                        * jacs[lambda(j) as usize].view((0, idx), (6, self.joint_dims[i]));

                    jacs[j]
                        .view_mut((0, idx), (6, self.joint_dims[i]))
                        .copy_from(&temp);
                }
            }
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
        let mut h = vec![Isometry3::<f64>::identity(); NUM_BODIES];

        for i in 0..NUM_BODIES {
            let idx = i + self.joint_size_offsets[i];
            for j in i + 1..NUM_BODIES {
                let lambda = |x: usize| -> i32 { self.parent[x] as i32 - 1 };
                if lambda(j) >= 0 {
                    let idx_j = j + self.joint_size_offsets[j];

                    let Phi_j = self.Phi.view((0, idx_j), (6, self.joint_dims[j]));
                    let mu_j = mu.rows(idx_j, self.joint_dims[j]);
                    h[j] = self.offset_matrices[j] * config[j];

                    let djac_ji = Ad(&h[j].inverse())
                        * jacobian_derivs[lambda(j) as usize]
                            .view((0, idx), (6, self.joint_dims[i]))
                        - ad_se3_dyn(&(Phi_j * mu_j))
                            * jacs[j].view((0, idx), (6, self.joint_dims[i]));

                    jacobian_derivs[j]
                        .view_mut((0, idx), (6, self.joint_dims[i]))
                        .copy_from(&djac_ji);
                }
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

        let mut j = body_id;
        let lambda = |x: usize| -> i32 { self.parent[x] as i32 - 1 };

        let mut k = Isometry3::<f64>::identity();
        while lambda(j) >= 0 {
            let h_j = self.offset_matrices[j] * config[j];

            if j == body_id {
                k = h_j;
            } else {
                k = h_j * k;
            }
            j = lambda(j) as usize;
            let idx_j = j + self.joint_size_offsets[j];

            let Phi_j = self.Phi.view((0, idx_j), (6, self.joint_dims[j]));
            jacobian
                .view_mut((0, idx_j), (6, self.joint_dims[j]))
                .copy_from(&(Ad(&k.inverse()) * Phi_j));
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
}

#[cfg(test)]
mod tests {
    use core::f64::consts::PI;

    use approx::assert_relative_eq;
    use na::{Translation3, Vector2};

    use super::*;

    fn comp_mass_matrix(m: f64, r: &Vector3<f64>, inertia_mat: &Matrix3<f64>) -> Matrix6<f64> {
        let mut mass_matrix = Matrix6::zeros();
        mass_matrix
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(m * Matrix3::identity()));
        mass_matrix
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(-m * skew(&r)));
        mass_matrix
            .fixed_view_mut::<3, 3>(3, 0)
            .copy_from(&(m * skew(&r)));
        mass_matrix
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&inertia_mat);
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

        // let p_01 = Translation3::new(l_1, 0.0, 0.0);
        let p_01 = Translation3::new(0.0, 0.0, 0.0);
        let q_01 = UnitQuaternion::identity();

        let p_12 = Translation3::new(l_1, 0.0, 0.0);
        // let q_12 = UnitQuaternion::from_euler_angles(0.0, 0.0, PI);
        let q_12 = UnitQuaternion::identity();

        let offset_matrix1 = Isometry3::from_parts(p_01, q_01);
        let offset_matrix2 = Isometry3::from_parts(p_12, q_12);
        let offset_matrices = vec![offset_matrix1, offset_matrix2];

        // let I_1 = Matrix3::identity();
        let I_1 = Matrix3::identity() - m_1 * skew(&r_cg1) * skew(&r_cg1);
        let I_2 = Matrix3::identity() - m_2 * skew(&r_cg2) * skew(&r_cg2);
        // let I_2 = Matrix3::identity();

        let M_1 = comp_mass_matrix(m_1, &r_cg1, &I_1);
        let M_2 = comp_mass_matrix(m_2, &r_cg2, &I_2);
        // let M_2 = Matrix6::zeros();

        let mass_matrices = vec![M_1, M_2];
        let inertia_mats = vec![I_1, I_2];

        // let joint_types = vec![JointType::SixDOF, JointType::Revolute];
        let joint_types = vec![JointType::Revolute, JointType::Revolute];
        let parent = vec![0, 1];
        let mut masses = Vec::new();
        masses.push(m_1);
        masses.push(m_2);

        let mut r_cg = vec![Vector3::<f64>::zeros(); 2];
        r_cg[0] = r_cg1;
        r_cg[1] = r_cg2;

        let mut multibody: MultiBody<2, 2> = MultiBody::new(
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

        let mut conf: Vec<Isometry3<f64>> = Vec::new();
        let temp = Isometry3::identity();
        conf.push(temp);
        conf.push(Isometry3::from_parts(
            Translation3::identity(),
            UnitQuaternion::from_euler_angles(0.0, 0.0, PI / 2.0),
        ));

        let mu = Vector2::new(0.0, 1.0);
        let sigma_prime = Vector2::new(0.0, 0.0);
        let eta = SVector::<f64, 2>::zeros();
        // let rigid_body_forces = vec![Vector6::<f64>::zeros(); 2];
        let rigid_body_forces_func =
            &|x: &[Vector6<f64>], y: &[Vector6<f64>]| -> SMatrix<f64, 6, 2> { SMatrix::<f64, 6, 2>::zeros()};

        let zeta = multibody.generalized_newton_euler(
            &conf,
            &mu,
            &mu,
            &sigma_prime,
            rigid_body_forces_func,
            &eta,
        );


        println!("zeta: {}", zeta);
        // println!("zeta2: {}", zeta2);
        assert_relative_eq!(zeta, Vector2::new(-0.5, 0.0), epsilon = 0.00001);
        // assert_relative_eq!(zeta2, Vector2::new(-0.5, 0.0), epsilon = 0.00001);
    }

    #[test]
    fn snake_like_model_test() {
        let c1 = Isometry3::<f64>::identity();
        let q12 = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), -PI / 2.0);
        let q21 = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), PI / 2.0);

        let l1 = 0.62;
        let l2 = 0.10;

        let c2 = Isometry3::from_parts(Translation3::new(l1, 0.0, 0.0), UnitQuaternion::identity());
        let c3 = Isometry3::from_parts(Translation3::new(l2, 0.0, 0.0), q12);
        let c4 = Isometry3::from_parts(Translation3::new(l1, 0.0, 0.0), q21);

        // let offset_matrices = vec![c0, c1, c2, c1, c2, c1, c2, c1, c2];
        let offset_matrices = vec![c1, c2, c3, c4, c3, c4, c3, c4, c3];
        // let offset_matrices = vec![c1, c2, c4, c3, c4, c3, c4, c3, c4];

        let mut joint_types = vec![JointType::Revolute; 9];
        joint_types[0] = JointType::SixDOF;
        let parent: Vec<u16> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];

        let r_cg1 = Vector3::new(l1 / 2.0, 0.0, 0.0);
        let r_cg2 = Vector3::new(l2 / 2.0, 0.0, 0.0);
        let r_cg = vec![r_cg1, r_cg2, r_cg1, r_cg2, r_cg1, r_cg2, r_cg1, r_cg2, r_cg1];

        let m1 = PI * 0.09 * 0.09 * l1 * 1000.0;
        let m2 = PI * 0.09 * 0.09 * l2 * 1000.0;
        let mut mass = vec![0.0; 9];
        mass[0] = m1;
        mass[1] = m2;
        mass[2] = m1;
        mass[3] = m2;
        mass[4] = m1;
        mass[5] = m2;
        mass[6] = m1;
        mass[7] = m2;
        mass[8] = m1;

        let inertia_mat1 = Matrix3::new(
            1.0 / 2.0 * m1 * 0.09 * 0.09,
            0.0,
            0.0,
            0.0,
            1.0 / 12.0 * m1 * (3.0 * 0.09 * 0.09 + l1 * l1),
            0.0,
            0.0,
            0.0,
            1.0 / 12.0 * m1 * (3.0 * 0.09 * 0.09 + l1 * l1),
        ) - m1 * skew(&r_cg1) * skew(&r_cg1);

        let inertia_mat2 = Matrix3::new(
            1.0 / 2.0 * m2 * 0.09 * 0.09,
            0.0,
            0.0,
            0.0,
            1.0 / 12.0 * m2 * (3.0 * 0.09 * 0.09 + l2 * l2),
            0.0,
            0.0,
            0.0,
            1.0 / 12.0 * m2 * (3.0 * 0.09 * 0.09 + l2 * l2),
        ) - m2 * skew(&r_cg2) * skew(&r_cg2);

        let M_1 = comp_mass_matrix(m1, &r_cg1, &inertia_mat1);
        let M_2 = comp_mass_matrix(m2, &r_cg2, &inertia_mat2);

        println!("M_1: {}", M_1);

        let mass_matrices = vec![M_1, M_2, M_1, M_2, M_1, M_2, M_1, M_2, M_1];

        // let mut multibody =
        // MultiBody::<9, 14>::new(offset_matrices, mass_matrices, joint_types, parent);
        let mut multibody: MultiBody<9, 14> = MultiBody::new(
            offset_matrices,
            Some(mass_matrices),
            None,
            None,
            joint_types,
            parent,
            Vector3::new(0.0, 0.0, 9.81),
            Some(r_cg),
            None,
            Some(mass),
            None,
            None,
        )
        .unwrap();

        // let mut conf: Vec<Isometry3<f64>> = Vec::new();

        let configuration_base = Isometry3::identity();
        // let configuration_base = Isometry3::from_parts(Translation3::new(0.0, 0.0, 0.0), UnitQuaternion::from_axis_angle(&Vector3::x_axis(), PI / 2.0)));

        // let joint_angles = SVector::<f64, 8>::zeros();
        let joint_angles = SVector::<f64, 8>::from_vec(vec![1.0; 8]);

        let conf =
            multibody.minimal_to_homogenous_configuration(&vec![configuration_base], &joint_angles);

        // let mu = SVector::<f64, 14>::zeros();
        let mu = SVector::<f64, 14>::repeat(1.0);
        // let mu_prime = SVector::<f64, 14>::repeat(1.0);
        let mu_prime = SVector::<f64, 14>::from_vec(vec![
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ]);
        // let mu_prime = SVector::<f64, 14>::repeat(1.0);
        // let mu = SVector::<f64, 14>::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        // println!("mu: {}", mu);
        let sigma_prime = SVector::<f64, 14>::zeros();
        // let sigma_prime = SVector::<f64, 14>::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let eta = SVector::<f64, 14>::zeros();
        // let rigid_body_forces = vec![Vector6::<f64>::zeros(); 9];
        let rigid_body_forces_func =
            &|x: &[Vector6<f64>], y: &[Vector6<f64>]| -> SMatrix<f64, 6, 9> { SMatrix::<f64, 6, 9>::zeros()};


        let zeta = multibody.generalized_newton_euler(
            &conf,
            &mu,
            &mu_prime,
            &sigma_prime,
            // &rigid_body_forces,
            rigid_body_forces_func,
            &eta,
        );

        let M_o = multibody.compute_mass_matrix(&conf);

        println!("zeta: {}", zeta);
        // println!("zeta2: {}", zeta2);
        println!("mass matrix: {}", M_o);
        println!("mass matrix: {}", M_o.column(13));
    }

    #[test]
    fn jacobian_test() {
        let c1 = Isometry3::<f64>::identity();
        let q12 = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), -PI / 2.0);
        let q21 = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), PI / 2.0);

        let l1 = 0.62;
        let l2 = 0.10;

        let c2 = Isometry3::from_parts(Translation3::new(l1, 0.0, 0.0), UnitQuaternion::identity());
        let c3 = Isometry3::from_parts(Translation3::new(l2, 0.0, 0.0), q12);
        let c4 = Isometry3::from_parts(Translation3::new(l1, 0.0, 0.0), q21);

        // let offset_matrices = vec![c0, c1, c2, c1, c2, c1, c2, c1, c2];
        let offset_matrices = vec![c1, c2, c3, c4, c3, c4, c3, c4, c3];
        // let offset_matrices = vec![c1, c2, c4, c3, c4, c3, c4, c3, c4];

        let mut joint_types = vec![JointType::Revolute; 9];
        joint_types[0] = JointType::SixDOF;
        let parent: Vec<u16> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];

        let r_cg1 = Vector3::new(l1 / 2.0, 0.0, 0.0);
        let r_cg2 = Vector3::new(l2 / 2.0, 0.0, 0.0);

        let m1 = PI * 0.09 * 0.09 * l1 * 1000.0;
        let m2 = PI * 0.09 * 0.09 * l2 * 1000.0;

        let inertia_mat1 = Matrix3::new(
            1.0 / 2.0 * m1 * 0.09 * 0.09,
            0.0,
            0.0,
            0.0,
            1.0 / 12.0 * m1 * (3.0 * 0.09 * 0.09 + l1 * l1),
            0.0,
            0.0,
            0.0,
            1.0 / 12.0 * m1 * (3.0 * 0.09 * 0.09 + l1 * l1),
        ) - m1 * skew(&r_cg1) * skew(&r_cg1);

        let inertia_mat2 = Matrix3::new(
            1.0 / 2.0 * m2 * 0.09 * 0.09,
            0.0,
            0.0,
            0.0,
            1.0 / 12.0 * m2 * (3.0 * 0.09 * 0.09 + l2 * l2),
            0.0,
            0.0,
            0.0,
            1.0 / 12.0 * m2 * (3.0 * 0.09 * 0.09 + l2 * l2),
        ) - m2 * skew(&r_cg2) * skew(&r_cg2);

        let M_1 = comp_mass_matrix(m1, &r_cg1, &inertia_mat1);
        let M_2 = comp_mass_matrix(m2, &r_cg2, &inertia_mat2);

        let mass_matrices = vec![M_1, M_2, M_1, M_2, M_1, M_2, M_1, M_2, M_1];

        // let mut multibody =
        //     MultiBody::<9, 14>::new(offset_matrices, Some(mass_matrices), joint_types, parent);
        let mut multibody: MultiBody<9, 14> = MultiBody::new(
            offset_matrices,
            Some(mass_matrices),
            None,
            None,
            joint_types,
            parent,
            Vector3::new(0.0, 0.0, 9.81),
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // let mut conf: Vec<Isometry3<f64>> = Vec::new();

        let configuration_base = Isometry3::identity();
        // let configuration_base = Isometry3::from_parts(Translation3::new(0.0, 0.0, 0.0), UnitQuaternion::from_axis_angle(&Vector3::x_axis(), PI / 2.0)));

        // let joint_angles = SVector::<f64, 8>::zeros();
        // let joint_angles = SVector::<f64, 8>::from_vec(vec![1.0; 8]);
        let joint_angles = SVector::<f64, 8>::from_vec(vec![
            PI / 4.0,
            PI / 3.0,
            PI / 5.0,
            PI / 7.0,
            PI / 15.0,
            PI / 10.0,
            PI / 4.0,
            PI / 2.5,
        ]);

        // let conf = multibody.minimal_to_homogenous_configuration(&vec![configuration_base], &joint_angles);
        let conf =
            multibody.minimal_to_homogenous_configuration(&configuration_base, &joint_angles);

        let jacs = multibody.compute_jacobians(&conf);
        let mu = SVector::<f64, 14>::repeat(1.0);
        let djacs = multibody.compute_jacobian_derivatives(&jacs, &conf, &mu);
        let djac = multibody.compute_jacobian_derivative(&conf, &mu, 7);

        let jac = multibody.compute_jacobian(&conf, 7);
        // println!("size jacs: {}", jacs.size());

        println!("end-effector jac: {}", jacs[7].column(0));
        println!("end-effector jac_i: {}", jac.column(0));
        println!("end-effector djac: {}", djacs[7]);
        println!("end-effector djac_i: {}", djac);

        // assert_relative_eq!(djacs[7].column(6), djac.column(6), epsilon = 0.00000001);
        assert_relative_eq!(jacs[7].column(6), jac.column(6), epsilon = 0.00000001);

        for i in 0..6 {
            // let i = 0;
            assert_relative_eq!(jacs[7].column(i), jac.column(i), epsilon = 0.00000001);
            assert_relative_eq!(djacs[7].column(i), djac.column(i), epsilon = 0.00000001);
        }

        let djac_col: SVector<f64, 6> = djac.column(6).try_into().unwrap();
        let jac_col: SVector<f64, 6> = jac.column(6).try_into().unwrap();
        assert_relative_eq!(
            djac_col,
            Vector6::new(
                -2.086813791,
                -3.2459777,
                -1.17134587,
                3.35849774,
                -0.0166829,
                -1.869255169
            ),
            epsilon = 0.001
        );
        assert_relative_eq!(
            jac_col,
            Vector6::new(
                0.559325,
                -0.029257,
                -0.62383325,
                -0.04200122,
                0.99555,
                -0.08434
            ),
            epsilon = 0.001
        );
    }

    #[test]
    fn forward_dynamics_ab_test() {
        let c1 = Isometry3::<f64>::identity();
        let q12 = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), -PI / 2.0);
        let q21 = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), PI / 2.0);

        let l1 = 0.62;
        let l2 = 0.10;

        let c2 = Isometry3::from_parts(Translation3::new(l1, 0.0, 0.0), UnitQuaternion::identity());
        let c3 = Isometry3::from_parts(Translation3::new(l2, 0.0, 0.0), q12);
        let c4 = Isometry3::from_parts(Translation3::new(l1, 0.0, 0.0), q21);

        // let offset_matrices = vec![c0, c1, c2, c1, c2, c1, c2, c1, c2];
        let offset_matrices = vec![c1, c2, c3, c4, c3, c4, c3, c4, c3];
        // let offset_matrices = vec![c1, c2, c4, c3, c4, c3, c4, c3, c4];

        let mut joint_types = vec![JointType::Revolute; 9];
        joint_types[0] = JointType::SixDOF;
        let parent: Vec<u16> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];

        let r_cg1 = Vector3::new(l1 / 2.0, 0.0, 0.0);
        let r_cg2 = Vector3::new(l2 / 2.0, 0.0, 0.0);

        let m1 = PI * 0.09 * 0.09 * l1 * 1000.0;
        let m2 = PI * 0.09 * 0.09 * l2 * 1000.0;

        let inertia_mat1 = Matrix3::new(
            1.0 / 2.0 * m1 * 0.09 * 0.09,
            0.0,
            0.0,
            0.0,
            1.0 / 12.0 * m1 * (3.0 * 0.09 * 0.09 + l1 * l1),
            0.0,
            0.0,
            0.0,
            1.0 / 12.0 * m1 * (3.0 * 0.09 * 0.09 + l1 * l1),
        ) - m1 * skew(&r_cg1) * skew(&r_cg1);

        let inertia_mat2 = Matrix3::new(
            1.0 / 2.0 * m2 * 0.09 * 0.09,
            0.0,
            0.0,
            0.0,
            1.0 / 12.0 * m2 * (3.0 * 0.09 * 0.09 + l2 * l2),
            0.0,
            0.0,
            0.0,
            1.0 / 12.0 * m2 * (3.0 * 0.09 * 0.09 + l2 * l2),
        ) - m2 * skew(&r_cg2) * skew(&r_cg2);

        let mut M_1 = comp_mass_matrix(m1, &r_cg1, &inertia_mat1);
        M_1[(0, 0)] = 10.0;
        let M_2 = comp_mass_matrix(m2, &r_cg2, &inertia_mat2);

        let mass_matrices = vec![M_1, M_2, M_1, M_2, M_1, M_2, M_1, M_2, M_1];

        // let mut multibody =
        //     MultiBody::<9, 14>::new(offset_matrices, Some(mass_matrices), joint_types, parent);
        // let mut multibody: MultiBody<9, 14> =
        //     MultiBody::new(offset_matrices, Some(mass_matrices), None, joint_types, parent, Vector3::new(0.0, 0.0, 9.81), Some(vec![Vector3::zeros();9]), Some(vec![Vector3::zeros(); 9]), Some(vec![1.0;9]), Some(vec![1.0;9]), Some(1000.0)).unwrap();
        let mut multibody: MultiBody<9, 14> = MultiBody::new(
            offset_matrices,
            Some(mass_matrices),
            None,
            None,
            joint_types,
            parent,
            Vector3::new(0.0, 0.0, 0.0),
            Some(vec![Vector3::zeros(); 9]),
            Some(vec![Vector3::zeros(); 9]),
            Some(vec![1.0; 9]),
            Some(vec![1.0; 9]),
            Some(1000.0),
        )
        .unwrap();

        // let mut conf: Vec<Isometry3<f64>> = Vec::new();

        let configuration_base = Isometry3::identity();
        // let configuration_base = Isometry3::from_parts(Translation3::new(0.0, 0.0, 0.0), UnitQuaternion::from_axis_angle(&Vector3::x_axis(), PI / 2.0)));

        // let joint_angles = SVector::<f64, 8>::zeros();
        // let joint_angles = SVector::<f64, 8>::from_vec(vec![1.0; 8]);
        let joint_angles = SVector::<f64, 8>::from_vec(vec![
            PI / 4.0,
            PI / 3.0,
            PI / 5.0,
            PI / 7.0,
            PI / 15.0,
            PI / 10.0,
            PI / 4.0,
            PI / 2.5,
        ]);

        let conf =
            multibody.minimal_to_homogenous_configuration(&configuration_base, &joint_angles);

        let mu = SVector::<f64, 14>::repeat(1.0);

        // let damping_func = |x: &Vector6<f64>, i: usize| -> Box<dyn Fn(&Vector6<f64>) -> Vector6<f64> > {
        //     Box::new(0.0 * Vector6::<f64>::zeros())
        // };
        let rigid_body_forces_func1 =
            &|x: &[Vector6<f64>], y: &[Vector6<f64>]| -> SMatrix<f64, 6, 9> { SMatrix::<f64, 6, 9>::zeros()};
        let rigid_body_forces_func2 =
            &|x: &[Isometry3<f64>], y: &[Vector6<f64>]| -> SMatrix<f64, 6, 9> { SMatrix::<f64, 6, 9>::zeros()};
        // let lambda = |x: usize| -> i32 { self.parent[x] as i32 - 1 };

        let thruster_forces = vec![Vector6::<f64>::zeros(); 9];
        let eta = SVector::<f64, 14>::zeros();
        let lin_vel_current = SVector::<f64, 3>::zeros();
        let lin_accel_current = SVector::<f64, 3>::zeros();

        let accel = multibody.forward_dynamics_ab(
            &conf,
            &mu,
            rigid_body_forces_func2,
            &thruster_forces,
            &eta,
            &lin_vel_current,
            &lin_accel_current,
        );

        let sigma_prime = SVector::<f64, 14>::zeros();


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


        let lin_vel_current = Vector3::new(10.0, 20.0, 30.0);
        let accel3 = multibody.forward_dynamics_ab(
            &conf,
            &mu,
            rigid_body_forces_func2,
            &thruster_forces,
            &eta,
            &lin_vel_current,
            &lin_accel_current,
        );

        println!("accel: {}", accel);
        println!("accel2: {}", accel2);
        assert_relative_eq!(accel, accel2, epsilon = 1e-7);

    }
}
