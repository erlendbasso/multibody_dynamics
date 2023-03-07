// #![no_std]
#![cfg_attr(feature = "no_std", no_std)]

#[cfg(feature = "no_std")]
extern crate core as std;

extern crate nalgebra as na;
use crate::math_functions::*;
use na::{
    DMatrix, DVector, Isometry3, Matrix1, Matrix3, Matrix4, Matrix6, Quaternion, SMatrix, SVector,
    Translation3, UnitQuaternion, Vector1, Vector3, Vector6,
};

#[derive(Clone, Debug)]
pub enum JointType {
    Revolute,
    Prismatic,
    SixDOF,
}

pub struct MultiBody<const NUM_BODIES: usize, const NUM_DOFS: usize> {
    offset_matrices: Vec<Isometry3<f64>>,
    mass_matrices: Vec<Matrix6<f64>>,
    joint_types: Vec<JointType>,
    parent: Vec<u16>,
    g: Vec<Isometry3<f64>>,
    h: Vec<Isometry3<f64>>,
    nu: Vec<Vector6<f64>>,
    nu_prime: Vec<Vector6<f64>>,
    alpha: Vec<Vector6<f64>>,
    Phi: SMatrix<f64, 6, NUM_DOFS>,
    // joint_dims: SVector<usize, NUM_BODIES>,
    joint_size_offsets: Vec<usize>,
}

impl<const NUM_BODIES: usize, const NUM_DOFS: usize> MultiBody<NUM_BODIES, NUM_DOFS> {
    pub fn new(
        offset_matrices: Vec<Isometry3<f64>>,
        mass_matrices: Vec<Matrix6<f64>>,
        joint_types: Vec<JointType>,
        parent: Vec<u16>,
    ) -> MultiBody<NUM_BODIES, NUM_DOFS> {
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

                    joint_size_offsets += joint_dims[i] as usize - 1;
                }
            }
        }

        MultiBody {
            offset_matrices: offset_matrices,
            mass_matrices: mass_matrices,
            joint_types: joint_types,
            parent: parent,
            g: vec![Isometry3::identity(); NUM_BODIES],
            h: vec![Isometry3::identity(); NUM_BODIES],
            nu: vec![Vector6::zeros(); NUM_BODIES],
            nu_prime: vec![Vector6::zeros(); NUM_BODIES],
            alpha: vec![Vector6::zeros(); NUM_BODIES],
            Phi: Phi,
            // joint_dims: joint_dims,
            joint_size_offsets: joint_offset_vec,
        }
    }

    pub fn minimal_to_homogenous_configuration<const D: usize>(
        &self,
        six_dof_vars: &Vec<Isometry3<f64>>,
        scalar_joint_vars: &SVector<f64, D>,
    ) -> Vec<Isometry3<f64>> {
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
                    j = j + 1;
                }
                JointType::Prismatic => {
                    let mut temp = Isometry3::identity();
                    temp.translation = Translation3::new(0.0, 0.0, scalar_joint_vars[j]);

                    conf[i] = temp;
                    j = j + 1;
                }
                JointType::SixDOF => {
                    conf[i] = six_dof_vars[k];
                    k = k + 1;
                }
            }
        }
        conf
    }

    pub fn generalized_newton_euler(
        &mut self,
        conf: &Vec<Isometry3<f64>>,
        mu: &SVector<f64, NUM_DOFS>,
        mu_prime: &SVector<f64, NUM_DOFS>,
        sigma_prime: &SVector<f64, NUM_DOFS>,
        rigid_body_forces: &Vec<Vector6<f64>>,
        eta: &SVector<f64, NUM_DOFS>,
    ) -> SVector<f64, NUM_DOFS> {
        let mut w: Vec<Vector6<f64>> = vec![Vector6::zeros(); NUM_BODIES];
        let mut zeta = SVector::<f64, NUM_DOFS>::zeros();

        for i in 0..NUM_BODIES {
            match self.joint_types[i] {
                JointType::Revolute | JointType::Prismatic => {
                    let idx = i + self.joint_size_offsets[i];
                    let mu_i = mu.fixed_rows::<1>(idx).try_into().unwrap();
                    let mu_prime_i = mu_prime.fixed_rows::<1>(idx).try_into().unwrap();
                    let sigma_prime_i = sigma_prime.fixed_rows::<1>(idx).try_into().unwrap();

                    w[i] = self.forward_step(
                        &conf[i],
                        &mu_i,
                        &mu_prime_i,
                        &sigma_prime_i,
                        &rigid_body_forces[i],
                        i,
                    );
                }
                JointType::SixDOF => {
                    let idx = i + self.joint_size_offsets[i];
                    let mu_i = mu.fixed_rows::<6>(idx).try_into().unwrap();
                    let mu_prime_i = mu_prime.fixed_rows::<6>(idx).try_into().unwrap();
                    let sigma_prime_i = sigma_prime.fixed_rows::<6>(idx).try_into().unwrap();

                    w[i] = self.forward_step(
                        &conf[i],
                        &mu_i,
                        &mu_prime_i,
                        &sigma_prime_i,
                        &rigid_body_forces[i],
                        i,
                    );
                }
            }
        }

        // backward step
        for i in (0..NUM_BODIES).rev() {
            match self.joint_types[i] {
                JointType::Revolute | JointType::Prismatic => {
                    let idx = i + self.joint_size_offsets[i];
                    let Phi_i = self.Phi.fixed_view::<6, 1>(0, idx);
                    // zeta[i] = (Phi_i.transpose() * w[i])[(0, 0)] + eta[idx];
                    zeta.fixed_view_mut::<1, 1>(idx, 0).copy_from(&((Phi_i.transpose() * w[i]) + Matrix1::new(eta[idx])));
                }
                JointType::SixDOF => {
                    let idx = i + self.joint_size_offsets[i];
                    let Phi_i = self.Phi.fixed_view::<6, 6>(0, idx);
                    zeta.fixed_view_mut::<6, 1>(idx, 0)
                        .copy_from(&(Phi_i.transpose() * w[i] + eta.fixed_rows::<6>(idx)));
                }
            }

            let lambda_i = self.parent[i] as i32 - 1;
            if lambda_i >= 0 {
                w[lambda_i as usize] =
                    w[lambda_i as usize] + Ad(&self.h[i].inverse()).transpose() * w[i];
            }
        }
        zeta
    }

    fn forward_step<const D: usize>(
        &mut self,
        conf: &Isometry3<f64>,
        mu: &SVector<f64, D>,
        mu_prime: &SVector<f64, D>,
        sigma_prime: &SVector<f64, D>,
        rigid_body_forces: &SVector<f64, 6>,
        node_idx: usize,
    ) -> Vector6<f64> {
        let idx_offset = self.joint_size_offsets[node_idx];

        let h = self.offset_matrices[node_idx] * conf;
        // println!("offset mat ind: {} : {}", node_idx, self.offset_matrices[node_idx].to_homogeneous());
        self.h[node_idx] = h;
        // println!("h[{}] {}", node_idx, self.h[node_idx].to_homogeneous());

        let Phi: SMatrix<f64, 6, D> = self
            .Phi
            .fixed_view::<6, D>(0, node_idx + idx_offset)
            .try_into()
            .unwrap();

        let lambda_i = self.parent[node_idx] as i32 - 1;

        if lambda_i < 0 {
            self.g[node_idx] = self.h[node_idx];
            self.nu[node_idx] = Phi * mu;
            self.nu_prime[node_idx] = Phi * mu_prime;

            match self.joint_types[node_idx] {
                JointType::Revolute | JointType::Prismatic => {
                    self.alpha[node_idx] =
                        ad_se3(&self.nu_prime[node_idx]) * Phi * mu + Phi * sigma_prime;
                }
                JointType::SixDOF => {
                    let mu: SVector<f64, 6> = mu.fixed_rows::<6>(0).try_into().unwrap();
                    let mu_prime: SVector<f64, 6> = mu_prime.fixed_rows::<6>(0).try_into().unwrap();
                    let sigma_prime: SVector<f64, 6> =
                        sigma_prime.fixed_rows::<6>(0).try_into().unwrap();
                    let Phi: SMatrix<f64, 6, 6> = Phi.fixed_view::<6, 6>(0, 0).try_into().unwrap();

                    self.alpha[node_idx] = ad_se3(&self.nu_prime[node_idx]) * Phi * mu
                        + Phi * (sigma_prime + ad_se3(&mu) * mu_prime);
                }
            }
        } else {
            self.g[node_idx] = self.g[lambda_i as usize] * self.h[node_idx];

            let Ad_h_inv = Ad(&self.h[node_idx].inverse());

            self.nu[node_idx] = Ad_h_inv * self.nu[lambda_i as usize] + Phi * mu;

            self.nu_prime[node_idx] = Ad_h_inv * self.nu_prime[lambda_i as usize] + Phi * mu_prime;

            match self.joint_types[node_idx] {
                JointType::Revolute | JointType::Prismatic => {
                    self.alpha[node_idx] = Ad_h_inv * self.alpha[lambda_i as usize]
                        + ad_se3(&self.nu_prime[node_idx]) * Phi * mu
                        + Phi * sigma_prime;
                }
                JointType::SixDOF => {
                    let mu: SVector<f64, 6> = mu.fixed_rows::<6>(0).try_into().unwrap();
                    let mu_prime: SVector<f64, 6> = mu_prime.fixed_rows::<6>(0).try_into().unwrap();
                    let sigma_prime: SVector<f64, 6> =
                        sigma_prime.fixed_rows::<6>(0).try_into().unwrap();
                    let Phi: SMatrix<f64, 6, 6> = Phi.fixed_view::<6, 6>(0, 0).try_into().unwrap();

                    self.alpha[node_idx] = Ad_h_inv * self.alpha[lambda_i as usize]
                        + ad_se3(&self.nu_prime[node_idx]) * Phi * mu
                        + Phi * (sigma_prime + ad_se3(&mu) * mu_prime);
                }
            }
        }

        // println!("alpha[{}] {}", node_idx, self.alpha[node_idx].to_homogeneous());

        self.mass_matrices[node_idx] * self.alpha[node_idx]
            - ad_se3(&self.nu_prime[node_idx]).transpose()
                * self.mass_matrices[node_idx]
                * self.nu[node_idx]
            + rigid_body_forces
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

        // let joint_types = vec![JointType::SixDOF, JointType::Revolute];
        let joint_types = vec![JointType::Revolute, JointType::Revolute];
        let parent = vec![0, 1];

        let mut multibody: MultiBody<2, 2> =
            MultiBody::new(offset_matrices, mass_matrices, joint_types, parent);

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
        let rigid_body_forces = vec![Vector6::<f64>::zeros(); 2];

        let zeta = multibody.generalized_newton_euler(
            &conf,
            &mu,
            &mu,
            &sigma_prime,
            &rigid_body_forces,
            &eta,
        );

        println!("zeta: {}", zeta);
        assert_relative_eq!(zeta, Vector2::new(-0.5, 0.0), epsilon = 0.00001);
    }

    #[test]
    fn snake_like_model_test() {
        let c1 = Isometry3::<f64>::identity();
        let q12 = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), - PI / 2.0);
        let q21 = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), PI / 2.0);

        let l1 = 0.62;
        let l2 = 0.10;

        let c2 = Isometry3::from_parts(Translation3::new(l1, 0.0, 0.0), UnitQuaternion::identity());
        let c3 = Isometry3::from_parts(Translation3::new(l2, 0.0, 0.0), q12);
        let c4 = Isometry3::from_parts(Translation3::new(l1, 0.0, 0.0), q21);


        // let offset_matrices = vec![c0, c1, c2, c1, c2, c1, c2, c1, c2];
        let offset_matrices = vec![c1, c2, c3, c4, c3, c4, c3, c4, c3];
        // let offset_matrices = vec![c1, c2, c4, c3, c4, c3, c4, c3, c4];

        let mut joint_types =  vec![JointType::Revolute; 9];
        joint_types[0] = JointType::SixDOF;
        let parent: Vec<u16> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];

        let r_cg1 = Vector3::new(l1/2.0, 0.0, 0.0);
        let r_cg2 = Vector3::new(l2/2.0, 0.0, 0.0);

        let m1 = PI * 0.09 * 0.09 * l1 * 1000.0;
        let m2 = PI * 0.09 * 0.09 * l2 * 1000.0;

        let inertia_mat1 = Matrix3::new(
            1.0/2.0 * m1 * 0.09 * 0.09,
            0.0,
            0.0,
            0.0,
            1.0/12.0 * m1 * (3.0 * 0.09 * 0.09 + l1 * l1),
            0.0,
            0.0,
            0.0,
            1.0/12.0 * m1 * (3.0 * 0.09 * 0.09 + l1 * l1),
        ) - m1 * skew(&r_cg1) * skew(&r_cg1);

        let inertia_mat2 = Matrix3::new(
            1.0/2.0 * m2 * 0.09 * 0.09,
            0.0,
            0.0,
            0.0,
            1.0/12.0 * m2 * (3.0 * 0.09 * 0.09 + l2 * l2),
            0.0,
            0.0,
            0.0,
            1.0/12.0 * m2 * (3.0 * 0.09 * 0.09 + l2 * l2),
        )  - m2 * skew(&r_cg2) * skew(&r_cg2);

        let M_1 = comp_mass_matrix(m1, &r_cg1, &inertia_mat1);
        let M_2 = comp_mass_matrix(m2, &r_cg2, &inertia_mat2);

        println!("M_1: {}", M_1);
        

        let mass_matrices = vec![M_1, M_2, M_1, M_2, M_1, M_2, M_1, M_2, M_1];

        let mut multibody = MultiBody::<9, 14>::new(offset_matrices, mass_matrices, joint_types, parent);

        // let mut conf: Vec<Isometry3<f64>> = Vec::new();

        let configuration_base = Isometry3::identity();
        // let configuration_base = Isometry3::from_parts(Translation3::new(0.0, 0.0, 0.0), UnitQuaternion::from_axis_angle(&Vector3::x_axis(), PI / 2.0)));

        let joint_angles = SVector::<f64, 8>::zeros();

        let conf = multibody.minimal_to_homogenous_configuration(&vec![configuration_base], &joint_angles);
        
        // let mu = SVector::<f64, 14>::zeros();
        let mu = SVector::<f64, 14>::repeat(1.0);
        // let mu_prime = SVector::<f64, 14>::repeat(1.0);
        let mu_prime = SVector::<f64, 14>::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
        // let mu_prime = SVector::<f64, 14>::repeat(1.0);
        // let mu = SVector::<f64, 14>::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        // println!("mu: {}", mu);
        let sigma_prime = SVector::<f64, 14>::zeros();
        // let sigma_prime = SVector::<f64, 14>::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let eta = SVector::<f64, 14>::zeros();
        let rigid_body_forces = vec![Vector6::<f64>::zeros(); 9];

        let zeta = multibody.generalized_newton_euler(
            &conf,
            &mu,
            &mu_prime,
            &sigma_prime,
            &rigid_body_forces,
            &eta,
        );

        println!("zeta: {}", zeta);

    }
}
