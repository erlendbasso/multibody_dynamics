#![allow(non_snake_case)]
#![allow(dead_code)]
extern crate nalgebra as na;

use na::{Dyn, Isometry3, Matrix3, Matrix6, OMatrix, OVector, SMatrix, Vector3, Vector6};

pub fn Ad_inv(h: &Isometry3<f64>) -> Matrix6<f64> {
    let mut Ad_h_inv = Matrix6::zeros();
    // let R_inv = h.rotation.to_rotation_matrix().matrix().transpose();
    // let R_inv = h.rotation.to_rotation_matrix()
    let h_inv = h.inverse().to_homogeneous();
    let R_inv = h_inv.fixed_view::<3, 3>(0, 0);
    let p: Vector3<f64> = h_inv.fixed_view::<3, 1>(0, 3).into();

    Ad_h_inv.fixed_view_mut::<3, 3>(0, 0).copy_from(&R_inv);

    Ad_h_inv
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(skew(&p) * R_inv));
    Ad_h_inv.fixed_view_mut::<3, 3>(3, 3).copy_from(&R_inv);
    Ad_h_inv
}

pub fn Ad(h: &Isometry3<f64>) -> Matrix6<f64> {
    let mut Ad_h = Matrix6::zeros();
    let h = h.to_homogeneous();
    let R = h.fixed_view::<3, 3>(0, 0);
    let p: Vector3<f64> = h.fixed_view::<3, 1>(0, 3).into();

    Ad_h.fixed_view_mut::<3, 3>(0, 0).copy_from(&R);

    Ad_h.fixed_view_mut::<3, 3>(0, 3).copy_from(&(skew(&p) * R));
    Ad_h.fixed_view_mut::<3, 3>(3, 3).copy_from(&R);
    Ad_h
}

pub fn skew<T: na::RealField + Copy>(v: &Vector3<T>) -> Matrix3<T> {
    let mut skew = Matrix3::<T>::zeros();
    skew[(0, 1)] = -v[2];
    skew[(0, 2)] = v[1];
    skew[(1, 0)] = v[2];
    skew[(1, 2)] = -v[0];
    skew[(2, 0)] = -v[1];
    skew[(2, 1)] = v[0];
    skew
}

pub fn ad_se3(v: &Vector6<f64>) -> SMatrix<f64, 6, 6> {
    let mut ad = SMatrix::<f64, 6, 6>::zeros();
    let lin_vel = v.fixed_view::<3, 1>(0, 0).into();
    let ang_vel = v.fixed_view::<3, 1>(3, 0).into();

    ad.fixed_view_mut::<3, 3>(0, 0).copy_from(&skew(&ang_vel));
    ad.fixed_view_mut::<3, 3>(0, 3).copy_from(&skew(&lin_vel));
    ad.fixed_view_mut::<3, 3>(3, 3).copy_from(&skew(&ang_vel));

    ad
}

pub fn ad_se3_dyn(v: &OVector<f64, Dyn>) -> OMatrix<f64, Dyn, Dyn> {
    let mut ad = OMatrix::<f64, Dyn, Dyn>::zeros(6, 6);
    // let mut ad = OMatrix::<f64>::zeros(6, 6);
    // let lin_vel = v.fixed_view::<3, 1>(0, 0).into();
    // let ang_vel = v.fixed_view::<3, 1>(3, 0).into();
    let lin_vel = v.rows(0, 3).into();
    let ang_vel = v.rows(3, 3).into();

    // ad.fixed_view_mut::<3, 3>(0, 0).copy_from(&skew(&ang_vel));
    // ad.fixed_view_mut::<3, 3>(0, 3).copy_from(&skew(&lin_vel));
    // ad.fixed_view_mut::<3, 3>(3, 3).copy_from(&skew(&ang_vel));
    ad.view_mut((0, 0), (3, 3)).copy_from(&skew_dyn(&ang_vel));
    ad.view_mut((0, 3), (3, 3)).copy_from(&skew_dyn(&lin_vel));
    ad.view_mut((3, 3), (3, 3)).copy_from(&skew_dyn(&ang_vel));

    ad
}

fn skew_dyn(v: &OVector<f64, Dyn>) -> Matrix3<f64> {
    let mut skew = Matrix3::zeros();
    skew[(0, 1)] = -v[2];
    skew[(0, 2)] = v[1];
    skew[(1, 0)] = v[2];
    skew[(1, 2)] = -v[0];
    skew[(2, 0)] = -v[1];
    skew[(2, 1)] = v[0];
    skew
}

pub fn comp_rb_mass_matrix(m: f64, r: &Vector3<f64>, inertia_mat: &Matrix3<f64>) -> Matrix6<f64> {
    let mut mass_matrix = Matrix6::zeros();
    let skew_r = skew(r);

    mass_matrix
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(m * Matrix3::identity()));
    mass_matrix
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-m * skew_r));
    mass_matrix
        .fixed_view_mut::<3, 3>(3, 0)
        .copy_from(&(m * skew_r));
    mass_matrix
        .fixed_view_mut::<3, 3>(3, 3)
        .copy_from(inertia_mat);
    mass_matrix
}
