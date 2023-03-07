extern crate nalgebra as na;
use na::{SMatrix, Vector3, Vector6, Matrix3, Matrix6, Isometry3};


pub fn Ad_inv(h: &Isometry3<f64>) -> Matrix6<f64> {
    let mut Ad_h_inv = Matrix6::zeros();
    // let R_inv = h.rotation.to_rotation_matrix().matrix().transpose();
    // let R_inv = h.rotation.to_rotation_matrix()
    let h_inv = h.inverse().to_homogeneous();
    let R_inv = h_inv.fixed_view::<3, 3>(0, 0);
    let p: Vector3<f64> = h_inv.fixed_view::<3, 1>(0, 3).try_into().unwrap();

    Ad_h_inv.fixed_view_mut::<3, 3>(0, 0).copy_from(&R_inv);

    Ad_h_inv
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(skew(&p) * &R_inv));
    Ad_h_inv.fixed_view_mut::<3, 3>(3, 3).copy_from(&R_inv);
    Ad_h_inv
}

pub fn Ad(h: &Isometry3<f64>) -> Matrix6<f64> {
    let mut Ad_h = Matrix6::zeros();
    let h = h.to_homogeneous();
    let R = h.fixed_view::<3, 3>(0, 0);
    let p: Vector3<f64> = h.fixed_view::<3, 1>(0, 3).try_into().unwrap();

    Ad_h.fixed_view_mut::<3, 3>(0, 0).copy_from(&R);

    Ad_h
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(skew(&p) * &R));
    Ad_h.fixed_view_mut::<3, 3>(3, 3).copy_from(&R);
    Ad_h
}

pub fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    let mut skew = Matrix3::zeros();
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
    let lin_vel = v.fixed_view::<3, 1>(0, 0).try_into().unwrap();
    let ang_vel = v.fixed_view::<3, 1>(3, 0).try_into().unwrap();

    ad.fixed_view_mut::<3, 3>(0, 0).copy_from(&skew(&ang_vel));
    ad.fixed_view_mut::<3, 3>(0, 3).copy_from(&skew(&lin_vel));
    ad.fixed_view_mut::<3, 3>(3, 3).copy_from(&skew(&ang_vel));

    ad
}