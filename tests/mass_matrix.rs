use approx::assert_relative_eq;
use multibody_dynamics::math_functions::skew;
use multibody_dynamics::multibody::*;
use nalgebra as na;

#[test]
fn mass_matrix_symmetry_and_basic_values() {
    // 2-body planar-like chain about Z.
    let m1 = 2.0;
    let m2 = 3.0;
    let l1 = 1.0;
    let l2 = 0.7;
    let r1 = na::Vector3::new(l1 / 2.0, 0.0, 0.0);
    let r2 = na::Vector3::new(l2 / 2.0, 0.0, 0.0);
    let offset_matrices = vec![
        na::Isometry3::identity(),
        na::Isometry3::from_parts(
            na::Translation3::new(l1, 0.0, 0.0),
            na::UnitQuaternion::identity(),
        ),
    ];
    let joint_types = vec![JointType::Revolute(Axis::Z), JointType::Revolute(Axis::Z)];
    let parent = vec![0u16, 1u16];
    // Simple diagonal-ish inertia (not physically exact, fine for test)
    let i1 = na::Matrix3::identity() * 0.1 - m1 * skew(&r1) * skew(&r1);
    let i2 = na::Matrix3::identity() * 0.2 - m2 * skew(&r2) * skew(&r2);
    let inertia = vec![i1, i2];
    let masses = vec![m1, m2];
    let r_com = vec![r1, r2];
    let mb: MultiBody<2, 2> = MultiBody::new(
        offset_matrices,
        None,
        None,
        Some(inertia),
        joint_types,
        parent,
        na::Vector3::new(0.0, 0.0, -9.81),
        Some(r_com),
        None,
        Some(masses),
        None,
        None,
    )
    .unwrap();

    // Config with both revolute angles = 0
    let base = na::Isometry3::identity();
    let joint_angles = na::SVector::<f64, 2>::zeros();
    let conf = mb.minimal_to_homogeneous_configuration(&base, &joint_angles);
    let m_mat = mb.compute_mass_matrix(&conf);

    // Symmetry
    assert_relative_eq!(m_mat, m_mat.transpose(), epsilon = 1e-10);
    // Positive diagonal entries
    for i in 0..2 {
        assert!(m_mat[(i, i)] > 0.0);
    }
}
