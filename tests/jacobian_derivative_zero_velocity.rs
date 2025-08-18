use multibody_dynamics::multibody::*;
use nalgebra as na;
use approx::assert_relative_eq;

#[test]
fn jacobian_derivative_zero_when_static() {
    // Chain: SixDOF + revolute
    let offsets = vec![na::Isometry3::identity(), na::Isometry3::identity()];
    let mut joint_types = vec![JointType::Revolute(Axis::Z);2]; joint_types[0]=JointType::SixDOF;
    let parent = vec![0u16,1u16];
    let masses = vec![1.0,1.0];
    let r_cg = vec![na::Vector3::zeros();2];
    let inertia = vec![na::Matrix3::identity();2];
    let mb: MultiBody<2,7> = MultiBody::new(offsets,None,None,Some(inertia),joint_types,parent,na::Vector3::zeros(),Some(r_cg),None,Some(masses),None,None).unwrap();

    let base = na::Isometry3::identity();
    let joint_angles = na::SVector::<f64,1>::from_vec(vec![0.5]);
    let conf = mb.minimal_to_homogeneous_configuration(&base,&joint_angles);

    let mu = na::SVector::<f64,7>::zeros();
    let djac = mb.compute_jacobian_derivative(&conf,&mu,1);
    assert_relative_eq!(djac, na::SMatrix::<f64,6,7>::zeros(), epsilon=1e-14);
}
