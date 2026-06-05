#![allow(deprecated)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use multibody_dynamics::multibody::{Axis, JointType, MultiBody};
use nalgebra::{Isometry3, Matrix3, SVector, Vector3, Vector6};

fn build_chain<const N: usize, const DOFS: usize>() -> MultiBody<N, DOFS> {
    // Simple chain: first joint 6DOF, rest revolute Z
    let mut joint_types = vec![JointType::Revolute(Axis::Z); N];
    joint_types[0] = JointType::SixDOF;
    let parent: Vec<u16> = (0..N as u16).collect();
    let offset_mats = (0..N).map(|_| Isometry3::identity()).collect();
    let masses = vec![1.0; N];
    let r_cg = vec![Vector3::zeros(); N];
    let inertia = Matrix3::identity();
    let inertia_vec = vec![inertia; N];
    MultiBody::<N, DOFS>::new(
        offset_mats,
        None,
        None,
        Some(inertia_vec),
        joint_types,
        parent,
        Vector3::new(0.0, 0.0, 9.81),
        Some(r_cg.clone()),
        None,
        Some(masses),
        None,
        None,
    )
    .unwrap()
}

fn bench_forward_dynamics(c: &mut Criterion) {
    let mut group = c.benchmark_group("forward_dynamics_chain");
    for &n in &[5usize, 10, 20, 40] {
        macro_rules! run_case {
            ($n:literal, $d:literal) => {{
                let mb = build_chain::<$n, $d>();
                let base = Isometry3::identity();
                let joint_angles = SVector::<f64, { $d - 6 }>::from_vec(vec![0.1; $d - 6]);
                let conf = mb.minimal_to_homogeneous_configuration(&base, &joint_angles);
                let mu = SVector::<f64, $d>::repeat(0.01);
                let thruster = vec![Vector6::zeros(); $n];
                let eta = SVector::<f64, $d>::zeros();
                let zero3 = Vector3::zeros();
                let rb = |_: &[Isometry3<f64>], _: &[Vector6<f64>]| {
                    nalgebra::SMatrix::<f64, 6, $n>::zeros()
                };
                group.bench_with_input(BenchmarkId::new("N", $n), &n, |b, _| {
                    b.iter(|| {
                        let acc = mb.forward_dynamics_ab(
                            black_box(&conf),
                            black_box(&mu),
                            black_box(rb),
                            black_box(&thruster),
                            black_box(&eta),
                            black_box(&zero3),
                            black_box(&zero3),
                        );
                        black_box(acc);
                    });
                });
            }};
        }
        match n {
            5 => run_case!(5, 10), // 6 + 4 revolute
            10 => run_case!(10, 15),
            20 => run_case!(20, 25),
            40 => run_case!(40, 45),
            _ => unreachable!(),
        }
    }
    group.finish();
}

fn bench_mass_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("mass_matrix_chain");
    for &n in &[5usize, 10, 20, 40] {
        macro_rules! run_case {
            ($n:literal, $d:literal) => {{
                let mb = build_chain::<$n, $d>();
                let base = Isometry3::identity();
                let joint_angles = SVector::<f64, { $d - 6 }>::from_vec(vec![0.1; $d - 6]);
                let conf = mb.minimal_to_homogeneous_configuration(&base, &joint_angles);
                group.bench_with_input(BenchmarkId::new("N", $n), &n, |b, _| {
                    b.iter(|| {
                        let m = mb.compute_mass_matrix(black_box(&conf));
                        black_box(m);
                    });
                });
            }};
        }
        match n {
            5 => run_case!(5, 10),
            10 => run_case!(10, 15),
            20 => run_case!(20, 25),
            40 => run_case!(40, 45),
            _ => unreachable!(),
        }
    }
    group.finish();
}

fn bench_gne(c: &mut Criterion) {
    let mut group = c.benchmark_group("generalized_newton_euler");
    for &n in &[5usize, 10, 20, 40] {
        macro_rules! run_case {
            ($n:literal, $d:literal) => {{
                let mb = build_chain::<$n, $d>();
                let base = Isometry3::identity();
                let joint_angles = SVector::<f64, { $d - 6 }>::from_vec(vec![0.1; $d - 6]);
                let conf = mb.minimal_to_homogeneous_configuration(&base, &joint_angles);
                let mu = SVector::<f64, $d>::repeat(0.01);
                let sigma_prime = SVector::<f64, $d>::zeros();
                let eta = SVector::<f64, $d>::zeros();
                let rb = |_: &[Vector6<f64>], _: &[Vector6<f64>]| {
                    nalgebra::SMatrix::<f64, 6, $n>::zeros()
                };
                group.bench_with_input(BenchmarkId::new("N", $n), &n, |b, _| {
                    b.iter(|| {
                        let z = mb.generalized_newton_euler(
                            black_box(&conf),
                            black_box(&mu),
                            black_box(&mu),
                            black_box(&sigma_prime),
                            black_box(rb),
                            black_box(&eta),
                        );
                        black_box(z);
                    });
                });
            }};
        }
        match n {
            5 => run_case!(5, 10),
            10 => run_case!(10, 15),
            20 => run_case!(20, 25),
            40 => run_case!(40, 45),
            _ => unreachable!(),
        }
    }
    group.finish();
}

fn bench_jacobians(c: &mut Criterion) {
    let mut group = c.benchmark_group("jacobians_chain");
    for &n in &[5usize, 10, 20, 40] {
        macro_rules! run_case {
            ($n:literal, $d:literal) => {{
                let mb = build_chain::<$n, $d>();
                let base = Isometry3::identity();
                let joint_angles = SVector::<f64, { $d - 6 }>::from_vec(vec![0.1; $d - 6]);
                let conf = mb.minimal_to_homogeneous_configuration(&base, &joint_angles);
                group.bench_with_input(BenchmarkId::new("N", $n), &n, |b, _| {
                    b.iter(|| {
                        let jacs = mb.compute_jacobians(black_box(&conf));
                        black_box(jacs);
                    });
                });
            }};
        }
        match n {
            5 => run_case!(5, 10),
            10 => run_case!(10, 15),
            20 => run_case!(20, 25),
            40 => run_case!(40, 45),
            _ => unreachable!(),
        }
    }
    group.finish();
}

fn bench_jacobian_derivatives(c: &mut Criterion) {
    let mut group = c.benchmark_group("jacobian_derivatives_chain");
    for &n in &[5usize, 10, 20, 40] {
        macro_rules! run_case {
            ($n:literal, $d:literal) => {{
                let mb = build_chain::<$n, $d>();
                let base = Isometry3::identity();
                let joint_angles = SVector::<f64, { $d - 6 }>::from_vec(vec![0.1; $d - 6]);
                let conf = mb.minimal_to_homogeneous_configuration(&base, &joint_angles);
                let mu = SVector::<f64, $d>::repeat(0.01);
                // Precompute Jacobians once; benchmark derivative-only and full pipeline.
                let jacs = mb.compute_jacobians(&conf);
                group.bench_with_input(BenchmarkId::new("derivative_only", $n), &n, |b, _| {
                    b.iter(|| {
                        let djacs = mb.compute_jacobian_derivatives(
                            black_box(&jacs),
                            black_box(&conf),
                            black_box(&mu),
                        );
                        black_box(djacs);
                    });
                });
                group.bench_with_input(BenchmarkId::new("baseline_full", $n), &n, |b, _| {
                    b.iter(|| {
                        let j_full = mb.compute_jacobians(black_box(&conf));
                        let dj_full = mb.compute_jacobian_derivatives(
                            black_box(&j_full),
                            black_box(&conf),
                            black_box(&mu),
                        );
                        black_box((j_full, dj_full));
                    });
                });
            }};
        }
        match n {
            5 => run_case!(5, 10),
            10 => run_case!(10, 15),
            20 => run_case!(20, 25),
            40 => run_case!(40, 45),
            _ => unreachable!(),
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_forward_dynamics,
    bench_mass_matrix,
    bench_gne,
    bench_jacobians,
    bench_jacobian_derivatives
);
criterion_main!(benches);
