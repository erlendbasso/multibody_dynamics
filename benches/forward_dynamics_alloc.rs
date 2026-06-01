#![allow(deprecated)]

use criterion::{criterion_group, criterion_main, Criterion};
use multibody_dynamics::multibody::{Axis, JointType, MultiBody};
use nalgebra::{Isometry3, Matrix3, SVector, Vector3, Vector6};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

// Simple counting global allocator to observe heap activity during forward_dynamics_ab.
struct CountingAlloc;

static ALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);
static DEALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
static DEALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        DEALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        DEALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

fn build_chain<const N: usize, const DOFS: usize>() -> MultiBody<N, DOFS> {
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

fn bench_forward_dynamics_alloc(c: &mut Criterion) {
    // Choose a moderate chain size to exercise loops.
    const N: usize = 20;
    const DOFS: usize = 26; // 6 + 20-1 revolute
    let mb = build_chain::<N, DOFS>();
    let base = Isometry3::identity();
    let joint_angles = SVector::<f64, { DOFS - 6 }>::from_vec(vec![0.1; DOFS - 6]);
    let conf = mb.minimal_to_homogeneous_configuration(&base, &joint_angles);
    let mu = SVector::<f64, DOFS>::repeat(0.01);
    let thruster = vec![Vector6::zeros(); N];
    let eta = SVector::<f64, DOFS>::zeros();
    let zero3 = Vector3::zeros();
    let rb = |_: &[Isometry3<f64>], _: &[Vector6<f64>]| nalgebra::SMatrix::<f64, 6, N>::zeros();

    // Warm up once to populate any lazy statics.
    let _ = mb.forward_dynamics_ab(&conf, &mu, rb, &thruster, &eta, &zero3, &zero3);

    c.bench_function("forward_dynamics_alloc_counts", |b| {
        b.iter(|| {
            // Reset counters per measured iteration.
            ALLOC_CALLS.store(0, Ordering::Relaxed);
            DEALLOC_CALLS.store(0, Ordering::Relaxed);
            ALLOC_BYTES.store(0, Ordering::Relaxed);
            DEALLOC_BYTES.store(0, Ordering::Relaxed);
            let _acc = mb.forward_dynamics_ab(&conf, &mu, rb, &thruster, &eta, &zero3, &zero3);
            // Fetch counts (acts as a compiler barrier by reading atomics).
            let a = ALLOC_CALLS.load(Ordering::Relaxed);
            let d = DEALLOC_CALLS.load(Ordering::Relaxed);
            let ab = ALLOC_BYTES.load(Ordering::Relaxed);
            let db = DEALLOC_BYTES.load(Ordering::Relaxed);
            // Emit via criterion's debug (println ends up in report); lightweight vs storing.
            // (Can comment out if noisy.)
            criterion::black_box((a, d, ab, db));
        });
    });

    // After the benchmark finishes, print a representative run outside timing.
    ALLOC_CALLS.store(0, Ordering::Relaxed);
    DEALLOC_CALLS.store(0, Ordering::Relaxed);
    ALLOC_BYTES.store(0, Ordering::Relaxed);
    DEALLOC_BYTES.store(0, Ordering::Relaxed);
    let _acc = mb.forward_dynamics_ab(&conf, &mu, rb, &thruster, &eta, &zero3, &zero3);
    println!(
        "Representative single-call allocations: calls={} deallocs={} bytes={} dealloc_bytes={}",
        ALLOC_CALLS.load(Ordering::Relaxed),
        DEALLOC_CALLS.load(Ordering::Relaxed),
        ALLOC_BYTES.load(Ordering::Relaxed),
        DEALLOC_BYTES.load(Ordering::Relaxed)
    );
}

criterion_group!(benches, bench_forward_dynamics_alloc);
criterion_main!(benches);
