# multibody_dynamics

`multibody_dynamics` is a Rust crate for rigid multibody dynamics with
spatial-vector style algorithms for kinematics, Jacobians, mass matrices,
inverse dynamics, and forward dynamics.

The crate is built around fixed-size const generics:

- `NUM_BODIES`: number of bodies/joints in the model.
- `NUM_DOFS`: total generalized coordinate dimension.

`NUM_DOFS` must equal the sum of joint dimensions. Revolute and prismatic
joints each contribute 1 DOF; `SixDOF` joints contribute 6 DOFs.

## Installation

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
multibody_dynamics = "0.4.0"
```

For local development from this repository:

```toml
[dependencies]
multibody_dynamics = { path = "." }
```

## Example

```rust
use multibody_dynamics::multibody::{
    Axis, Environment, JointType, LinkProperties, MultiBody, MultiBodyConfig, Topology,
};
use nalgebra::{Isometry3, Matrix3, SVector, Vector3};

let cfg = MultiBodyConfig::<2, 2> {
    topology: Topology {
        offset_matrices: vec![Isometry3::identity(); 2],
        joint_types: vec![
            JointType::Revolute(Axis::Z),
            JointType::Revolute(Axis::Z),
        ],
        // Parent indices are one-based. 0 means root, 1 means body 0, etc.
        parent: vec![0, 1],
    },
    link_props: Some(vec![
        LinkProperties {
            mass: Some(1.0),
            r_com: Some(Vector3::zeros()),
            inertia3: Some(Matrix3::identity()),
            ..LinkProperties::default()
        },
        LinkProperties {
            mass: Some(1.0),
            r_com: Some(Vector3::zeros()),
            inertia3: Some(Matrix3::identity()),
            ..LinkProperties::default()
        },
    ]),
    env: Environment {
        gravity: Vector3::new(0.0, 0.0, -9.81),
        rho: 1025.0,
    },
};

let model = MultiBody::<2, 2>::from_config(cfg).unwrap();
let no_six_dof_joints: Vec<Isometry3<f64>> = Vec::new();
let q = SVector::<f64, 2>::zeros();
let conf = model.minimal_to_homogeneous_configuration(&no_six_dof_joints, &q);
let mass_matrix = model.compute_mass_matrix(&conf);
```

## Topology Conventions

The `parent` vector uses one-based indexing:

- `parent[i] == 0`: body `i` is a root.
- `parent[i] == k`: body `i` has parent body `k - 1`.

Non-root parents must reference an earlier body. This keeps traversal order
well-defined and prevents cycles.

## Link Properties

You can provide either:

- a full spatial mass matrix with `mass6`, or
- scalar `mass`, `r_com`, and `inertia3`.

When `mass6` is provided, the crate derives scalar mass and center of mass from
the spatial mass matrix unless explicit `mass` or `r_com` values are also
provided. Buoyancy uses `volume` and `r_cob` when present.

## Checked APIs

The crate provides checked variants for input-sensitive operations:

- `MultiBody::from_config`
- `try_minimal_to_homogeneous_configuration`
- `try_compute_regressor_matrix`

The unchecked compatibility methods panic with clear messages when the checked
variant would return an error.

## Development

Run the main verification commands before publishing changes:

```bash
cargo fmt --check
cargo check
cargo test
cargo clippy --all-targets --all-features
cargo bench --no-run
```

Run benchmarks with:

```bash
cargo bench
```

## License

This project is licensed under GPL-3.0. See `LICENSE.txt`.
