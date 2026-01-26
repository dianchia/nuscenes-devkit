use pyo3::prelude::*;

mod common;
mod domain;
mod model;
mod nusc;
mod proxy;
mod table;

/// A Python module implemented in Rust.
#[pymodule]
mod _nuscenes_rs {
    use pyo3::prelude::*;

    #[pymodule_export]
    use super::nusc::NuScenes;
}
