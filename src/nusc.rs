use std::fs::File;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use ahash::{HashMap, HashMapExt};
use enum_map::EnumMap;
use pyo3::exceptions::{PyAssertionError, PyFileNotFoundError, PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::Deserialize;

use crate::common::SensorChannel;
use crate::domain::*;
use crate::model::*;
use crate::proxy::*;
use crate::table::{AsRefToken, Table};

macro_rules! load_table {
    ($scope:ident, $root:expr, $filename:expr, $target:ident, $error_cell:ident) => {
        $scope.spawn(|_| {
            if $error_cell.lock().unwrap().is_ok() {
                match load_json(&$root, $filename) {
                    Ok(data) => $target = Some(data),
                    Err(e) => {
                        let mut guard = $error_cell.lock().unwrap();
                        if guard.is_ok() {
                            *guard = Err(e);
                        }
                    }
                }
            }
        });
    };
}

fn load_json<T>(path: &Path, name: &str) -> PyResult<T>
where
    T: Deserialize<'static>,
{
    let path = path.join(name);
    if !path.exists() {
        return Err(PyValueError::new_err(format!("{name} not found at {}", path.display())));
    }
    let file = File::open(path).map_err(|e| PyValueError::new_err(format!("Failed to open {name}: {}", e)))?;
    let mmap = unsafe { memmap2::Mmap::map(&file) }
        .map_err(|e| PyValueError::new_err(format!("Failed to map {name}: {}", e)))?;
    let mmap = Box::leak(Box::new(mmap));
    let json =
        std::str::from_utf8(mmap).map_err(|e| PyValueError::new_err(format!("Invalid UTF-8 token found: {e}")))?;
    serde_json::from_str(&json).map_err(|e| PyValueError::new_err(format!("Failed to parse {name}: {}", e)))
}

#[pyclass]
/// NuScenes dataset implemented in Rust.
pub struct NuScenes {
    // Vehicle
    logs:        Table<Log<'static>>,
    maps:        Table<Map<'static>>,
    sensors:     Table<Sensor>,
    cal_sensors: Table<CalibratedSensor>,
    // Extraction
    scenes:      Table<Scene<'static>>,
    samples:     Table<Sample>,
    sample_data: Table<SampleData<'static>>,
    ego_poses:   Table<EgoPose>,
    // Annotation
    instances:   Table<Instance>,
    sample_anns: Table<SampleAnnotation<'static>>,
    // Taxonomy
    categories:  Table<Category<'static>>,
    attributes:  Table<Attribute<'static>>,
}

#[pymethods]
impl NuScenes {
    #[new]
    fn new(version: &str, dataroot: &str) -> PyResult<Self> {
        let table_root = Path::new(dataroot).join(version);
        if !table_root.exists() {
            return Err(PyFileNotFoundError::new_err(format!(
                "nuScenes dataset not found at {}",
                table_root.display()
            )));
        }

        let error_capture = Mutex::new(Ok(()));

        let mut logs: Option<Vec<LogModel>> = None;
        let mut maps: Option<Vec<MapModel>> = None;
        let mut sensors: Option<Vec<SensorModel>> = None;
        let mut cal_sensors: Option<Vec<CalibratedSensorModel>> = None;
        let mut scenes: Option<Vec<SceneModel>> = None;
        let mut samples: Option<Vec<SampleModel>> = None;
        let mut sample_data: Option<Vec<SampleDataModel>> = None;
        let mut ego_poses: Option<Vec<EgoPoseModel>> = None;
        let mut instances: Option<Vec<InstanceModel>> = None;
        let mut sample_anns: Option<Vec<SampleAnnotationModel>> = None;
        let mut categories: Option<Vec<CategoryModel>> = None;
        let mut attributes: Option<Vec<AttributeModel>> = None;

        rayon::scope(|s| {
            load_table!(s, table_root, "log.json", logs, error_capture);
            load_table!(s, table_root, "map.json", maps, error_capture);
            load_table!(s, table_root, "sensor.json", sensors, error_capture);
            load_table!(s, table_root, "calibrated_sensor.json", cal_sensors, error_capture);
            load_table!(s, table_root, "scene.json", scenes, error_capture);
            load_table!(s, table_root, "sample.json", samples, error_capture);
            load_table!(s, table_root, "sample_data.json", sample_data, error_capture);
            load_table!(s, table_root, "ego_pose.json", ego_poses, error_capture);
            load_table!(s, table_root, "instance.json", instances, error_capture);
            load_table!(s, table_root, "sample_annotation.json", sample_anns, error_capture);
            load_table!(s, table_root, "category.json", categories, error_capture);
            load_table!(s, table_root, "attribute.json", attributes, error_capture);
        });

        error_capture.into_inner().unwrap()?;

        // into_iter() is used instead of into_par_iter() for tables that are relatively small.
        // For example, only 23 categories exists, and even lesser attributes.
        // For tables that doesn't need a reverse indexing, it is directly converted from the
        // models to the domain, then to Table. As for those who need, it is only unwrapped.

        // The following count is taken from nuscenes-trainval subset
        let logs = logs.unwrap(); // 68
        let maps = Table::new(maps.unwrap().into_iter().map(Map::from).collect()); // 4
        let sensors = Table::new(sensors.unwrap().into_iter().map(Sensor::from).collect()); // 12
        let cal_sensors = Table::new(cal_sensors.unwrap().into_par_iter().map(CalibratedSensor::from).collect()); // 10200

        let scenes = Table::new(scenes.unwrap().into_par_iter().map(Scene::from).collect()); // 850
        let samples = samples.unwrap(); // 34,149
        let sample_data = sample_data.unwrap(); // 2,631,083
        let ego_poses = Table::new(ego_poses.unwrap().into_par_iter().map(EgoPose::from).collect()); // 2,631,083

        let instances = Table::new(instances.unwrap().into_par_iter().map(Instance::from).collect()); // 64,386
        let sample_anns = sample_anns.unwrap(); // 1,166,187

        let categories = Table::new(categories.unwrap().into_iter().map(Category::from).collect()); // 23
        let attributes = Table::new(attributes.unwrap().into_iter().map(Attribute::from).collect()); // 8

        let logs = {
            let log_to_map: HashMap<[u8; 16], [u8; 16]> =
                maps.iter().flat_map(|map| map.log_tokens.iter().map(move |tok| (*tok, map.token))).collect();
            let logs = logs
                .into_iter()
                .map(|log| Log::from_model(log_to_map.get(&log.token).copied().unwrap(), log))
                .collect();
            Table::new(logs)
        };

        let sample_data = {
            let sample_data = sample_data
                .into_par_iter()
                .map(|sd| {
                    let cal = cal_sensors.get(&sd.calibrated_sensor_token).unwrap();
                    let sen = sensors.get(&cal.sensor_token).unwrap();
                    SampleData::from_model(sen.modality, sen.channel, sd)
                })
                .collect();
            Table::new(sample_data)
        };

        let sample_anns = {
            let sample_anns = sample_anns
                .into_par_iter()
                .map(|ann| {
                    let ins = instances.get(&ann.instance_token).unwrap();
                    let cat = categories.get(&ins.category_token).unwrap();
                    SampleAnnotation::from_model(cat.name.clone(), ann)
                })
                .collect();
            Table::new(sample_anns)
        };

        let samples = {
            let sample_to_sd: HashMap<[u8; 16], EnumMap<SensorChannel, [u8; 16]>> =
                sample_data.iter().fold(HashMap::with_capacity(samples.len()), |mut acc, sd| {
                    let map = acc.entry(sd.sample_token).or_default();
                    map[sd.channel] = sd.token;
                    acc
                });
            let sample_to_ann: HashMap<[u8; 16], Vec<[u8; 16]>> =
                sample_anns.iter().fold(HashMap::with_capacity(samples.len()), |mut acc, ann| {
                    acc.entry(ann.sample_token).or_default().push(ann.token);
                    acc
                });
            let samples = samples
                .into_par_iter()
                .map(|sample| {
                    let data = sample_to_sd.get(&sample.token).cloned().unwrap_or_default();
                    let anns = sample_to_ann.get(&sample.token).cloned().unwrap_or_default().into_boxed_slice();
                    Sample::from_model(data, anns, sample)
                })
                .collect();
            Table::new(samples)
        };

        Ok(Self {
            logs,
            maps,
            sensors,
            cal_sensors,
            scenes,
            samples,
            sample_data,
            ego_poses,
            instances,
            sample_anns,
            categories,
            attributes,
        })
    }

    #[getter]
    fn log(&self) -> LogView {
        LogView { data: self.logs.data.clone() }
    }

    #[getter]
    fn map(&self) -> MapView {
        MapView { data: self.maps.data.clone() }
    }

    #[getter]
    fn sensor(&self) -> SensorView {
        SensorView { data: self.sensors.data.clone() }
    }

    #[getter]
    fn calibrated_sensor(&self) -> CalibratedSensorView {
        CalibratedSensorView { data: self.cal_sensors.data.clone() }
    }

    #[getter]
    fn scene(&self) -> SceneView {
        SceneView { data: self.scenes.data.clone() }
    }

    #[getter]
    fn sample(&self) -> SampleView {
        SampleView { data: self.samples.data.clone() }
    }

    #[getter]
    fn sample_data(&self) -> SampleDataView {
        SampleDataView { data: self.sample_data.data.clone() }
    }

    #[getter]
    fn ego_pose(&self) -> EgoPoseView {
        EgoPoseView { data: self.ego_poses.data.clone() }
    }

    #[getter]
    fn instance(&self) -> InstanceView {
        InstanceView { data: self.instances.data.clone() }
    }

    #[getter]
    fn sample_annotation(&self) -> SampleAnnotationView {
        SampleAnnotationView { data: self.sample_anns.data.clone() }
    }

    #[getter]
    fn category(&self) -> CategoryView {
        CategoryView { data: self.categories.data.clone() }
    }

    #[getter]
    fn attribute(&self) -> AttributeView {
        AttributeView { data: self.attributes.data.clone() }
    }

    fn get<'py>(slf: PyRef<'py, Self>, table: &str, token: &str) -> PyResult<Bound<'py, PyDict>> {
        let mut bytes = [0u8; 16];
        hex::decode_to_slice(token, &mut bytes)
            .map_err(|_| PyValueError::new_err(format!("Invalid token format: {token}")))?;

        match table {
            "log" => slf.lookup_in_table(slf.py(), &slf.logs, &bytes),
            "map" => slf.lookup_in_table(slf.py(), &slf.maps, &bytes),
            "sensor" => slf.lookup_in_table(slf.py(), &slf.sensors, &bytes),
            "calibrated_sensor" => slf.lookup_in_table(slf.py(), &slf.cal_sensors, &bytes),
            "scene" => slf.lookup_in_table(slf.py(), &slf.scenes, &bytes),
            "sample" => slf.lookup_in_table(slf.py(), &slf.samples, &bytes),
            "sample_data" => slf.lookup_in_table(slf.py(), &slf.sample_data, &bytes),
            "ego_pose" => slf.lookup_in_table(slf.py(), &slf.ego_poses, &bytes),
            "instance" => slf.lookup_in_table(slf.py(), &slf.instances, &bytes),
            "sample_annotation" => slf.lookup_in_table(slf.py(), &slf.sample_anns, &bytes),
            "category" => slf.lookup_in_table(slf.py(), &slf.categories, &bytes),
            "attribute" => slf.lookup_in_table(slf.py(), &slf.attributes, &bytes),
            _ => Err(PyAssertionError::new_err(format!("Table '{table}' not found"))),
        }
    }
}

impl NuScenes {
    fn lookup_in_table<'py, T: ToPyDict + AsRefToken>(
        &self, py: Python<'py>, table: &Table<T>, token: &[u8; 16],
    ) -> PyResult<Bound<'py, PyDict>> {
        table.get(token).map(|d| d.to_py_dict(py)).ok_or_else(|| PyKeyError::new_err(hex::encode(token)))?
    }
}
