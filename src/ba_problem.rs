use std::fs::File;
use std::io::{self, prelude::*, BufReader};
use std::time::{SystemTime, UNIX_EPOCH};

use nalgebra as na;
use rand::Rng;

use crate::bundle_adjustment::*;
use crate::linear_solver::*;

#[derive(Debug)]
pub struct Observation {
    point_idx: usize,
    camera_idx: usize,
    x: f64,
    y: f64,
    x_std: f64,
    y_std: f64,
}

pub struct BaProblem {
    num_cameras: usize,
    observations: Vec<Observation>,
    parameters: na::DVector<f64>,
}

impl BaProblem {
    pub fn new(filename: &str, max_num_cameras: usize, max_num_points: usize) -> Self {
        // parse file 
        let file = File::open(filename).expect("open file failed!");
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        let (num_cameras, num_points, num_observations) = match lines.next() {
            Some(line) => {
                let line = line.expect("line read failed!");
                let mut iter = line.split_whitespace();
                let num_cameras = iter.next().unwrap().parse::<usize>().unwrap();
                let num_points = iter.next().unwrap().parse::<usize>().unwrap();
                let num_observations = iter.next().unwrap().parse::<usize>().unwrap();

                (num_cameras, num_points, num_observations)
            },
            None => panic!("file is empty!"),
        };

        let expected_num_cameras = if num_cameras > max_num_cameras {
            max_num_cameras   
        } else {
            num_cameras
        };
        let expected_num_points = if num_points > max_num_points {
            max_num_points   
        } else {
            num_points
        };

        let mut observations = Vec::with_capacity(num_observations);
        let mut parameters_cameras = Vec::with_capacity(9 * expected_num_cameras);
        let mut parameters_points = Vec::with_capacity(3 * expected_num_points);

        let parse_observation = |line: &str| -> Option<Observation> {
            let mut split = line.split_whitespace();
            let camera_idx = split.next().unwrap().parse::<usize>().unwrap();
            let point_idx = split.next().unwrap().parse::<usize>().unwrap();
            if camera_idx >= expected_num_cameras || point_idx >= expected_num_points {
                return None;
            }

            let x = split.next().unwrap().parse::<f64>().unwrap();
            let y = split.next().unwrap().parse::<f64>().unwrap();
            Some(Observation {
                point_idx,
                camera_idx,
                x,
                y,
                x_std: 0.0,
                y_std: 0.0,
            })           
        };

        for (line_idx, line) in lines.enumerate() {
            let line = line.expect("line read failed!");
            if line_idx < num_observations {
                let observation = match parse_observation(&line) {
                    Some(observation) => observation,
                    None => continue,
                };
                observations.push(observation);
            } else if line_idx < num_observations + 9 * num_cameras {
                let e = line.parse::<f64>().unwrap();
                if parameters_cameras.len() < parameters_cameras.capacity() {
                    parameters_cameras.push(e);
                }
            } else {
                let e = line.parse::<f64>().unwrap();
                if parameters_points.len() < parameters_points.capacity() {
                    parameters_points.push(e);
                }
            }
        }

        parameters_cameras.append(&mut parameters_points);

        Self {
            num_cameras: expected_num_cameras,
            observations,
            parameters: na::DVector::from_vec(parameters_cameras),
        }
    }

    fn get_camera_index(&self, camera_idx: usize) -> usize {
        9 * camera_idx
    }

    fn get_point_index(&self, point_idx: usize) -> usize {
        9 * self.num_cameras + 3 * point_idx
    }

    fn generate() -> Self {
        let num_cameras = 1;
        let num_points = 1;
        let num_observations = 1;

        let points3d = vec![
            na::Vector3::<f64>::new(0.0, 1.0, 0.5),
        ];

        let cameras = vec![
            na::Isometry3::<f64>::from_parts(
                na::Translation3::<f64>::new(0.0, 0.0, -0.4),
                na::UnitQuaternion::<f64>::from_euler_angles(0.0, 0.0, -10.0),
            ),
        ];

        let mut observations = Vec::with_capacity(num_observations);
        let mut parameters = Vec::with_capacity(9 * num_cameras + 3 * num_points);

        let mut rng = rand::thread_rng();
        let mut rng2 = rand::thread_rng();

        for i in 0..num_cameras {
            let lie_algebra = log_map(&cameras[i]);
            parameters.push(lie_algebra[3]);
            parameters.push(lie_algebra[4]);
            parameters.push(lie_algebra[5]);
            parameters.push(lie_algebra[0]);
            parameters.push(lie_algebra[1]);
            parameters.push(lie_algebra[2]);
            parameters.push(10.0);
            parameters.push(0.0);
            parameters.push(0.0);
        }

        for (i, point3d) in points3d.iter().enumerate() {
            parameters.push(point3d[0]);
            parameters.push(point3d[1]);
            parameters.push(point3d[2]);
            
            for j in 0..num_cameras {
                let camera_idx = j;
                let point_idx = i;
                let m = cameras[j] * point3d;
                let m = m / m[2];
                let x = rng.gen_range(-1.0..1.0);
                let y = rng.gen_range(-1.0..1.0);
                observations.push(Observation {
                    camera_idx,
                    point_idx,
                    x: m[0] * 10.0 + x,
                    y: m[1] * 10.0 + y,
                    x_std: m[0] * 10.0,
                    y_std: m[1] * 10.0,
                });
            }
        }
        Self {
            num_cameras,
            observations,
            parameters: na::DVector::from_vec(parameters),
        }
    }
}

type CameraInstrinsics = na::Vector3<f64>;

fn jacobian_pp_wrt_pn(
    pn: &na::Vector2<f64>, 
    intrinsics: &CameraInstrinsics,
) -> na::Matrix2<f64> 
{
    let f = intrinsics[0];
    let k1 = intrinsics[1];
    let k2 = intrinsics[2];
    let x = pn[0];
    let y = pn[1];

    let rn2 = x.powi(2) + y.powi(2);
    let rn4 = rn2.powi(2);

    na::Matrix2::<f64>::new(
        f * (k1 * rn2 + k2 * rn4 + 1.0), 0.0,
        0.0, f * (k1 * rn2 + k2 * rn4 + 1.0)
    )
}

fn jacobian_pn_wrt_ps(
    ps: &na::Vector3<f64>,
) -> na::Matrix2x3<f64>
{
    let x = ps[0];
    let y = ps[1];
    let z = ps[2];
    let z2 = z.powi(2);

    -na::Matrix2x3::<f64>::new(
        1.0 / z, 0.0, -x / z2, 
        0.0, 1.0 / z, -y / z2)
}

/// Produces a skew-symmetric or "cross-product matrix" from
/// a 3-vector. This is needed for the `exp_map` and `log_map`
/// functions
fn skew_sym(v: na::Vector3<f64>) -> na::Matrix3<f64> {
    let mut ss = na::Matrix3::zeros();
    ss[(0, 1)] = -v[2];
    ss[(0, 2)] = v[1];
    ss[(1, 0)] = v[2];
    ss[(1, 2)] = -v[0];
    ss[(2, 0)] = -v[1];
    ss[(2, 1)] = v[0];
    ss
}

/// Converts an NAlgebra Isometry to a 6-Vector Lie Algebra representation
/// of a rigid body transform.
///
/// This is largely taken from this paper:
/// https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf
fn log_map(input: &na::Isometry3<f64>) -> na::Vector6<f64> {
    let t: na::Vector3<f64> = input.translation.vector;

    let quat = input.rotation;
    let theta: f64 = 2.0 * (quat.scalar()).acos();
    let half_theta = 0.5 * theta;
    let mut omega = na::Vector3::<f64>::zeros();

    let mut v_inv = na::Matrix3::<f64>::identity();
    if theta > 1e-6 {
        omega = quat.vector() * theta / (half_theta.sin());
        let ssym_omega = skew_sym(omega);
        v_inv -= ssym_omega * 0.5;
        v_inv += ssym_omega * ssym_omega * (1.0 - half_theta * half_theta.cos() / half_theta.sin())
            / (theta * theta);
    }

    let mut ret = na::Vector6::<f64>::zeros();
    ret.fixed_view_mut::<3, 1>(0, 0).copy_from(&(v_inv * t));
    ret.fixed_view_mut::<3, 1>(3, 0).copy_from(&omega);

    ret
}

fn jacobian_ps_wrt_Tsw(
    ps: &na::Vector3<f64>,
) -> na::Matrix3x6<f64>
{
    let x = ps[0];
    let y = ps[1];
    let z = ps[2];

    let mut jac = na::Matrix3x6::<f64>::zeros();
    jac.fixed_view_mut::<3, 3>(0, 3).copy_from(&na::Matrix3::<f64>::identity());
    jac.fixed_view_mut::<3, 3>(0, 0).copy_from(&-skew_sym(na::Vector3::<f64>::new(x, y, z)));
    jac
}

fn jacobian_ps_wrt_pw(
    Rsw: &na::Rotation3<f64>,
) -> na::Matrix3<f64>
{
    let mut jac = na::Matrix3::<f64>::zeros();
    jac.copy_from(Rsw.matrix());
    jac
}

fn jacobian_pp_wrt_intrinsics(
    pn: &na::Vector2<f64>,
    intrinsics: &CameraInstrinsics,
) -> na::Matrix2x3<f64>
{
    let f = intrinsics[0];
    let k1 = intrinsics[1];
    let k2 = intrinsics[2];
    let x = pn[0];
    let y = pn[1];

    let rn2 = x.powi(2) + y.powi(2);
    let rn4 = rn2.powi(2);

    na::Matrix2x3::<f64>::new(
        x * (k1 * rn2 + k2 * rn4 + 1.0), f * rn2 * x, f * rn4 * x,
        y * (k1 * rn2 + k2 * rn4 + 1.0), f * rn2 * y, f * rn4 * y
    )
}

impl super::lm::LMProblem for BaProblem {
    fn solve(&self, a_star: &super::lm::SolveCoeff, b: &crate::bundle_adjustment::Param) -> Param {
        // a_star.clone().lu().solve(b).unwrap()
        solve(a_star, b, self.num_cameras * 9).unwrap()
    }

    fn jacobian(&self, p: &crate::bundle_adjustment::Param) -> Jacobian {
        let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        println!("Jac1.Current Micro Seconds: {}", time.as_micros());
        let mut jac = Jacobian::zeros(self.observations.len() * 2, self.parameters.len());

        for (observation_idx, observation) in self.observations.iter().enumerate() {
            let camera_idx = self.get_camera_index(observation.camera_idx);
            let camera = p.fixed_rows::<9>(camera_idx); 
            let point_idx = self.get_point_index(observation.point_idx);
            let point = p.fixed_rows::<3>(point_idx);

            // if camera_idx == 0 {
            //     continue;
            // }

            let p = na::Matrix3x1::new(point[0], point[1], point[2]);
            let rotation = na::Rotation3::new(
                na::Vector3::<f64>::new(camera[0], camera[1], camera[2])
            );
            let t = na::Vector3::<f64>::new(camera[3], camera[4], camera[5]);
            let intrinsics = na::Vector3::<f64>::new(camera[6], camera[7], camera[8]);

            let ps = rotation * p + t;
            let pn = -ps / ps.z;
            let pn = pn.fixed_view::<2, 1>(0, 0).clone_owned();

            let jacobian_r_wrt_Tsw = jacobian_pp_wrt_pn(&pn, &intrinsics) 
                * jacobian_pn_wrt_ps(&ps) 
                * jacobian_ps_wrt_Tsw(&ps);
            let jacobian_r_wrt_intrinsics = jacobian_pp_wrt_intrinsics(&pn, &intrinsics);
            let jacobian_r_wrt_pw = jacobian_pp_wrt_pn(&pn, &intrinsics) 
                * jacobian_pn_wrt_ps(&ps) 
                * jacobian_ps_wrt_pw(&rotation);
            
            jac.fixed_view_mut::<2, 6>(observation_idx * 2, camera_idx)
                .copy_from(&jacobian_r_wrt_Tsw);
            jac.fixed_view_mut::<2, 3>(observation_idx * 2, camera_idx + 6)
                .copy_from(&jacobian_r_wrt_intrinsics);
            jac.fixed_view_mut::<2, 3>(observation_idx * 2, point_idx)
                .copy_from(&jacobian_r_wrt_pw);
        }
        let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        println!("Jac2.Current Micro Seconds: {}", time.as_micros());
        jac
    }

    fn residual(&self, p: &crate::bundle_adjustment::Param) -> Residual {
        let mut res = Residual::zeros(self.observations.len() * 2);

        for (observation_idx, observation) in self.observations.iter().enumerate() {
            let camera = self.get_camera_index(observation.camera_idx);
            let camera = p.fixed_rows::<9>(camera); 
            let point = self.get_point_index(observation.point_idx);
            let point = p.fixed_rows::<3>(point);

            let p = na::Matrix3x1::new(point[0], point[1], point[2]);
            let rotation = na::Rotation3::new(
                na::Vector3::<f64>::new(camera[0], camera[1], camera[2])
            );
            let t = na::Vector3::<f64>::new(camera[3], camera[4], camera[5]);
            let f = camera[6];
            let k1 = camera[7];
            let k2 = camera[8];

            let ps = rotation * p + t;
            let pn = -ps / ps.z;
            let pn = pn.fixed_view::<2, 1>(0, 0);
            let pp = f * (1.0 + k1 * pn.norm().powi(2) + k2 * pn.norm().powi(4)) * pn;

            res.fixed_view_mut::<2, 1>(observation_idx * 2, 0)
                .copy_from(&(pp - na::Vector2::new(observation.x, observation.y)));
        }

        res
    }
}

use std::fs::OpenOptions;
use std::io::prelude::*;
use ply_rs::ply::{ Ply, DefaultElement, Encoding, ElementDef, PropertyDef, PropertyType, ScalarType, Property, Addable };
use ply_rs::writer::{ Writer };
fn savePly(filename: &str, vertices: &[f64]) {
    // crete a ply objet
    let mut ply = {
        let mut ply = Ply::<DefaultElement>::new();
        ply.header.encoding = Encoding::Ascii;
        ply.header.comments.push("A beautiful comment!".to_string());

        // Define the elements we want to write. In our case we write a 2D Point.
        // When writing, the `count` will be set automatically to the correct value by calling `make_consistent`
        let mut point_element = ElementDef::new("point".to_string());
        let p = PropertyDef::new("x".to_string(), PropertyType::Scalar(ScalarType::Float));
        point_element.properties.add(p);
        let p = PropertyDef::new("y".to_string(), PropertyType::Scalar(ScalarType::Float));
        point_element.properties.add(p);
        let p = PropertyDef::new("z".to_string(), PropertyType::Scalar(ScalarType::Float));
        point_element.properties.add(p);
        ply.header.elements.add(point_element);

        // Add data
        let mut points = Vec::new();

        for v in vertices.chunks(3) {
            // Add first point
            let mut point = DefaultElement::new();
            point.insert("x".to_string(), Property::Float(v[0] as f32));
            point.insert("y".to_string(), Property::Float(v[1] as f32));
            point.insert("z".to_string(), Property::Float(v[2] as f32));
            points.push(point);
        }

        ply.payload.insert("point".to_string(), points);

        // only `write_ply` calls this by itself, for all other methods the client is
        // responsible to make the data structure consistent.
        // We do it here for demonstration purpose.
        ply.make_consistent().unwrap();
        ply
    };

    let mut file_result = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(filename).unwrap();

    // set up a writer
    let w = Writer::new();
    let written = w.write_ply(&mut file_result, &mut ply).unwrap();
}    

#[test]
fn test_ba_problem() {
    let num_camera = 5;
    let num_point = 2000;
    let problem = BaProblem::new("problem-16-22106-pre.txt", num_camera, num_point);
    // let problem = BaProblem::generate();
    println!("num_observations: {}", problem.observations.len());
    savePly("output.ply", &problem.parameters.as_slice()[9 * num_camera ..]);
    let x0 = problem.parameters.clone();
    let prime = problem.parameters.clone();

    let solver = super::lm::LM::default(Box::new(problem)).with_max_iter(100);

    let x = solver.optimize(&x0);
    // for i in 0..x.len() {
    //     println!("x[{}]: {}, prime[{}]: {}", i, x[i], i, prime[i]);
    // }
    savePly("output_optimized.ply", &x.as_slice()[9 * num_camera ..]);
}