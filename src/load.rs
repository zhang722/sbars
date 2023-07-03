use std::error::Error;

use nalgebra as na;

use crate::bundle_adjustment::*;
use crate::linear_solver::*;

type Vector8<T> = na::Matrix<T, na::U8, na::U1, na::ArrayStorage<T, 8, 1>>;
type Matrix2x8<T> = na::Matrix<T, na::U2, na::U8, na::ArrayStorage<T, 2, 8>>;

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

/// Converts a 6-Vector Lie Algebra representation of a rigid body
/// transform to an NAlgebra Isometry (quaternion+translation pair)
///
/// This is largely taken from this paper:
/// https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf
fn exp_map(param_vector: &na::Vector6<f64>) -> na::Isometry3<f64> {
    let t = param_vector.fixed_view::<3, 1>(0, 0);
    let omega = param_vector.fixed_view::<3, 1>(3, 0);
    let theta = omega.norm();
    let half_theta = 0.5 * theta;
    let quat_axis = omega * half_theta.sin() / theta;
    let quat = if theta > 1e-6 {
        na::UnitQuaternion::from_quaternion(na::Quaternion::new(
            half_theta.cos(),
            quat_axis.x,
            quat_axis.y,
            quat_axis.z,
        ))
    } else {
        na::UnitQuaternion::identity()
    };

    let mut v = na::Matrix3::<f64>::identity();
    if theta > 1e-6 {
        let ssym_omega = skew_sym(omega.clone_owned());
        v += ssym_omega * (1.0 - theta.cos()) / (theta.powi(2))
            + (ssym_omega * ssym_omega) * ((theta - theta.sin()) / (theta.powi(3)));
    }

    let trans = na::Translation::from(v * t);

    na::Isometry3::from_parts(trans, quat)
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

/// Produces the Jacobian of the exponential map of a lie algebra transform
/// that is then applied to a point with respect to the transform.
///
/// i.e.
/// d exp(t) * p
/// ------------
///      d t
///
/// The parameter 'transformed_point' is assumed to be transformed already
/// and thus: transformed_point = exp(t) * p
///
/// This is largely taken from this paper:
/// https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf
fn exp_map_jacobian(transformed_point: &na::Point3<f64>) -> na::Matrix3x6<f64> {
    let mut ss = na::Matrix3x6::zeros();
    ss.fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&na::Matrix3::<f64>::identity());
    ss.fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-skew_sym(transformed_point.coords)));
    ss
}

/// Projects a point in camera coordinates into the image plane
/// producing a floating-point pixel value
fn project(
    camera_model: &CameraModel, /*fx, fy, cx, cy, k1, k2, p1, p2*/
    pt: &na::Point3<f64>,
) -> na::Point2<f64> {
    let fx = camera_model.params[0];
    let fy = camera_model.params[1];
    let cx = camera_model.params[2];
    let cy = camera_model.params[3];
    let k1 = camera_model.params[4];
    let k2 = camera_model.params[5];
    let p1 = camera_model.params[6];
    let p2 = camera_model.params[7];

    let xn = pt.x / pt.z;
    let yn = pt.y / pt.z;
    let rn2 = xn * xn + yn * yn;
    na::Point2::<f64>::new(
        fx * (xn * (1.0 + k1 * rn2 + k2 * rn2 * rn2) + 2.0 * p1 * xn * yn + p2 * (rn2 + 2.0 * xn * xn)) + cx,
        fy * (yn * (1.0 + k1 * rn2 + k2 * rn2 * rn2) + 2.0 * p2 * xn * yn + p1 * (rn2 + 2.0 * yn * yn)) + cy
    )
}


/// Jacobian of the projection function with respect to the 3D point in camera
/// coordinates. The 'transformed_pt' is a point already in
/// or transformed to camera coordinates.
fn proj_jacobian_wrt_point(
    camera_model: &CameraModel, /*fx, fy, cx, cy, k1, k2, p1, p2*/
    transformed_pt: &na::Point3<f64>,
) -> na::Matrix2x3<f64> {
    let fx = camera_model.params[0];
    let fy = camera_model.params[1];
    let k1 = camera_model.params[4];
    let k2 = camera_model.params[5];
    let p1 = camera_model.params[6];
    let p2 = camera_model.params[7];

    let xn = transformed_pt.x / transformed_pt.z;
    let yn = transformed_pt.y / transformed_pt.z;
    let rn2 = xn * xn + yn * yn;
    let jacobian1 = na::Matrix2::<f64>::new(
        fx * (k1 * rn2 + k2 * rn2 * rn2 + 2.0 * p1 * yn + 4.0 * p2 * xn + 1.0),
        2.0 * fx * p1 * xn,
        2.0 * fy * p2 * yn,
        fy * (k1 * rn2 + k2 * rn2 * rn2 + 4.0 * p1 * yn + 2.0 * p2 * xn + 1.0),
    );
    let jacobian2 = na::Matrix2x3::<f64>::new(
        1.0 / transformed_pt.z,
        0.0,
        -transformed_pt.x / (transformed_pt.z.powi(2)),
        0.0,
        1.0 / transformed_pt.z,
        -transformed_pt.y / (transformed_pt.z.powi(2)),
    );
    jacobian1 * jacobian2
}

fn jacobian_ps_wrt_pw(transform: &na::Isometry3<f64>) -> na::Matrix3<f64> {
    *transform.rotation.to_rotation_matrix().matrix()
}

fn jacobian_r_wrt_se3(
    camera_model: &CameraModel, /*fx, fy, cx, cy, k1, k2, p1, p2*/
    transformed_pt: &na::Point3<f64>,
) -> na::Matrix2x6<f64> {
    proj_jacobian_wrt_point(camera_model, transformed_pt) * exp_map_jacobian(transformed_pt)
}

fn jacobian_r_wrt_pw(
    camera_model: &CameraModel, /*fx, fy, cx, cy, k1, k2, p1, p2*/
    transform: &na::Isometry3<f64>,
    transformed_pt: &na::Point3<f64>,
) -> na::Matrix2x3<f64> {
    proj_jacobian_wrt_point(camera_model, transformed_pt) * jacobian_ps_wrt_pw(transform)
}


#[derive(Debug)]
pub struct CameraModel {
    params: Vector8<f64>,
}

#[derive(Debug)]
pub struct TwoViewBAModel {
    initial_param: na::DVector<f64>,
    measurements: Vec<(Vec<na::Point2<f64>>, CameraModel)>,
}

impl TwoViewBAModel {
    pub fn decode_param(param: &na::DVector<f64>) -> (Vec<na::Point3<f64>>, Vec<na::Isometry3<f64>>) {
        let num_cameras = 2;
        let num_points = (param.len() - num_cameras * 6) / 3;
        let mut points3d = Vec::with_capacity(num_points);
        let mut frame = Vec::with_capacity(num_cameras);
        for i in 0..num_cameras {
            let param = param.fixed_rows::<6>(i * 6).clone_owned();
            frame.push(exp_map(&param));
        }

        for i in 0..num_points {
            points3d.push(na::Point3::<f64>::from(param.fixed_rows::<3>(num_cameras * 6 + i * 3).clone_owned()));
        }

        (points3d, frame)
    }
}

impl LMProblem for TwoViewBAModel {
    fn solve(&self, a_star: SolveCoeff, b: Param) -> Param {
        a_star.lu().solve(&b).unwrap()
        // let num_camera = self.measurements.len();
        // solve(a_star, b, num_camera).unwrap()
    }

    fn jacobian(&self, p: &Param) -> Jacobian {
        let (points3d, frame) = Self::decode_param(p);
        let num_cameras = frame.len();
        assert!(num_cameras == 2);
        let num_measurement = points3d.len() * frame.len() * 2;
        let ncols = p.len();

        let mut jacobian = Jacobian::zeros(num_measurement, ncols);
        for (point_idx, point3d) in points3d.into_iter().enumerate() {
            for (frame_idx, transform) in frame.iter().enumerate() {
                let transformed_point = transform * point3d;
                jacobian.fixed_view_mut::<2, 6>(point_idx * num_cameras * 2 + frame_idx * 2, frame_idx * 6)
                    .copy_from(&jacobian_r_wrt_se3(&self.measurements[frame_idx].1, &transformed_point));

                jacobian.fixed_view_mut::<2, 3>(point_idx * num_cameras * 2 + frame_idx * 2, num_cameras * 6 + point_idx * 3)
                    .copy_from(&jacobian_r_wrt_pw(&self.measurements[frame_idx].1, &transform, &transformed_point));
            } 
        }

        jacobian
    }

    fn residual(&self, p: &Param) -> Residual {
        let (points3d, frame) = Self::decode_param(p);
        assert!(frame.len() == 2);

        let num_measurement = points3d.len() * frame.len() * 2;
        let mut residual = na::DVector::<f64>::zeros(num_measurement);
        for (point_idx, point3d) in points3d.into_iter().enumerate() {
            for (frame_idx, transform) in frame.iter().enumerate() {
                let transformed_point = transform * point3d;
                let projected_point = project(&self.measurements[frame_idx].1, &transformed_point);
                residual[point_idx * 4 + frame_idx * 2] = projected_point.x - self.measurements[frame_idx].0[point_idx].x;
                residual[point_idx * 4 + frame_idx * 2 + 1] = projected_point.y - self.measurements[frame_idx].0[point_idx].y;
            } 
        }
        println!("residual: {}", residual.norm());

        residual
    }
}

pub fn load(path: &str) -> Result<TwoViewBAModel, Box<dyn Error>>  {
    // read path file to string
    let content = std::fs::read_to_string(path)?;

    let v: serde_json::Value = serde_json::from_str(&content)?;
    let mut points3d = Vec::new();
    let mut poses = Vec::new();
    let mut measurements = Vec::new();
    for p in v["points"].as_array().unwrap() {
        points3d.push(na::Point3::new(p[0].as_f64().unwrap(), p[1].as_f64().unwrap(), p[2].as_f64().unwrap()));
    }

    for f in v["keyframes"].as_array().unwrap() {
        let mut pt2d = Vec::new();
        let mut camera_model = Vec::new();
        for p in f["points"].as_array().unwrap() {
            pt2d.push(na::Point2::new(p[0].as_f64().unwrap(), p[1].as_f64().unwrap()));
        }
        for param in f["camera_model"].as_array().unwrap() {
            camera_model.push(param.as_f64().unwrap());
        }
        poses.push(log_map(&na::Isometry3::from_parts(
            na::Translation3::new(
                f["pose"]["x"].as_f64().unwrap(),
                f["pose"]["y"].as_f64().unwrap(),
                f["pose"]["z"].as_f64().unwrap(),
            ),
            na::UnitQuaternion::from_quaternion(na::Quaternion::new(
                    f["pose"]["q"][3].as_f64().unwrap(),
                    f["pose"]["q"][0].as_f64().unwrap(),
                    f["pose"]["q"][1].as_f64().unwrap(),
                    f["pose"]["q"][2].as_f64().unwrap(),
            )),
        ))); 
        measurements.push((pt2d, CameraModel{params: Vector8::from_row_slice(&camera_model)}));
    }

    assert!(poses.len() == 2);
    let mut initial_param = na::DVector::<f64>::zeros(poses.len() * 6 + points3d.len() * 3);
    for i in 0..poses.len() {
        initial_param.fixed_rows_mut::<6>(i * 6).copy_from(&poses[i]);
    }
    for i in 0..points3d.len() {
        initial_param.fixed_rows_mut::<3>(poses.len() * 6 + i * 3).copy_from(&points3d[i].coords);
    }

    Ok(TwoViewBAModel{initial_param, measurements})
}


#[test]
fn test_load() {
    let ba = load("scene.json").unwrap();
    println!("{:?}", ba);    
}

#[test]
fn test_ba() {
    let ba = load("scene.json").unwrap();
    let eps1 = 1e-6;
    let eps2 = 1e-6;
    let tau = 1e-1;
    let k_max = 100;
    let x0 = ba.initial_param.clone();
    let x = lm(ba, x0, eps1, eps2, tau, k_max);
    
    println!("{:?}", x);
}