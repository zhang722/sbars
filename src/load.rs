use std::error::Error;

use nalgebra as na;

#[derive(Debug)]
pub struct TwoViewBAModel {
    points3d: Vec<na::Point3<f64>>,
    frame: Vec<(na::Isometry3<f64>, Vec<na::Point2<f64>>)>,
}

pub fn load(path: &str) -> Result<TwoViewBAModel, Box<dyn Error>>  {
    // read path file to string
    let content = std::fs::read_to_string(path)?;

    let v: serde_json::Value = serde_json::from_str(&content)?;
    let mut ba = TwoViewBAModel {
        points3d: Vec::new(),
        frame: Vec::new(),
    };
    for p in v["points"].as_array().unwrap() {
        ba.points3d.push(na::Point3::new(p[0].as_f64().unwrap(), p[1].as_f64().unwrap(), p[2].as_f64().unwrap()));
    }

    for f in v["keyframes"].as_array().unwrap() {
        let mut pt2d = Vec::new();
        for p in f["points"].as_array().unwrap() {
            pt2d.push(na::Point2::new(p[0].as_f64().unwrap(), p[1].as_f64().unwrap()));
        }
        ba.frame.push((na::Isometry3::from_parts(
            na::Translation3::new(
                f["pose"]["x"].as_f64().unwrap(),
                f["pose"]["y"].as_f64().unwrap(),
                f["pose"]["z"].as_f64().unwrap(),
            ),
            na::UnitQuaternion::from_quaternion(na::Quaternion::new(
                    f["pose"]["q"][0].as_f64().unwrap(),
                    f["pose"]["q"][1].as_f64().unwrap(),
                    f["pose"]["q"][2].as_f64().unwrap(),
                    f["pose"]["q"][3].as_f64().unwrap(),
            )),
        ), pt2d));
    }

    Ok(ba)
}


#[test]
fn test_load() {
    let ba = load("scene.json").unwrap();
    println!("{:?}", ba);    
}