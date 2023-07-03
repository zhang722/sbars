use nalgebra as na;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn solve(a: &na::DMatrix<f64>, b: &na::DVector<f64>, split: usize) -> Option<na::DVector<f64>> {
    assert!(b.len() >= split);
    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    println!("1.Current Micro Seconds: {}", time.as_micros());

    let ncams = split;
    let npts = b.len() - split;
    let u_star = a.view((0, 0), (ncams, ncams)); 
    let w = a.view((0, ncams), (ncams, npts)); 
    let v_star = a.view((ncams, ncams), (npts, npts)); 
    let e_cam = b.view((0, 0), (ncams, 1));
    let e_pt = b.view((ncams, 0), (npts, 1));

    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    println!("2.Current Micro Seconds: {}", time.as_micros());
    let v_star_inv = v_star_inverse(&v_star)?;
    let a_cam = u_star - w * v_star_inv.clone() * w.transpose();
    let b_cam = e_cam - w * v_star_inv.clone() * e_pt;
    let h_cam = a_cam.lu().solve(&b_cam)?;

    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    println!("3.Current Micro Seconds: {}", time.as_micros());

    let b_pt = e_pt - w.transpose() * &h_cam;
    let h_pt = v_star_inv * b_pt;

    // copy h_cam h_pt to result and return
    let mut h = na::DVector::zeros(b.len());
    h.view_mut((0, 0), (ncams, 1)).copy_from(&h_cam);
    h.view_mut((ncams, 0), (npts, 1)).copy_from(&h_pt);

    let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    println!("4.Current Micro Seconds: {}", time.as_micros());

    Some(h)
}

fn v_star_inverse<'a>(v_star: &na::DMatrixView<'a, f64>) -> Option<na::DMatrix<f64>> {
    assert!(v_star.is_square());
    assert!(v_star.nrows() % 3 == 0);
    
    let mut v_star_inv = v_star.clone_owned();
    for i in 0..v_star.nrows() / 3 {
        // println!("i*3: {}, v_star_len: {}", i * 3, v_star.len());
        let v_star_block = v_star.view((i * 3, i * 3), (3, 3));
        let v_star_block_inv = v_star_block.try_inverse()?;
        v_star_inv.fixed_view_mut::<3, 3>(i * 3, i * 3).copy_from(&v_star_block_inv);
    }

    Some(v_star_inv)
}

