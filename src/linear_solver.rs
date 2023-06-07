use nalgebra as na;

pub fn solve(a: na::DMatrix<f64>, b: na::DVector<f64>, split: usize) -> Option<na::DVector<f64>> {
    assert!(b.len() >= split);

    let ncams = split;
    let npts = b.len() - split;
    let u_star = a.view((0, 0), (ncams, ncams)); 
    let w = a.view((0, ncams), (ncams, npts)); 
    let v_star = a.view((ncams, ncams), (npts, npts)); 
    let e_cam = b.view((0, 0), (ncams, 1));
    let e_pt = b.view((ncams, 0), (npts, 1));

    let a_cam = u_star - w * v_star.try_inverse()? * w.transpose();
    let b_cam = e_cam - w * v_star.try_inverse()? * e_pt;
    let h_cam = a_cam.lu().solve(&b_cam)?;

    let b_pt = e_pt - w.transpose() * &h_cam;
    let h_pt = v_star.lu().solve(&b_pt)?;

    // copy h_cam h_pt to result and return
    let mut h = na::DVector::zeros(b.len());
    h.view_mut((0, 0), (ncams, 1)).copy_from(&h_cam);
    h.view_mut((ncams, 0), (npts, 1)).copy_from(&h_pt);

    Some(h)
}

