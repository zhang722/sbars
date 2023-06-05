use nalgebra as na;

pub fn solve(a: na::DMatrix<f64>) -> na::DVector<f64> {
    a.row(0).transpose()
}