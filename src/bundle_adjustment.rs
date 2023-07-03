use nalgebra as na;

pub type Param = na::DVector<f64>;
pub type SolveCoeff = na::DMatrix<f64>;
pub type Residual = na::DVector<f64>;
pub type Jacobian = na::DMatrix<f64>;
// R: measurement dimension
// C: param dimension
// so that: J: RxC, residual: 1xR, param: 1xC
pub trait LMProblem {
    fn solve(&self, a_star: SolveCoeff, b: Param) -> Param;
    fn jacobian(&self, p: &Param) -> Jacobian;
    fn residual(&self, p: &Param) -> Residual;
}

    // eps1: stop condition for gradient, suggested: 1e-6
    // eps2: stop condition for step, suggested: 1e-6
    // tau: initial damping factor, suggested: 1e-1
pub fn lm<P: LMProblem>(p: P, x0: Param, eps1: f64, eps2: f64, tau: f64, k_max: usize) -> Param {
        let mut k: usize = 0;
        let mut v = 2.0;
        let mut x = x0;
        let j = p.jacobian(&x);
        let mut a = j.transpose() * &j;
        let mut e = p.residual(&x);
        let mut g = j.transpose() * &e;
        let mut stop = g.norm() <= eps1;
        println!("stop: {}", stop);
        let ncol = x.len();
        let mut mu: f64 = tau * {
            let mut max = 0.0;
            for i in 0..ncol {
                if a[(i, i)] > max {
                    max = a[(i, i)];
                }
            }
            max
        };

        while !stop && k < k_max {
            println!("mu: {}, k: {}", mu, k);
            k += 1;
            let mut a_star = a.clone();
            for i in 0..ncol {
                a_star[(i, i)] += mu;
            }
            let h = p.solve(a_star, -(g.clone()));
            if h.norm() <= eps2 * (x.norm() + eps2) {
                println!("end: h: {}", h.norm());
                stop = true;
            } else {
                let x_new = &h + &x;
                let e_new = p.residual(&x_new);
                let rho = (e.norm() - e_new.norm()) / (0.5 * h.transpose() * (mu * h - &g)).norm();
                println!("rho: {}", rho);
                if rho > 0.0 {
                    x = x_new;
                    a = p.jacobian(&x).transpose() * p.jacobian(&x);
                    e = p.residual(&x);
                    g = p.jacobian(&x).transpose() * &e;
                    stop = g.norm() <= eps1;
                    mu *= 0.3333333333_f64.max(1.0 - (2.0 * rho - 1.0).powi(3));
                    v = 2.0;
                } else {
                    mu *= v;
                    v *= 2.0;
                }
            }
            
        }
        x
}


mod test {
    use super::*;

    // y = ax^2 + bx + c
    // a = b = c = 1
    struct TestProblem {
        measurements: Vec<na::Vector2<f64>>,
    }

    impl LMProblem for TestProblem {
        fn solve(&self, a_star: SolveCoeff, b: Param) -> Param {
            a_star.lu().solve(&b).unwrap()
        }

        fn jacobian(&self, p: &Param) -> Jacobian {
            let n = self.measurements.len();
            let m = p.len();
            let mut jac = Jacobian::zeros(n, m);
            for i in 0..n {
                jac.fixed_view_mut::<1, 3>(i, 0)
                    .copy_from(
                        &(na::Matrix1x3::<f64>::new(
                            self.measurements[i][0].powi(2),
                            self.measurements[i][0],
                            1.0,
                        )),
                    )
            }
            jac
        }

        fn residual(&self, p: &Param) -> Residual {
            let n = self.measurements.len();
            let mut res = Residual::zeros(n);
            for i in 0..n {
                let x = self.measurements[i][0];
                let y = self.measurements[i][1];
                let y_hat = p[0] * x.powi(2) + p[1] * x + p[2];
                res[i] = y_hat - y;
            }
            res
        }
    }

    #[test]
    fn test_lm() {
        let measurements = vec![
            na::Vector2::<f64>::new(0.0, 1.0),
            na::Vector2::<f64>::new(1.0, 3.1),
            na::Vector2::<f64>::new(2.0, 6.9),
            na::Vector2::<f64>::new(3.0, 13.1),
        ];
        let problem = TestProblem { measurements };
        let x0 = na::DVector::<f64>::from_vec(vec![0.0, 0.0, 0.0]);
        let eps1 = 1e-6;
        let eps2 = 1e-6;
        let tau = 1e-1;
        let k_max = 100;
        let x = lm(problem, x0, eps1, eps2, tau, k_max);
        println!("x = {}", x);
    }

}
    