use nalgebra as na;


type Param<const M: usize> = na::SVector<f64, M>;
type Residual<const N: usize> = na::SVector<f64, N>;
type Jacobian<const N: usize, const M: usize> = na::SMatrix<f64, N, M>;

// R: measurement dimension
// C: param dimension
// so that: J: RxC, residual: 1xR, param: 1xC
pub trait LMProblem
    <T, const R: usize, const C: usize, 
    // A=na::SMatrix<f64, C, C>,
    // B=na::SVector<f64, C>,
    // Param = na::SVector<f64, C>,
    // Jacobian = na::SMatrix<f64, R, C>,
    // Residual = na::SVector<f64, R>
    > 
{
    fn solve(&self, a_star: &na::SMatrix<f64, C, C>, b: &na::SVector<f64, C>) -> Param<C>;
    fn jacobian(&self, p: &Param<C>) -> Jacobian<R, C>;
    fn residual(&self, p: &Param<C>) -> Residual<R>;
    // eps1: stop condition for gradient, suggested: 1e-6
    // eps2: stop condition for step, suggested: 1e-6
    // tau: initial damping factor, suggested: 1e-1
    fn run(&self, x0: Param<C>, eps1: f64, eps2: f64, tau: f64, k_max: usize) -> Param<C> {
        let mut k: usize = 0;
        let mut v = 2.0;
        let mut x = x0;
        let j = self.jacobian(&x);
        let mut a = j.transpose() * j;
        let mut e = self.residual(&x);
        let mut g = j.transpose() * e;
        let mut stop = g.norm() <= eps1;
        let mut mu: f64 = tau * {
            let mut max = 0.0;
            for i in 0..C {
                if a[(i, i)] > max {
                    max = a[(i, i)];
                }
            }
            max
        };

        while !stop && k < k_max {
            println!("mu: {}, k: {}", mu, k);
            k += 1;
            let mut a_star = a;
            for i in 0..C {
                a_star[(i, i)] += mu;
            }
            let h = self.solve(&a_star, &(-g));
            if h.norm() <= eps2 * (x.norm() + eps2) {
                stop = true;
            } else {
                let x_new = x + h;
                let e_new = self.residual(&x_new);
                let rho = (e.norm() - e_new.norm()) / (0.5 * h.transpose() * (mu * h - g)).norm();
                println!("rho: {}", rho);
                if rho > 0.0 {
                    x = x_new;
                    a = self.jacobian(&x).transpose() * self.jacobian(&x);
                    e = self.residual(&x);
                    g = self.jacobian(&x).transpose() * e;
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
}


mod test {
    use super::*;

    // y = ax^2 + bx + c
    // a = b = c = 1
    struct TestProblem {
        measurements: Vec<na::Vector2<f64>>,
    }

    const R: usize = 4;
    const C: usize = 3;
    impl LMProblem<f64, 4, 3> for TestProblem {
        fn solve(&self, a_star: &na::SMatrix<f64, C, C>, b: &na::SVector<f64, C>) -> Param<C> {
            a_star.lu().solve(b).unwrap()
        }

        fn jacobian(&self, p: &Param<C>) -> Jacobian<R, C> {
            let mut jac = na::SMatrix::<f64, R, C>::zeros();
            for i in 0..R {
                jac.fixed_view_mut::<1, 3>(i, 0)
                    .copy_from(
                        &(na::Matrix1x3::<f64>::new(
                            -self.measurements[i][0].powi(2),
                            -self.measurements[i][0],
                            -1.0,
                        )),
                    )
            }
            jac
        }

        fn residual(&self, p: &Param<C>) -> Residual<R> {
            let mut res = na::SVector::<f64, R>::zeros();
            for i in 0..R {
                let x = self.measurements[i][0];
                let y = self.measurements[i][1];
                let y_hat = p[0] * x.powi(2) + p[1] * x + p[2];
                res[i] = y - y_hat;
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
        let x0 = na::SVector::<f64, C>::new(10.0, 0.0, 0.0);
        let eps1 = 1e-6;
        let eps2 = 1e-6;
        let tau = 1e-1;
        let k_max = 100;
        let x = problem.run(x0, eps1, eps2, tau, k_max);
        println!("x = {}", x);
    }

}
    