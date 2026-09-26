from __future__ import annotations

import unittest
import numpy as np
from experiments.run_linear_geometry_mechanism import (
    cholesky2, classify, continuous_covariance, euler_covariance,
    paired_bounds, scatter_draws, scores, score_weights)


class LinearGeometryTests(unittest.TestCase):
    def test_closed_form_matches_euler_recursion(self):
        for H in [.2,1.,2.]:
            for L in [1,2,5,10]:
                for kappa in [-.8,0.,1.5]:
                    for sigma2 in [0.,.3]:
                        h=H/L; F=np.array([[1.,0.],[h*kappa,1.]])
                        covariance=np.zeros((2,2)); G=np.diag([.7,sigma2])
                        for _ in range(L):
                            covariance=F@covariance@F.T+h*G@G.T
                        np.testing.assert_allclose(covariance,euler_covariance(H,L,kappa,.7,sigma2),atol=1e-13)

    def test_support_boundary_and_limit(self):
        self.assertEqual(np.linalg.matrix_rank(euler_covariance(1,1,1,1,0)),1)
        self.assertEqual(np.linalg.matrix_rank(euler_covariance(1,2,1,1,0)),2)
        np.testing.assert_allclose(euler_covariance(1,100000,1,1,.4),
                                   continuous_covariance(1,1,1,.4),atol=1e-5)

    def test_weights_agree_with_explicit_quadratic_energy(self):
        rng=np.random.default_rng(31)
        U=rng.normal(size=(12,40,2))
        W=np.einsum('nki,nkj->nij',U,U)
        scatter=np.stack([W[:,0,0],W[:,0,1],W[:,1,1]],axis=-1)
        gen=euler_covariance(1,5,1.4,1.2,.8)
        coord=euler_covariance(1,3,.7,1.,.5)
        r=U@cholesky2(gen).T
        direct=np.einsum('nki,ij,nkj->n',r,np.linalg.inv(coord),r)/(2*40)
        np.testing.assert_allclose(scores(scatter,score_weights(gen,coord),40),direct,atol=1e-12)

    def test_exact_negative_control_scores(self):
        draw=scatter_draws(np.random.default_rng(32),(25,),40)
        for kappa,L in [(0.,10),(2.,1)]:
            inst=np.eye(2); finite=euler_covariance(1,L,kappa,1,1)
            gen=euler_covariance(1,L,kappa,1.2,.9)
            np.testing.assert_array_equal(scores(draw,score_weights(gen,inst),40),
                                          scores(draw,score_weights(gen,finite),40))

    def test_finite_step_null_energy_weights(self):
        cov=euler_covariance(1,10,3,1,1)
        np.testing.assert_allclose(score_weights(cov,cov),[1,0,1],atol=1e-14)

    def test_no_silent_regularization(self):
        with self.assertRaises(ValueError):
            cholesky2(np.diag([1.,0.]))

    def test_equivalence_is_not_nonsignificance(self):
        self.assertEqual(classify((-.08,.08),.02),'unresolved')
        self.assertEqual(classify((-.01,.01),.02),'equivalent')
        self.assertEqual(classify((.03,.08),.02),'materially_higher')
        lo,hi=paired_bounds(0,0,5000,96)
        self.assertLess(lo,0); self.assertGreater(hi,0)

    def test_five_component_constant_scale_identity(self):
        rng=np.random.default_rng(33); z=rng.normal(size=(50,40)); c=1.7
        def components(x):
            e=x*x-1
            return np.column_stack([x.mean(axis=1),np.max(np.abs(np.cumsum(x,axis=1)),axis=1),
                                    (x*x).mean(axis=1),
                                    [np.corrcoef(v[1:],v[:-1])[0,1] for v in e],
                                    [np.corrcoef(v[7:],v[:-7])[0,1] for v in e]])
        a,b=components(z),components(c*z)
        def standardize(x):
            return (x-np.median(x[:25],axis=0))/np.std(x[:25],axis=0,ddof=1)
        np.testing.assert_allclose(standardize(a),standardize(b),atol=1e-12)
        cumulative=np.cumsum(z*z-1,axis=1)
        scaled=np.cumsum((c*z)**2-1,axis=1)
        np.testing.assert_allclose(scaled,c*c*cumulative+(c*c-1)*np.arange(1,41),atol=1e-12)

    def test_cumulative_energy_rescaling_can_reverse_order(self):
        z=np.sqrt(np.array([[4.,0.],[0.,2.5]]))
        stat=lambda x: np.abs(np.cumsum(x*x-1,axis=1)).max(axis=1)
        original,scaled=stat(z),stat(np.sqrt(.1)*z)
        self.assertGreater(original[0],original[1])
        self.assertLess(scaled[0],scaled[1])

    def test_smooth_map_first_variation_error_order(self):
        xi=np.random.default_rng(34).normal(size=(1000,4))
        def error(epsilon):
            y=np.full(1000,.2); bar=.2; u=np.zeros(1000)
            for l in range(4):
                derivative=1-np.tanh(bar)**2
                u=(derivative+.1)*u+.5*derivative*xi[:,l]
                y=np.tanh(y+.5*epsilon*xi[:,l])+.1*y
                bar=np.tanh(bar)+.1*bar
            return np.mean(((y-bar)/epsilon-u)**2)
        coarse,fine=error(.01),error(.005)
        self.assertGreater(coarse/fine,3.8)
        self.assertLess(coarse/fine,4.2)

    def test_common_scale_are_closed_form(self):
        for kappa in [-3., -.5, 0., .5, 3.]:
            for L in [1, 2, 5, 10]:
                step=euler_covariance(1.,L,kappa,1.,1.)
                eigenvalues=np.linalg.eigvalsh(step)
                are_eigen=(2*np.sum(eigenvalues**2)/np.sum(eigenvalues)**2)
                a=(L-1)/(2*L)
                b=(L-1)*(2*L-1)/(6*L*L)
                eta=abs(kappa)
                are_closed=(1+(eta**4*b*b+4*eta*eta*a*a)/(2+eta*eta*b)**2)
                self.assertAlmostEqual(are_eigen,are_closed,places=13)
                self.assertGreaterEqual(are_closed,1.)
                self.assertLessEqual(are_closed,2.)

    def test_common_scale_are_equality_cells_and_limit(self):
        for L in [1, 2, 10]:
            self.assertAlmostEqual(
                2*np.trace(euler_covariance(1,L,0,1,1) @ euler_covariance(1,L,0,1,1))
                / np.trace(euler_covariance(1,L,0,1,1))**2,
                1.,
            )
        for kappa in [.5, 3., 1000.]:
            step=euler_covariance(1,1,kappa,1,1)
            self.assertAlmostEqual(2*np.trace(step@step)/np.trace(step)**2,1.)
        step=euler_covariance(1,10,1e6,1,1)
        self.assertAlmostEqual(2*np.trace(step@step)/np.trace(step)**2,2.,places=10)


if __name__ == '__main__':
    unittest.main()
