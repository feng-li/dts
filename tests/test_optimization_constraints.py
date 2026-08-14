from types import SimpleNamespace
import unittest

from jax import grad, hessian
import jax.numpy as jnp
import numpy as np

from dts.mcmc import parameter_bounds, sampler, whittle_log_posterior_jit
from dts.optimization import fit_map_and_proposal


class PosteriorConstraintTests(unittest.TestCase):
    def setUp(self):
        self.periodogram = jnp.array([1.0, 0.8, 1.2])
        self.omega = jnp.array([0.4, 0.8, 1.2])

    def log_posterior(self, theta):
        return whittle_log_posterior_jit(
            theta,
            1,
            0,
            self.periodogram,
            False,
            self.omega,
            1,
            0.0,
            1.0,
            1,
        )

    def test_posterior_is_negative_infinity_outside_process_support(self):
        value = self.log_posterior(jnp.array([1.01, 0.0]))
        self.assertTrue(np.isneginf(float(value)))

    def test_posterior_derivatives_are_finite_inside_process_support(self):
        theta = jnp.array([0.2, 0.0])
        objective = lambda value: -self.log_posterior(value)

        gradient = np.asarray(grad(objective)(theta))
        curvature = np.asarray(hessian(objective)(theta))

        self.assertTrue(np.all(np.isfinite(gradient)))
        self.assertTrue(np.all(np.isfinite(curvature)))

    def test_parameter_bounds_are_strict_for_process_parameters(self):
        lower, upper = parameter_bounds(5, tfi_term=True)

        self.assertTrue(np.all(lower[:2] > -1.0))
        self.assertTrue(np.all(upper[:2] < 1.0))
        np.testing.assert_array_equal(lower[-3:], -30.0)
        np.testing.assert_array_equal(upper[-3:], 30.0)

    def test_sampler_retains_only_supported_process_parameters(self):
        draws, _, _ = sampler(
            ar_order=1,
            ma_order=0,
            data=None,
            I_pg=np.asarray(self.periodogram),
            TFI_term=False,
            omega_shard=np.asarray(self.omega),
            n_samples=20,
            paramsStar=np.array([0.0, 0.0]),
            proposal_width=np.diag([4.0, 0.01]),
            Burn_in=0,
            random_state=7,
        )

        self.assertTrue(np.all(np.abs(draws[:, 0]) < 1.0))


class BoundedBasinhoppingTests(unittest.TestCase):
    def test_basinhopping_never_supplies_an_infeasible_local_start(self):
        settings = SimpleNamespace(
            optimize=True,
            max_iter_optim=50,
            gtol=1e-8,
            proposal_scale=None,
            basinhopping=True,
            basinhopping_iter=3,
            seed=11,
            progress=False,
        )
        lower = np.array([-0.05, -0.05])
        upper = np.array([0.05, 0.05])
        target = jnp.array([0.025, -0.02])

        fit = fit_map_and_proposal(
            lambda theta: jnp.sum((theta - target) ** 2),
            np.zeros(2),
            lower,
            upper,
            settings,
        )

        self.assertTrue(np.all(fit.theta >= lower))
        self.assertTrue(np.all(fit.theta <= upper))

    def test_infeasible_initial_point_is_rejected(self):
        settings = SimpleNamespace(optimize=False)

        with self.assertRaisesRegex(ValueError, "theta0 must satisfy"):
            fit_map_and_proposal(
                lambda theta: jnp.sum(theta**2),
                np.array([2.0]),
                np.array([-1.0]),
                np.array([1.0]),
                settings,
            )


if __name__ == "__main__":
    unittest.main()
