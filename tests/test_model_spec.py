import numpy as np

from dts.aggregation import parameter_names, transform_partial_draws
from dts.mcmc import ModelSpec
from dts.regression import RegressionSpec, regression_parameter_names


def test_model_spec_uses_explicit_ar_and_ma_orders():
    spec = ModelSpec(ar_order=2, ma_order=1, tfi_term=False)

    assert spec.ar_order == 2
    assert spec.ma_order == 1
    assert spec.n_params == 4
    assert parameter_names(spec.ar_order, spec.ma_order, spec.tfi_term) == [
        "phi1",
        "phi2",
        "theta1",
        "sigma2",
    ]


def test_transform_partial_draws_uses_ar_parameters_before_ma_parameters():
    draws = np.array([[0.2, 0.3, 0.4, np.log(2.0)]])

    transformed = transform_partial_draws(
        draws,
        ar_order=2,
        ma_order=1,
        tfi_term=False,
    )

    np.testing.assert_allclose(transformed[0], [0.14, 0.3, 0.4, 2.0])


def test_regression_spec_uses_explicit_error_orders():
    spec = RegressionSpec(ar_order=2, ma_order=1, n_exog=2)

    assert spec.n_params == 6
    assert spec.process_slice == slice(0, 3)
    assert spec.beta_slice == slice(3, 5)
    assert regression_parameter_names(spec) == [
        "phi1",
        "phi2",
        "theta1",
        "beta1",
        "beta2",
        "sigma2",
    ]
