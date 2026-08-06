import importlib.util
import unittest

import numpy as np

from DTS_AR2_Spark import (
    build_indexed_rdd,
    distributed_fft,
    regression_whittle_log_likelihood,
    residual_periodogram,
)


class RegressionWhittleTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(71)
        self.n_obs = 64
        self.x = rng.normal(size=self.n_obs)
        self.y = 0.7 * self.x + rng.normal(size=self.n_obs)
        self.x_hat = np.fft.fft(self.x)
        self.y_hat = np.fft.fft(self.y)
        self.indices = np.arange(1, (self.n_obs - 1) // 2 + 1)
        self.omega = 2 * np.pi * self.indices / self.n_obs
        self.params = np.array([0.25, -0.3, 0.7, np.log(1.2)])

    def test_residual_periodogram_matches_direct_fft(self):
        actual = residual_periodogram(
            self.y_hat[self.indices].real,
            self.y_hat[self.indices].imag,
            self.x_hat[self.indices, None].real,
            self.x_hat[self.indices, None].imag,
            self.params[2:3],
            self.n_obs,
        )
        residual_hat = np.fft.fft(self.y - self.params[2] * self.x)
        expected = np.abs(residual_hat[self.indices]) ** 2 / (
            2 * np.pi * self.n_obs
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_shard_likelihoods_sum_to_full_likelihood(self):
        kwargs = dict(
            params=self.params,
            y_real=self.y_hat[self.indices].real,
            y_imag=self.y_hat[self.indices].imag,
            x_real=self.x_hat[self.indices, None].real,
            x_imag=self.x_hat[self.indices, None].imag,
            omega=self.omega,
            n_obs=self.n_obs,
            n_exog=1,
        )
        full = float(regression_whittle_log_likelihood(**kwargs))
        pieces = 0.0
        for group in range(4):
            positions = np.arange(group, len(self.indices), 4)
            pieces += float(
                regression_whittle_log_likelihood(
                    self.params,
                    kwargs["y_real"][positions],
                    kwargs["y_imag"][positions],
                    kwargs["x_real"][positions],
                    kwargs["x_imag"][positions],
                    self.omega[positions],
                    self.n_obs,
                    1,
                )
            )
        self.assertAlmostEqual(full, pieces, places=11)


@unittest.skipUnless(importlib.util.find_spec("pyspark"), "pyspark is unavailable")
class DistributedFFTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pyspark.sql import SparkSession

        cls.spark = (
            SparkSession.builder.master("local[2]")
            .appName("dts-ar2-dfft-test")
            .config("spark.ui.enabled", "false")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .getOrCreate()
        )

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_dfft_matches_numpy_and_respects_time_column(self):
        values = np.array(
            [0.5, -1.0, 0.25, 2.0, -0.75, 1.25, 0.0, 0.8,
             1.1, -0.2, 0.3, -1.4, 0.9, 0.6, -0.5, 1.7]
        )
        rows = [(int(index), float(values[index])) for index in range(len(values) - 1, -1, -1)]
        frame = self.spark.createDataFrame(rows, ["time", "value"])
        expected = np.fft.fft(values)

        for partitions in (1, 2, 4, 8):
            indexed, n_obs, trimmed = build_indexed_rdd(
                frame, ["value"], partitions, "end", "time"
            )
            self.assertEqual(trimmed, 0)
            result = distributed_fft(
                self.spark, indexed, 0, n_obs, partitions, "value"
            )
            rows_out = result.orderBy("k").collect()
            actual = np.array(
                [complex(row["value_real"], row["value_imag"]) for row in rows_out]
            )
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
            result.unpersist()
            indexed.unpersist()

    def test_start_trim_retains_latest_regular_observations(self):
        values = np.arange(18, dtype=float) ** 2 / 10.0
        frame = self.spark.createDataFrame(
            [(int(index), float(value)) for index, value in enumerate(values)],
            ["time", "value"],
        )
        indexed, n_obs, trimmed = build_indexed_rdd(
            frame, ["value"], 4, "start", "time"
        )
        self.assertEqual(n_obs, 16)
        self.assertEqual(trimmed, 2)
        result = distributed_fft(self.spark, indexed, 0, n_obs, 4, "value")
        rows_out = result.orderBy("k").collect()
        actual = np.array(
            [complex(row["value_real"], row["value_imag"]) for row in rows_out]
        )
        np.testing.assert_allclose(
            actual,
            np.fft.fft(values[2:]),
            rtol=1e-12,
            atol=1e-12,
        )
        result.unpersist()
        indexed.unpersist()

    def test_duplicate_time_values_are_rejected(self):
        frame = self.spark.createDataFrame(
            [(0.0, 1.0), (1.0, 2.0), (1.0, 3.0), (2.0, 4.0)],
            ["time", "value"],
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            build_indexed_rdd(frame, ["value"], 2, "end", "time")

    def test_irregular_time_values_are_rejected(self):
        frame = self.spark.createDataFrame(
            [(0.0, 1.0), (1.0, 2.0), (3.0, 3.0), (4.0, 4.0)],
            ["time", "value"],
        )
        with self.assertRaisesRegex(ValueError, "regularly spaced"):
            build_indexed_rdd(frame, ["value"], 2, "end", "time")


if __name__ == "__main__":
    unittest.main()
