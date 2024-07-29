import numpy as np
import pytest
from pathlib import Path
import sys

applications_path = Path(__file__).parents[3].resolve() / "applications"
sys.path.insert(0, str((applications_path / "performUQ" / "UCSD_UQ").resolve()))
import loglike_script


def test_log_likelihood_scalar_standard_normal():
    residuals = np.atleast_2d(0.0).reshape((1, -1))
    mean = 0
    cov = np.atleast_2d(1.0)
    actual_ll = loglike_script.log_likelihood(residuals, mean, cov)

    desired_ll = np.log(1 / np.sqrt(2 * np.pi))

    assert np.allclose(actual_ll, desired_ll)


def test_single_variance():
    residuals = np.array([1, 2, 3])
    mean = 0  # Mean is not used in the current implementation
    cov = np.array([[2]])  # Single variance
    actual_ll = loglike_script.log_likelihood(residuals, mean, cov)

    desired_ll = (
        -3 / 2 * np.log(2)
        - 3 / 2 * np.log(2 * np.pi)
        - 1 / (2 * 2) * np.sum(residuals**2)
    )
    
    assert np.isclose(
        actual_ll,
        desired_ll
    )


def test_multivariate():
    residuals = np.array([1, 2])
    mean = 0  # Mean is not used in the current implementation
    cov = np.array([[1, 0.1], [0.1, 1]])  # Covariance matrix
    actual_ll = loglike_script.log_likelihood(residuals, mean, cov)

    # Calculate the expected log-likelihood
    length = len(residuals)
    t1 = length * np.log(2 * np.pi)
    eigenValues, eigenVectors = np.linalg.eigh(cov)
    logdet = np.sum(np.log(eigenValues))
    eigenValuesReciprocal = 1.0 / eigenValues
    z = eigenVectors * np.sqrt(eigenValuesReciprocal)
    mahalanobisDistance = np.square(np.dot(residuals, z)).sum()
    desired_ll = -0.5 * (t1 + logdet + mahalanobisDistance)

    assert np.isclose(
        actual_ll,
        desired_ll
    )


def test_covariance_vector():
    residuals = np.array([1, 2, 3])
    mean = 0  # Mean is not used in the current implementation
    cov = np.atleast_2d([1, 2, 3])  # Diagonal elements
    actual_ll = loglike_script.log_likelihood(residuals, mean, cov)

    length = len(residuals)
    t1 = length * np.log(2 * np.pi)
    logdet = np.sum(np.log(cov))
    eigenValuesReciprocal = 1.0 / cov
    z = np.sqrt(eigenValuesReciprocal)
    mahalanobisDistance = np.sum((residuals * z) ** 2)
    desired_ll = -0.5 * (t1 + logdet + mahalanobisDistance)

    assert np.isclose(
        actual_ll,
        desired_ll
    )


def test_incorrect_covariance_shape():
    residuals = np.array([1, 2, 3])
    mean = 0  # Mean is not used in the current implementation
    cov = np.array([[1, 2], [3, 4]])  # Incorrect shape

    with pytest.raises(ValueError):
        loglike_script.log_likelihood(residuals, mean, cov)
