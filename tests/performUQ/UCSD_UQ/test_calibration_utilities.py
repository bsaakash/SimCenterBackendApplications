from pathlib import Path
import sys

applications_path = Path(__file__).parents[3].resolve() / "applications"
sys.path.insert(
    0, str((applications_path / "performUQ" / "UCSD_UQ").resolve())
)
import calibration_utilities

import pytest
import numpy as np
from unittest.mock import MagicMock, mock_open, patch


class TestCovarianceMatrixPreparer:

    @pytest.fixture
    def setup(self):
        # Sample data for testing
        calibration_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        edp_lengths_list = [1, 2]
        edp_names_list = ["edp1", "edp2"]
        workdir_main = "test/workdir"
        num_experiments = 2
        log_file = MagicMock()
        run_type = "runningLocal"

        # Instance of CovarianceMatrixPreparer
        preparer = calibration_utilities.CovarianceMatrixPreparer(
            calibrationData=calibration_data,
            edpLengthsList=edp_lengths_list,
            edpNamesList=edp_names_list,
            workdirMain=workdir_main,
            numExperiments=num_experiments,
            logFile=log_file,
            runType=run_type,
        )

        return preparer, log_file
    
    @pytest.fixture
    def setup2(self):
        # Sample data for testing
        calibration_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        edp_lengths_list = [1, 2]
        edp_names_list = ["edp1", "edp2"]
        workdir_main = "test/workdir"
        num_experiments = 2
        log_file = MagicMock()
        run_type = "runningLocal"

        # Instance of CovarianceMatrixPreparer
        preparer = calibration_utilities.CovarianceMatrixPreparer(
            calibrationData=calibration_data,
            edpLengthsList=edp_lengths_list,
            edpNamesList=edp_names_list,
            workdirMain=workdir_main,
            numExperiments=num_experiments,
            logFile=log_file,
            runType=run_type,
        )

        return preparer, log_file

    def test_init(self, setup):
        preparer, log_file = setup

        # Check initialization of attributes
        assert np.array_equal(
            preparer.calibrationData,
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        )
        assert preparer.edpLengthsList == [1, 2]
        assert preparer.edpNamesList == ["edp1", "edp2"]
        assert preparer.workdirMain == "test/workdir"
        assert preparer.numExperiments == 2
        assert preparer.logFile == log_file
        assert preparer.runType == "runningLocal"

        # Check log file writes
        log_file.write.assert_any_call("\n\n==========================")
        log_file.write.assert_any_call(
            "\nProcessing options for variance/covariance:"
        )

    def test_get_default_error_variances(self, setup):
        preparer, log_file = setup

        # Call the method to compute default error variances
        preparer.getDefaultErrorVariances()

        # Verify the computed default error variances
        expected_variances = np.array([2.25, 2.5])
        assert np.allclose(
            preparer.defaultErrorVariances, expected_variances, atol=1e-12
        )

    @patch("os.path.isfile", return_value=False)
    def test_create_covariance_matrix_default(self, mock_isfile, setup):
        preparer, log_file = setup

        # Set default error variances before calling createCovarianceMatrix
        preparer.getDefaultErrorVariances()

        # Call the method to create the covariance matrix
        result = preparer.createCovarianceMatrix()

        # Verify the covariance matrices
        expected_covariance_matrix = [
            2.25,  # scalar variance
            2.5,  # scalar variance
        ]
        assert (
            len(result)
            == len(preparer.edpLengthsList) * preparer.numExperiments
        )
        assert np.allclose(
            result[0], expected_covariance_matrix[0], atol=1e-12
        )
        assert np.allclose(
            result[1], expected_covariance_matrix[1], atol=1e-12
        )

        # Verify log file writes
        log_file.write.assert_any_call(
            "\n\nLooping over the experiments and EDPs"
        )
        log_file.write.assert_any_call(
            "\n\t\tDid not find a user supplied file. Using the default variance value."
        )

    # @patch("os.path.isfile", return_value=True)
    # @patch(
    #     "builtins.open", new_callable=mock_open, read_data="1.0 0.5\n0.5 1.0"
    # )
    # @patch("shutil.copyfile")
    # def test_create_covariance_matrix_user_supplied(
    #     self, mock_copyfile, mock_open, mock_isfile, setup
    # ):
    def test_create_covariance_matrix_user_supplied(
        self, setup
    ):
        preparer, log_file = setup

        # Set default error variances before calling createCovarianceMatrix
        preparer.getDefaultErrorVariances()

        # Call the method to create the covariance matrix
        result = preparer.createCovarianceMatrix()

        print(f"{result = }")

        # Verify the covariance matrices read from user-supplied files
        expected_covariance_matrix = [
            np.array([[1.0, 0.5], [0.5, 1.0]]),
            np.array([[1.0, 0.5], [0.5, 1.0]]),
        ]
        assert len(result) == 2
        assert np.allclose(
            result[0], expected_covariance_matrix[0], atol=1e-12
        )
        assert np.allclose(
            result[1], expected_covariance_matrix[1], atol=1e-12
        )

        # Verify log file writes
        log_file.write.assert_any_call(
            "\n\nLooping over the experiments and EDPs"
        )
        log_file.write.assert_any_call("\n\t\tFound a user supplied file.")
        log_file.write.assert_any_call(
            "\n\t\tReading in user supplied covariance matrix from file: 'test/workdir/edp1.1.sigma'"
        )

    # Additional tests for edge cases and exception handling can be added here
