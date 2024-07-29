from pathlib import Path
import sys

applications_path = Path(__file__).parents[3].resolve() / "applications"
sys.path.insert(0, str((applications_path / "performUQ" / "UCSD_UQ").resolve()))
import parseData

import json
import sys
import pytest
import os
import time
import importlib

# Sample JSON data for the tests
with open(Path(__file__).parent / "data" / "input.json", "r") as f:
    sample_json_data = json.load(fp=f)
    print(f"{sample_json_data = }")

# Fixture to provide the sample JSON file
@pytest.fixture
def sample_json_file(tmp_path):
    json_file = tmp_path / "sample.json"
    with open(json_file, "w") as f:
        json.dump(sample_json_data, f)
    return json_file

# Fixture for the log file
@pytest.fixture
def log_file(tmp_path):
    where = tmp_path
    logfile_name = "logFileTMCMC.txt"
    logfile = open(os.path.join(where, logfile_name), "w")
    logfile.write(
        "Starting analysis at: {}".format(
            time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
        )
    )
    logfile.write("\nRunning quoFEM's UCSD_UQ engine workflow")
    logfile.write("\nCWD: {}".format(os.path.abspath(".")))
    return logfile

# # Mocking os.path.exists to return True
# @pytest.fixture
# def mock_os_path_exists():
#     with patch("os.path.exists", return_value=True) as mock_exists:
#         yield mock_exists

# Mocking import_module to simulate successful module import
@pytest.fixture
def mock_import_module():
    mod = "loglike_script.py"
    mock_import = importlib.import_module(mod)
    yield mock_import

def test_parse_data_function_success(
    sample_json_file, log_file,
):
    result = parseData.parseDataFunction(
        dakotaJsonFile=sample_json_file,
        logFile=log_file,
        tmpSimCenterDir="tmp/dir",
        mainscriptDir="main/dir",
    )
    assert result[0] == 200  # nSamples
    assert result[1] == 0  # seedValue
    assert result[2] == "eigData.csv"  # calDataFile

# def test_parse_data_function_missing_log_likelihood(
#     sample_json_file, log_file, 
# ):
#     json_data = sample_json_data.copy()
#     json_data["UQ"]["logLikelihoodFile"] = "non_existent_script.py"
    
#     with open(sample_json_file, "w") as f:
#         json.dump(json_data, f)
    
#     with pytest.raises(FileNotFoundError):
#         parseData.parseDataFunction(
#             dakotaJsonFile=sample_json_file,
#             logFile=log_file,
#             tmpSimCenterDir="tmp/dir",
#             mainscriptDir="main/dir",
#         )

# def test_parse_data_function_default_log_likelihood(
#     sample_json_file, log_file, mock_import_module
# ):
#     json_data = sample_json_data.copy()
#     assert json_data["UQ"]["logLikelihoodFile"] == ""

#     with open(sample_json_file, "w") as f:
#         json.dump(json_data, f)

#     result = parseData.parseDataFunction(
#         dakotaJsonFile=sample_json_file,
#         logFile=log_file,
#         tmpSimCenterDir="tmp/dir",
#         mainscriptDir="main/dir",
#     )
#     # assert mock_import_module.called
#     assert "defaultLogLikeScript" in mock_import_module.call_args[0][0]

def test_parse_data_function_rv_processing(sample_json_file, log_file):
    result = parseData.parseDataFunction(
        dakotaJsonFile=sample_json_file,
        logFile=log_file,
        tmpSimCenterDir="tmp/dir",
        mainscriptDir="main/dir",
    )
    variables_list = result[3]
    print(f'{variables_list = }')
    assert variables_list[0]["names"] == ['k1', 'k2', 'lambda.CovMultiplier', 'phi.CovMultiplier']
    assert variables_list[0]["distributions"] == ['Uniform', 'Uniform', 'InvGamma', 'InvGamma']
    assert variables_list[0]["Par1"] == [766.89, 383.44, 3, 3]
    assert variables_list[0]["Par2"] == [2108.94, 1150.33, 2, 2]
    assert variables_list[0]["Par3"] == [None, None, None, None]
    assert variables_list[0]["Par4"] == [None, None, None, None]

# More tests for different random variable distributions, EDP processing, etc.
