add_subdirectory(database)
add_subdirectory(gmpe)
add_subdirectory(lib)
simcenter_add_python_script(SCRIPT ComputeIntensityMeasure.py)
simcenter_add_python_script(SCRIPT CreateScenario.py)
simcenter_add_python_script(SCRIPT CreateStation.py)
simcenter_add_python_script(SCRIPT FetchOpenSHA.py)
simcenter_add_python_script(SCRIPT HazardSimulation.py)
simcenter_add_python_script(SCRIPT SelectGroundMotion.py)
