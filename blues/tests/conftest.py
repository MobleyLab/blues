# conftest.py
import openmm
import pytest

def get_preferred_platform():
    num_platforms = openmm.Platform.getNumPlatforms()
    available = [openmm.Platform.getPlatform(i).getName() for i in range(num_platforms)]
    print("Available platforms:", available)
    
    # Prefer CUDA if available
    if "CUDA" in available:
        return "CUDA"
    elif "OpenCL" in available:
        return "OpenCL"
    else:
        return "CPU"

@pytest.fixture(scope="session")
def preferred_platform():
    return get_preferred_platform()