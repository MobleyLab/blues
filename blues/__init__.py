#!/usr/local/bin/env python
"""
BLUES
"""
# Define global version.
#from . import version
#__version__ = version.version

# Add imports here
from .blues import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
