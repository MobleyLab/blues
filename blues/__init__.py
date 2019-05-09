#!/usr/local/bin/env python
"""
BLUES
"""
# Add imports here
#from blues import integrators, reporters, settings, ncmc, utils

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__short_version__ = __version__.split('+')[0]
__git_revision__ = versions['full-revisionid']
del get_versions, versions
