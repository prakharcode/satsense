"""Satsense package."""
from pkg_resources import get_distribution, DistributionNotFound
import os.path

try:
    _dist = get_distribution('foobar')
    # Normalize case for Windows systems
    dist_loc = os.path.normcase(_dist.location)
    here = os.path.normcase(__file__)
    if not here.startswith(os.path.join(dist_loc, 'foobar')):
        # not installed, but there is another version that *is*
        raise DistributionNotFound
except DistributionNotFound:
    __version__ = 'Please install this project with setup.py'
else:
    __version__ = _dist.version


from .bands import BANDS
from .extract import (extract_features, extract_features_parallel,
                      load_features, save_features, get_feature_filenames)
from .image import SatelliteImage

__all__ = [
    'BANDS',
    'extract_features',
    'extract_features_parallel',
    'load_features',
    'save_features',
    'SatelliteImage',
]
