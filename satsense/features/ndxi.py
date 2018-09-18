"""Implementation of the NDXI family of features."""
import numpy as np

from ..bands import BANDS
from .feature import Feature

NDXI_OPTIONS = {
    'nir_ndvi': ('red', 'nir-1'),
    'rg_ndvi': ('red', 'green'),
    'rb_ndvi': ('red', 'blue'),
    'ndsi': ('green', 'nir-1'),
    'ndwi': ('coastal', 'nir-2'),
    'wvsi': ('green', 'yellow'),
}


def ndxi_feature(image, option, bands):
    """Calculates the feature according to the ndxi option provided."""
    band_0_name, band_1_name = NDXI_OPTIONS[option]
    band_0 = image[:, :, bands[band_0_name]]
    band_1 = image[:, :, bands[band_1_name]]

    band_mix = band_0 + band_1
    # Ignore divide, this division may complain about division by 0
    # This usually happens in the edge, which is alright by us.
    old_settings = np.seterr(divide='ignore', invalid='ignore')
    ndxi = np.divide(band_0 - band_1, band_mix)
    ndxi[band_mix == 0] = 0
    np.seterr(**old_settings)

    return ndxi


def print_ndxi_statistics(ndxi, option):
    """Prints the ndvi matrix and the, min, max, mean and median."""
    print('{o} matrix: '.format(o=option))
    print(ndxi)

    print('\nMax {o}: {m}'.format(o=option, m=np.nanmax(ndxi)))
    print('Mean {o}: {m}'.format(o=option, m=np.nanmean(ndxi)))
    print('Median {o}: {m}'.format(o=option, m=np.nanmedian(ndxi)))
    print('Min {o}: {m}'.format(o=option, m=np.nanmin(ndxi)))


class NDXI(Feature):
    """The parent class of the family of NDXI features."""

    def __init__(self, option, windows=((25, 25), )):
        super().__init__(windows=windows)
        self.option = option
        self.feature_size = len(self.windows)
        self.base_image = 'normalized'

    def __call__(self, cell):
        result = np.zeros(self.feature_size)
        for i, window in enumerate(self.windows):
            win = cell.super_cell(window, padding=True)
            ndxi_result = ndxi_feature(
                win.normalized, self.option, bands=cell.bands)
            result[i] = ndxi_result.mean()
        return result


class NirNDVI(NDXI):
    """The infrared-green normalized difference vegetation index of the image."""

    def __init__(self, windows=((25, 25), )):
        super(NirNDVI, self).__init__('nir_ndvi', windows=windows)


class RgNDVI(NDXI):
    """The red-green normalized difference vegetation index of the image."""

    def __init__(self, windows=((25, 25), )):
        super(RgNDVI, self).__init__('rg_ndvi', windows=windows)


class RbNDVI(NDXI):
    """The red-blue normalized difference vegetation index of the image."""

    def __init__(self, windows=((25, 25), )):
        super(RbNDVI, self).__init__('rb_ndvi', windows=windows)


class NDSI(NDXI):
    """The snow cover index of the image."""

    def __init__(self, windows=((25, 25), )):
        super(NDSI, self).__init__('ndsi', windows=windows)


class NDWI(NDXI):
    """The water cover index of the image."""

    def __init__(self, windows=((25, 25), )):
        super(NDWI, self).__init__('ndwi', windows=windows)


class WVSI(NDXI):
    """The soil cover index of the image."""

    def __init__(self, windows=((25, 25), )):
        super(WVSI, self).__init__('wvsi', windows=windows)
