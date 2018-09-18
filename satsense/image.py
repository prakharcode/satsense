"""Methods for loading images."""

import logging
import math
import warnings

import numpy as np
import rasterio
from osgeo import gdal
from scipy import ndimage
from skimage import color, img_as_ubyte
from skimage.feature import canny
from skimage.filters import gabor_kernel, gaussian
from skimage.filters.rank import equalize
from skimage.morphology import disk

from .bands import BANDS

gdal.AllRegister()

logger = logging.getLogger(__name__)


class Image:
    def __init__(self, image, bands=None, itype='raw', compute=True):
        if bands is None:
            bands = self._guess_bands(image.shape)
        if not isinstance(bands, dict):
            bands = BANDS[bands]
        if len(bands) == 1 and image is not None and len(image.shape) == 2:
            image = np.reshape(image, image.shape + (1, ))
        self._bands = bands
        self._normalization_parameters = {
            'technique': 'cumulative',
            'percentiles': [2.0, 98.0],
            'numstds': 2,
        }

        # Don't store the raw image, it's not used anywhere
        # Only store the normalized one
        self._images = {}
        if compute:
            # When creating a window compute is set to False
            # to prevent re-calculating
            if itype == 'raw':
                self._images['normalized'] = get_normalized_image(
                    image, self.bands, **self._normalization_parameters)
            elif itype == 'normalized':
                self._images['normalized'] = image
            else:
                raise ValueError("Image only supports 'raw' and 'normalized' images")

    @staticmethod
    def _guess_bands(shape):
        """Try to guess the bands from the array shape."""
        bands = None
        if len(shape) == 2:
            bands = 'monochrome'
        elif len(shape) == 3:
            if shape[2] == 1:
                bands = 'monochrome'
            elif shape[2] == 3:
                bands = 'rgb'

        if bands is None:
            raise ValueError("Unable to guess bands for array of shape {}"
                             .format(shape))
        return bands

    @property
    def raw(self):
        return self._images['raw']

    @property
    def bands(self):
        return self._bands

    @property
    def normalized(self):
        return self._images['normalized']

    @property
    def rgb(self):
        return get_rgb_image(self.normalized, self.bands)

    @property
    def grayscale(self):
        return get_grayscale_image(self.rgb, BANDS['rgb'])

    @property
    def gray_ubyte(self):
        return get_gray_ubyte_image(self.grayscale)

    @property
    def canny_edge(self):
        if 'canny_edge' not in self._images:
            if isinstance(self, Window):
                raise ValueError("Unable to compute canny_edged on Window, "
                                 "compute this on the full image.")
            # TODO: check if we should use gray_ubyte instead
            self._images['canny_edge'] = get_canny_edge_image(
                self.grayscale, radius=30, sigma=0.5)

        return self._images['canny_edge']

    @property
    def texton_descriptors(self):
        if 'texton_descriptors' not in self._images:
            if isinstance(self, Window):
                raise ValueError("Unable to compute texton descriptors on "
                                 "Window, compute this on the full image.")
            self._images['texton_descriptors'] = get_texton_descriptors(
                self.grayscale)

        return self._images['texton_descriptors']

    @property
    def shape(self):
        """Two dimensional shape of the image."""
        return self._images[next(iter(self._images))].shape[:2]

    def shallow_copy_range(self, x_range, y_range, pad=True):
        """Create a shallow copy."""
        # We need a normalized image, because normalization breaks
        # if you do it on a smaller range
        if 'raw' in self._images or 'normalized' in self._images:
            itype = 'normalized'
            image = self.normalized
        else:
            # Pick something random if the preferred images are not available
            itype = next(iter(self._images))
            image = self._images[itype]
        img = Image(image[x_range, y_range], self._bands, itype=itype)

        for itype in self._images:
            if itype not in img._images:
                img._images[itype] = self._images[itype][x_range, y_range]

        if not pad:
            return img

        # Check whether we need padding. This should only be needed at the
        # right and bottom edges of the image
        x_pad_before = 0
        y_pad_before = 0

        x_pad_after = 0
        y_pad_after = 0
        pad_needed = False
        if x_range.stop is not None and x_range.stop > self.shape[0]:
            pad_needed = True
            x_pad_after = x_range.stop - self.shape[0]
        if y_range.stop is not None and y_range.stop > self.shape[1]:
            pad_needed = True
            y_pad_after = y_range.stop - self.shape[1]

        if pad_needed:
            img.pad(x_pad_before, x_pad_after, y_pad_before, y_pad_after)

        return img

    def pad(self, x_pad_before: int, x_pad_after: int, y_pad_before: int,
            y_pad_after: int):
        """Pad the image."""
        for img_type in self._images:
            pad_width = (
                (x_pad_before, x_pad_after),
                (y_pad_before, y_pad_after),
            )
            if len(self._images[img_type].shape) == 3:
                pad_width += ((0, 0), )

            self._images[img_type] = np.pad(
                self._images[img_type],
                pad_width,
                'constant',
                constant_values=0)

    @staticmethod
    def minimal_image_types(itypes):
        """Get the minimal set of images from which itypes can be derived."""
        required_images = set()

        # Images are derived in the following order, select only one
        order = ('raw', 'normalized', 'rgb', 'grayscale', 'gray_ubyte')
        for itype in order:
            if itype in itypes:
                required_images.add(itype)
                break

        # Add additional images types that must be computed on the entire image
        for itype in ('canny_edge', 'texton_descriptors'):
            if itype in itypes:
                required_images.add(itype)

        return required_images

    def precompute(self, itypes):
        """Precompute images."""
        for itype in itypes:
            getattr(self, itype)

    def collapse(self, itypes):
        """Precompute images and remove no longer needed image types."""
        required_images = self.minimal_image_types(itypes)
        self._images = {i: getattr(self, i) for i in required_images}


class Window(Image):
    """Part of an image.

    At a certain x, y location, with an x_range, y_range extent (slice).
    """

    def __init__(self,
                 image: Image,
                 x: int,
                 y: int,
                 x_range: slice,
                 y_range: slice,
                 orig: Image = None):
        super().__init__(image=image, bands=image.bands, compute=False)

        self._images = image._images

        self.x = x
        self.y = y
        self.x_range = x_range
        self.y_range = y_range

        if orig:
            self.image = orig
        else:
            self.image = image


class SatelliteImage(Image):
    def __init__(self, array, satellite, name='', crs=None, transform=None, bounds=None, res=None):
        super().__init__(array, bands=satellite)
        self.name = name
        self.transform = transform
        self.crs = crs
        self.bounds = bounds
        self.res = res

    @classmethod
    def load_from_file(cls, path, satellite):
        """Load the specified path and bands from file into a numpy array."""
        with rasterio.open(path) as dataset:
            image = dataset.read(masked=True)
            crs = dataset.crs
            transform = dataset.transform
            bounds = dataset.bounds
            res = dataset.res

        if len(image.shape) == 3:
            # The bands column is in the first position, but we want it last
            image = np.rollaxis(image, 0, 3)
        elif len(image.shape) == 2:
            # This image seems to have one band, so we add an axis for ease
            # of use in the rest of the library
            image = image[:, :, np.newaxis]

        return cls(image, satellite, name=path, crs=crs, transform=transform, bounds=bounds, res=res)

    def scaled_transform(self, cell_size):
        """Compute a transform for a scaled down version of the image."""
        x_length = math.ceil(self.shape[0] / cell_size[0])
        y_length = math.ceil(self.shape[1] / cell_size[1])
        (west, south, east, north) = rasterio.transform.array_bounds(
            self.shape[0], self.shape[1], self.transform)
        transform = rasterio.transform.from_bounds(west, south, east, north,
                                                   x_length, y_length)
        return transform


def get_normalized_image(image,
                         bands,
                         technique='cumulative',
                         percentiles=(2.0, 98.0),
                         numstds=2):
    """Normalize the image based on the band maximum."""
    logger.debug("Computing normalized image")
    result = image.copy()
    for band in bands.values():
        if np.ma.is_masked(image):
            # select only non-masked values for computing scale
            selection = image[:, :, band].compressed()
        else:
            selection = image[:, :, band]
        if technique == 'cumulative':
            percents = np.nanpercentile(selection, percentiles)
            new_min, new_max = percents
        elif technique == 'meanstd':
            mean = selection.nanmean()
            std = selection.nanstd()
            new_min = mean - (numstds * std)
            new_max = mean + (numstds * std)
        else:
            new_min = selection.nanmin()
            new_max = selection.nanmax()

        result[:, :, band] = remap(image[:, :, band], new_min, new_max, 0, 1)

        np.clip(result[:, :, band], a_min=0, a_max=1, out=result[:, :, band])
    logger.debug("Done computing normalized image")
    return result


def get_rgb_image(image, bands):
    """Convert the image to rgb format."""
    #     logger.debug("Computing rgb image")
    if bands != BANDS['monochrome']:
        red = image[:, :, bands['red']]
        green = image[:, :, bands['green']]
        blue = image[:, :, bands['blue']]

        result = np.rollaxis(np.array([red, green, blue]), 0, 3)
    else:
        result = color.grey2rgb(image)

    #     logger.debug("Done computing rgb image")
    return result


def remap(image, o_min, o_max, n_min, n_max):
    # range check
    if o_min == o_max:
        # print("Warning: Zero input range")
        return 0

    if n_min == n_max:
        # print("Warning: Zero output range")
        return 0

    # check reversed input range
    reverse_input = False
    old_min = min(o_min, o_max)
    old_max = max(o_min, o_max)
    if not old_min == o_min:
        reverse_input = True

    # check reversed output range
    reverse_output = False
    new_min = min(n_min, n_max)
    new_max = max(n_min, n_max)
    if not new_min == n_min:
        reverse_output = True


#     print("Remapping from range [{0}-{1}] to [{2}-{3}]"
#           .format(old_min, old_max, new_min, new_max))
    scale = (new_max - new_min) / (old_max - old_min)
    if reverse_input:
        portion = (old_max - image) * scale
    else:
        portion = (image - old_min) * scale

    if reverse_output:
        result = new_max - portion
    else:
        result = portion + new_min

    return result


def get_grayscale_image(image, bands):
    #     logger.debug("Computing grayscale image")
    if bands != BANDS['rgb']:
        rgb_image = get_rgb_image(image, bands)
    else:
        rgb_image = image

    result = color.rgb2gray(rgb_image)
    #     logger.debug("Done computing grayscale image")
    return result


def get_gray_ubyte_image(image):
    """Convert image in 0 - 1 scale format to ubyte 0 - 255 format.

    Uses img_as_ubyte from skimage.
    """
    #     logger.debug("Computing gray ubyte image")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Ignore loss of precision warning
        result = img_as_ubyte(image)

    #     logger.debug("Done computing gray ubyte image")
    return result


def get_canny_edge_image(image, radius, sigma):
    """Compute Canny edge image."""
    logger.debug("Computing Canny edge image")
    # local histogram equalization
    grayscale = equalize(image, selem=disk(radius))
    try:
        result = canny(grayscale, sigma=sigma)
    except TypeError:
        print("Canny type error")
        result = np.zeros(image.shape)
    logger.debug("Done computing Canny edge image")
    return result


def create_texton_kernels():
    """Create filter bank kernels."""
    kernels = []
    angles = 8
    thetas = np.linspace(0, np.pi, angles)
    for theta in thetas:
        for sigma in (1, ):
            for frequency in (0.05, ):
                kernel = np.real(
                    gabor_kernel(
                        frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)

    return kernels


def get_texton_descriptors(image):
    """Compute texton descriptors."""
    logger.debug("Computing texton descriptors")
    kernels = create_texton_kernels()
    length = len(kernels) + 1
    result = np.zeros(image.shape + (length, ), dtype=np.double)
    for k, kernel in enumerate(kernels):
        result[:, :, k] = ndimage.convolve(image, kernel, mode='wrap')

    # Calculate Difference-of-Gaussian
    dog = gaussian(image, sigma=1) - gaussian(image, sigma=3)
    result[:, :, length - 1] = dog
    logger.debug("Done computing texton descriptors")
    return result
