import rasterio
import logging
import numpy as np
from rasterio import windows
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from os import cpu_count
from .image import Image, remap

logger = logging.getLogger('disk_image')


class DiskImage(Image):
    """An image that is not loaded into memory, but read using windows from disk"""
    def __init__(self, dataset, path, image, satellite, name='', compute=False):
        super().__init__(image, bands=satellite, compute=False)

        # We set the image here, because in normal Image it is not stored to save memory space
        # In a DiskImage this either a 1x1 pixel for checking the bands
        # Or a window that has just been loaded from disk
        self.image = image

        self.name = name
        self.path = path
        self.crs = dataset.crs
        self.transform = dataset.transform
        self.bounds = dataset.bounds
        self.res = dataset.res

        if compute:
            self.calculate_normalization_factors(self.bands, **self._normalization_parameters)

    @classmethod
    def load_from_file(cls, path, satellite):
        """Load the specified path and bands from file into a numpy array."""
        with rasterio.open(path) as dataset:

            # load the first pixel to determine number of bands
            window = windows.Window(0, 0, 1, 1)
            image = dataset.read(masked=False, window=window)

            if len(image.shape) == 3:
                # The bands column is in the first position, but we want it last
                image = np.rollaxis(image, 0, 3)
            elif len(image.shape) == 2:
                # This image seems to have one band, so we add an axis for ease
                # of use in the rest of the library
                image = image[:, :, np.newaxis]

            return cls(dataset, path, image, satellite, name=path, compute=True)
    
    
    def calculate_normalization_factors(self, bands, technique='cumulative', percentiles=(2.0, 98.0), numstds=2):
        """Calculate the normalization factors on the image based on the band maximum.

        This function does not load the entire image at once (if possible), but retains the necesarry number of values
        to calculate the given percentiles over multiple windows
        """
        logger.debug("Computing normalization factors for image")
        if technique == 'cumulative':
            with rasterio.open(self.path) as dataset:
                self.min_max = self.calculate_cumulative_normalization_factors(dataset, bands, percentiles)
        elif technique == 'meanstd':
            pass
        else:
            pass

    def calculate_cumulative_normalization_factors(self, dataset, bands, percentiles):
        if len(percentiles) != 2:
            raise ValueError("Only support normalization using 2 percentiles")

        min_max = {}
        for key, band in bands.items():
            logging.info("Normalizing band %s (%i)", key, band)
            # select only non-masked values for computing scale
            # rasterio bands in read is indexed from 1
            selection = dataset.read(band+1, masked=True).compressed()

            # there should be no nans in a masked dataset
            percents = np.percentile(selection, percentiles)

            min_max[key] = {
                'min': percents[0],
                'max': percents[1]
            }

        logging.info("normalization values: %s", min_max)       
        return min_max

    # self.new_min = {}
    # self.new_max = {}
    # 
    #     percents = np.nanpercentile(selection, percentiles)
    #     new_min, new_max = percents
    # elif technique == 'meanstd':
    #     mean = selection.nanmean()
    #     std = selection.nanstd()
    #     new_min = mean - (numstds * std)
    #     new_max = mean + (numstds * std)
    # else:
    #     new_min = selection.nanmin()
    #     new_max = selection.nanmax()
    # self.new_min[key] = new_min
    # self.new_max[key] = new_max


def get_tile(dataset, x_tile, y_tile, width, height):
    x_pos = x_tile * width
    y_pos = y_tile * width

    window = windows.Window(x_pos, y_pos, width, height)
    return get_window(dataset, window)


def get_window(dataset, window):
    image = dataset.read(masked=True, window=window)

    if len(image.shape) == 3:
        # The bands column is in the first position, but we want it last
        image = np.rollaxis(image, 0, 3)
    elif len(image.shape) == 2:
        # This image seems to have one band, so we add an axis for ease
        # of use in the rest of the library
        image = image[:, :, np.newaxis]
    
    return image


class DiskImageCell(DiskImage):
    def __init__(self, dataset, image, x, y, x_range, y_range, orig=None):
        if np.ma.isMaskedArray(image):
            image = np.ma.filled(image, fill_value=0)

        super().__init__(dataset, orig.path, image, orig.bands, name=orig.name, compute=False)
        self.orig = orig
        self.min_max = orig.min_max
        self.x = x
        self.y = y
        self.x_range = x_range
        self.y_range = y_range

    def super_cell(self, size, padding=True):
        with rasterio.open(self.path) as dataset:
            x_offset = (size[0] / 2.0)
            y_offset = (size[1] / 2.0)

            x_middle = (self.x_range.stop + self.x_range.start) / 2.0
            y_middle = (self.y_range.stop + self.y_range.start) / 2.0

            x_start = int(np.floor(x_middle - x_offset))
            x_end = int(np.floor(x_middle + x_offset))

            y_start = int(np.floor(y_middle - y_offset))
            y_end = int(np.floor(y_middle + y_offset))

            y_pad_before = 0
            y_pad_after = 0
            x_pad_before = 0
            x_pad_after = 0
            pad_needed = False
            if x_start < 0:
                pad_needed = True
                x_pad_before = -x_start
                x_start = 0
            if x_end > dataset.shape[0]:
                pad_needed = True
                x_pad_after = x_end - dataset.shape[0]
                x_end = dataset.shape[0]
            if y_start < 0:
                pad_needed = True
                y_pad_before = -y_start
                y_start = 0
            if y_end > dataset.shape[1]:
                pad_needed = True
                y_pad_after = y_end - dataset.shape[1]
                y_end = dataset.shape[1]

            x_range = slice(x_start, x_end)
            y_range = slice(y_start, y_end)

            window = windows.Window.from_slices(x_range, y_range)
            img = get_window(dataset, window)
            if np.ma.isMaskedArray(img):
                img = np.ma.filled(img, fill_value=0)

            if padding and pad_needed:
                img = pad(img, x_pad_before, x_pad_after, y_pad_before, y_pad_after)

            return DiskImageCell(dataset, img, self.x, self.y, x_range, y_range, orig=self.orig)
    
    @property
    def normalized(self):
        if 'normalized' not in self._images:
            result = self.image.copy()
            for key, band in self.bands.items():
                new_min = self.orig.min_max[key]['min']
                new_max = self.orig.min_max[key]['max']

                result[:, :, band] = remap(result[:, :, band], new_min, new_max, 0, 1)

                np.clip(result[:, :, band], a_min=0, a_max=1, out=result[:, :, band])
                self._images['normalized'] = result
        return self._images['normalized']

def pad(image, x_pad_before: int, x_pad_after: int, y_pad_before: int, y_pad_after: int):
    """Pad the image."""
    pad_width = (
        (x_pad_before, x_pad_after),
        (y_pad_before, y_pad_after),
    )
    if len(image.shape) == 3:
        pad_width += ((0, 0), )

    return np.pad(image, pad_width, 'constant', constant_values=0)

class DiskCellGenerator:
    def __init__(self,
                 disk_image: DiskImage,
                 size: tuple,
                 offset=(None, None),
                 length=(None, None)):

        self.image = disk_image

        self.x_size, self.y_size = size
        with rasterio.open(self.image.path) as dataset:
            self.x_length = int(np.ceil(dataset.shape[0] / self.x_size))
            self.y_length = int(np.ceil(dataset.shape[1] / self.y_size))

        offset = list(offset)
        if offset[0] is None:
            offset[0] = 0
        if offset[0] < 0:
            offset[0] = self.x_length - offset[0]
        if offset[0] > self.x_length:
            raise IndexError("x offset {} larger than image size {}".format(
                offset[0], self.x_length))
        self.x_offset = offset[0]

        if offset[1] is None:
            offset[1] = 0
        if offset[1] < 0:
            offset[1] = self.y_length - offset[1]
        if offset[1] > self.y_length:
            raise IndexError("y offset {} larger than image size {}".format(
                offset[1], self.y_length))
        self.y_offset = offset[1]

        length = list(length)
        if length[0] is None:
            length[0] = self.x_length
        if length[0] < 0:
            length[0] = self.x_length - length[0]
        if length[0] > self.x_length:
            raise IndexError("x length {} larger than image size {}".format(
                length[0], self.x_length))
        self.x_length = length[0]

        if length[1] is None:
            length[1] = self.y_length
        if length[1] < 0:
            length[1] = self.y_length - length[1]
        if length[1] > self.y_length:
            raise IndexError("y length {} larger than image size {}".format(
                length[1], self.y_length))
        self.y_length = length[1]

    def __len__(self):
        return self.x_length * self.y_length

    @property
    def shape(self):
        return (self.x_length, self.y_length)

    def __iter__(self):
        for x in range(self.x_length):
            for y in range(self.y_length):
                yield self[x, y]

    def __getitem__(self, index):
        x, y = index
        x = self.x_length - x if x < 0 else x
        y = self.y_length - y if y < 0 else y

        if x >= self.x_length or y >= self.y_length:
            raise IndexError('{} out of range for image of shape {}'.format(
                index, self.shape))

        x += self.x_offset
        y += self.y_offset

        x_start = x * self.x_size
        x_range = slice(x_start, x_start + self.x_size)

        y_start = y * self.y_size
        y_range = slice(y_start, y_start + self.y_size)

        window = windows.Window.from_slices(x_range, y_range)
        with rasterio.open(self.image.path) as dataset:
            im = get_window(dataset, window)

            return DiskImageCell(dataset, im, x, y, x_range, y_range, orig=self.image)


def extract_features_diskimage(features, generator):
    """Compute features."""
    shape = (generator.shape[0], generator.shape[1], features.index_size)
    feature_vector = np.zeros(
        (shape[0] * shape[1], shape[2]), dtype=np.float32)
    logger.debug("Feature vector shape %s", shape)

    size = len(generator)
    for i, cell in enumerate(generator):
        if i % (size // 10 or 1) == 0:
            logger.debug("%s%% ready", 100 * i // size)
        for feature in features.items.values():
            feature_vector[i, feature.indices] = feature(cell)
    feature_vector.shape = shape

    return feature_vector

def extract_single_diskimage(features, shape, size, cell):
    """Compute features."""
    i = cell.x * shape[0] + cell.y
    if cell.y == 0:
        logger.info("%s%% ready", 100 * i // size)
    feature_vector = np.zeros((1, 1, features.index_size), dtype=np.float32)
    for feature in features.items.values():
        feature_vector[feature.indices] = feature(cell)
    return cell, feature_vector

def extract_features_parallel_diskimage(features, generator, n_jobs=cpu_count()):
    """Extract features in parallel."""
    logger.debug("Extracting features using at most %s processes", n_jobs)

    shape = (generator.shape[0], generator.shape[1], features.index_size)
    feature_vector = np.zeros(
        (shape[0], shape[1], shape[2]), dtype=np.float32)

    # Execute jobs
    extracted_features = []
    with ProcessPoolExecutor() as executor:
        extract = partial(extract_single_diskimage, features, generator.shape, len(generator))
        extracted_features = executor.map(extract, generator, chunksize=generator.shape[0])
    
    for cell, vector in extracted_features:
        feature_vector[cell.x, cell.y, :] = vector

    logger.debug("Done extracting features. Feature vector shape %s",
                 feature_vector.shape)
    return feature_vector
