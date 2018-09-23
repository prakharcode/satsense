import rasterio
import logging
import numpy as np
from rasterio import windows
from .image import Image, remap

class DiskImage(Image):
    """An image that is not loaded into memory, but read using windows from disk"""
    def __init__(self, dataset, image, satellite, name='', compute=False):
        super().__init__(image, bands=satellite, compute=False)
        # We set the image here, because in normal Image it is not stored to save memory space
        # In a DiskImage this either a 1x1 pixel for checking the bands
        # Or a window that has just been loaded from disk
        self.image = image

        self.name = name
        self.dataset = dataset
        self.crs = dataset.crs
        self.transform = dataset.transform
        self.bounds = dataset.bounds
        self.res = dataset.res

        self.logger = logging.getLogger(self.__class__.__name__)

        if compute:
            self.calculate_normalization_factors(self.bands, **self._normalization_parameters)

    @classmethod
    def load_from_file(cls, path, satellite):
        """Load the specified path and bands from file into a numpy array."""
        dataset = rasterio.open(path)

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

        return cls(dataset, image, satellite, name=path, compute=True)
    
    
    def calculate_normalization_factors(self, bands, technique='cumulative', percentiles=(2.0, 98.0), numstds=2):
        """Calculate the normalization factors on the image based on the band maximum.

        This function does not load the entire image at once (if possible), but retains the necesarry number of values
        to calculate the given percentiles over multiple windows
        """
        self.logger.debug("Computing normalization factors for image")
        if technique == 'cumulative':
            self.min_max = self.calculate_cumulative_normalization_factors(self.dataset, bands, percentiles)
        elif technique == 'meanstd':
            pass
        else:
            pass

    def calculate_cumulative_normalization_factors(self, dataset, bands, percentiles):
        if len(percentiles) != 2:
            raise ValueError("Only support normalization using 2 percentiles")

        # total number of array elements per band
        items = np.prod(dataset.shape)
        # The nth item using the supplied percentiles
        nth = items * np.array(percentiles) / 100.0
        half = items / 2.0
        # Zomgwtfbbq list comprehension
        # This calculates the number of items to remember during the calculation
        buffer_sizes = [int(np.ceil(x)) if x <= half else int(np.ceil(items-x)) for x in nth]

        # TODO: if the sum of the buffers is larger than
        # the total size of the image, it makes more sense to
        # use the old method
        # whole_image = False
        # if np.sum(buffer_sizes) > items:
        #    whole_image = True

        # Create 2 buffers of the right size for each band
        buffers = {}
        for key, band in bands.items():
            buffers[key] = {i:{'array': np.array([]), 'len': 0} for i,x in enumerate(buffer_sizes)}

        # Arbitrary small window size
        window_size = (400, 400)
        x_tiles = int(np.ceil(dataset.shape[0] / window_size[0]))
        y_tiles = int(np.ceil(dataset.shape[1] / window_size[1]))
        for x_tile in range(x_tiles):
            for y_tile in range(y_tiles):
                self.logger.debug("x, y: ({}, {})".format(x_tile, y_tile))
                image = get_tile(self.dataset, x_tile, y_tile, *window_size)

                for key, band in bands.items():
                    if np.ma.isMaskedArray(image):
                        # select only non-masked values for computing scale
                        selection = image[:, :, band].compressed()
                    else:
                        selection = image[:, :, band].flatten()
                        selection = selection[~np.isnan(selection)]

                    # Use selection to only take those values that are smaller than the largest value in the buffer
                    # Insert all the new values
                    buffers[key][0]['array'] = np.concatenate((buffers[key][0]['array'], selection))
                    buffers[key][1]['array'] = np.concatenate((buffers[key][1]['array'], selection))

                    # Store the actual lenghts of the arrays
                    buffers[key][0]['len'] += len(selection)
                    buffers[key][1]['len'] += len(selection)

                    # Sort the arrays
                    buffers[key][0]['array'].sort()
                    # Sort in reverse order
                    buffers[key][1]['array'][::-1].sort()

                    # Only keep those values that we're interested in
                    buffers[key][0]['array'] = buffers[key][0]['array'][0:buffer_sizes[0]]
                    buffers[key][1]['array'] = buffers[key][1]['array'][0:buffer_sizes[1]]


        # After calculation the last value
        # holds the low and high percentile
        min_max = {}
        for key, band in bands.items():
            min_max[key] = {}

            # Recalculate the nth percentile using the actual number of items
            items = buffers[key][0]['len']
            nth = items * np.array(percentiles) / 100.0
            half = items / 2.0
            locs  = [int(np.ceil(x)) if x <= half else int(np.ceil(items-x)) for x in nth]

            min_max[key][0] = buffers[key][0]['array'][locs[0]-1]
            min_max[key][1] = buffers[key][1]['array'][locs[0]-1]
        
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
    def __init__(self, image, x, y, x_range, y_range, orig=None):
        super().__init__(orig.dataset, image, orig.bands, name=orig.name, compute=False)
        self.orig = orig
        self.min_max = orig.min_max
        self.x = x
        self.y = y
        self.x_range = x_range
        self.y_range = y_range

    def super_cell(self, size, padding=True):
        x_offset = (size[0] / 2.0)
        y_offset = (size[1] / 2.0)

        x_middle = (self.x_range.stop + self.x_range.start) / 2.0
        y_middle = (self.y_range.stop + self.y_range.start) / 2.0

        x_start = np.floor(x_middle - x_offset)
        x_end = np.floor(x_middle + x_offset)

        y_start = np.floor(y_middle - y_offset)
        y_end = np.floor(y_middle + y_offset)

        y_pad_before = 0
        y_pad_after = 0
        x_pad_before = 0
        x_pad_after = 0
        pad_needed = False
        if x_start < 0:
            pad_needed = True
            x_pad_before = -x_start
            x_start = 0
        if x_end > self.dataset.shape[0]:
            pad_needed = True
            x_pad_after = x_end - self.dataset.shape[0]
            x_end = self.dataset.shape[0]
        if y_start < 0:
            pad_needed = True
            y_pad_before = -y_start
            y_start = 0
        if y_end > self.dataset.shape[1]:
            pad_needed = True
            y_pad_after = y_end - self.dataset.shape[1]
            y_end = self.dataset.shape[1]

        x_range = slice(x_start, x_end)
        y_range = slice(y_start, y_end)

        window = windows.Window.from_slices(x_range, y_range)
        img = get_window(self.dataset, window)
        #img = self.image.shallow_copy_range(x_range, y_range)
        if padding and pad_needed:
            img = pad(img, x_pad_before, x_pad_after, y_pad_before, y_pad_after)

        return DiskImageCell(img, self.x, self.y, x_range, y_range, orig=self.dataset)
    
    @property
    def normalized(self):
        if 'normalized' not in self._images:
            result = self.image.copy()
            for key, band in self.bands.items():
                new_min = self.orig.min_max[key][0]
                new_max = self.orig.min_max[key][1]

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
