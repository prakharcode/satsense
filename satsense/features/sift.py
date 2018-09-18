"""Sift feature."""
from typing import Iterator

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from .. import SatelliteImage
from .feature import Feature

SIFT = cv2.xfeatures2d.SIFT_create()


def sift_cluster(sat_images: Iterator[SatelliteImage],
                 n_clusters=32,
                 sample_size=100000) -> MiniBatchKMeans:
    """Create the clusters needed to compute the sift feature."""
    sift = cv2.xfeatures2d.SIFT_create()
    descriptors = None
    for sat_image in sat_images:
        _, new_descriptors = sift.detectAndCompute(sat_image.gray_ubyte, None)
        del _  # Free up memory

        # Add descriptors if we already had some
        if descriptors is None:
            descriptors = new_descriptors
        else:
            descriptors = np.append(descriptors, new_descriptors, axis=0)

    if descriptors.shape[0] > sample_size:
        # Limit the number of descriptors to sample_size
        # by randomly selecting some rows
        descriptors = descriptors[np.random.choice(
            descriptors.shape[0], sample_size, replace=False), :]

    # Cluster the descriptors
    mbkmeans = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=42).fit(descriptors)

    return mbkmeans


class Sift(Feature):
    """Sift feature."""

    def __init__(self,
                 kmeans: MiniBatchKMeans,
                 windows=((25, 25), ),
                 normalized=True):
        """Create sift feature."""
        super().__init__(windows=windows)
        self.kmeans = kmeans
        self.feature_size = len(self.windows) * kmeans.n_clusters
        self.normalized = normalized
        self.base_image = 'gray_ubyte'

    def __call__(self, cell):
        result = np.zeros(self.feature_size)
        n_clusters = self.kmeans.n_clusters
        for i, window in enumerate(self.windows):
            win = cell.super_cell(window, padding=True)
            start_index = i * n_clusters
            end_index = (i + 1) * n_clusters
            result[start_index:end_index] = self.sift(win.gray_ubyte,
                                                      self.kmeans)
        return result

    def __str__(self):
        normalized = "normalized" if self.normalized else "not-normalized"
        return "Sift-{}-{}".format(str(self.windows), normalized)

    def sift(self, window_gray_ubyte, kmeans: MiniBatchKMeans):
        """Calculate the sift feature on the given window."""
        _, descriptors = SIFT.detectAndCompute(window_gray_ubyte, None)
        del _  # Free up memory

        # Is none if no descriptors are found, i.e. on 0 input range
        cluster_count = kmeans.n_clusters
        if descriptors is None:
            return np.zeros((cluster_count))

        codewords = kmeans.predict(descriptors)
        counts = np.bincount(codewords, minlength=cluster_count)

        # Perform normalization
        if self.normalized:
            counts = counts / cluster_count

        return counts
