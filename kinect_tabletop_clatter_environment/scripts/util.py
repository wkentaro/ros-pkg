#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.ndimage as ndi
from skimage.segmentation import slic


def masked_slic(img, mask, n_segments=1000, compactness=10):
    labels = slic(img, n_segments=n_segments, compactness=compactness)
    labels += 1
    labels[mask == 0] = 0  # set bg_label
    return labels


def closed_mask_roi(mask):
    closed_mask = ndi.binary_closing(mask,
                                     structure=np.ones((3, 3)), iterations=8)
    roi = ndi.find_objects(closed_mask, max_label=1)[0]
    return roi
