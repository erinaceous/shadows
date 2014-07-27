#!/usr/bin/env python
# vim: set tabstop=4 shiftwidth=4 textwidth=79 cc=72,79:
"""
    lalonde_extractor: Extract edge features using the methods outlined
    in their Ground Shadow paper.

    Original Author: Owain Jones [odj@aber.ac.uk]
"""

from __future__ import print_function
import numpy.lib.nanfunctions as nm
import numpy as np


def _get_feature(bgr, hsv, pixel, ground_truths, angle):
    """Generate the feature set for an individual pixel."""
    y, x = (
        np.clamp(pixel[0], 0, bgr.shape[0]),
        np.clamp(pixel[1], 0, bgr.shape[1])
    )
    variance = [1, 2, 4, 8]

    # Generate a Gaussian derivative kernel with the given angle.



def get_features(bgr, hsv, contours, ground_truths, length=0, debug=True,
                 scale=1.0, extra_args=None, ignore_objects=False):
    """Loop over each pixel in a contour getting the overcomplete
       >40-feature set of features, and then save the mean averages
       of those features as the edge feature for each edge (contour).

       Returns: A dictionary containing lists of values for each
       feature.
    """

    features = {}

    return features
