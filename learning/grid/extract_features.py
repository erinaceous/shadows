#!/usr/bin/env python2
# vim: set tabstop=4 shiftwidth=4 textwidth=79 cc=72,79:
"""
    extract_features: Extract features from an image, using adjustable
    sized square regions.

    Original Author: Owain Jones [odj@aber.ac.uk]
"""

from __future__ import print_function
import numpy as np
import argparse
import cv2


labels = {
    0: 'shadow',
    25: 'penumbra',
    153: 'object',
    255: 'background'
}


channels = {
    'hue': 0,
    'sat': 1,
    'val': 2,
    'red': 2,
    'green': 1,
    'blue': 0
}


def grid_ground_truth(ground, n):
    """Converts a ground truth image (greyscale) into a mosaic image,
       with NxN square cells. The value with the highest count per cell
       is the value used in the output.
    """
    new_ground = ground.copy()
    new_ground.fill(255)
    for i in range(0, ground.shape[0], n):
        for j in range(0, ground.shape[1], n):
            square = ground[i:i + n, j:j + n]
            hist = cv2.calcHist([square], [0], None, [256], [0, 256])
            # print(hist)
            new_ground[i:i + n, j:j + n] = np.argmax(hist)

    return new_ground


def cell_features(cell_bgr, cell_hsv, cell_ground, image_means):
    """Generates features for this image cell."""
    ground_truth_hist = cv2.calcHist([cell_ground], [0], None, [256], [0, 256])
    ground_truth_val = np.argmax(ground_truth_hist)

    features = {'label': labels[ground_truth_val]}

    for channel in ['hue', 'sat', 'val']:
        features[channel + '_mean'] = np.mean(cell_hsv[channels[channel]])
        features[channel + '_stdev'] = np.std(cell_hsv[channels[channel]])
        features[channel + '_diff'] =\
            abs(features[channel + '_mean'] - image_means[channel])

    for channel in ['red', 'green', 'blue']:
        features[channel + '_mean'] = np.mean(cell_bgr[channels[channel]])
        features[channel + '_stdev'] = np.std(cell_bgr[channels[channel]])
        features[channel + '_diff'] =\
            abs(features[channel + '_mean'] - image_means[channel])

    return features


def grid_image_features(image, ground_truth, n):
    """Loops over an image, turning it into a grid, extracting features for
       cells.

       Returns a dict, where every key is the column of a table (type of
       feature). So to generate a table, loop over it value-first.
    """

    features = {}

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_bgr = image.copy()

    image_means = {}
    for channel in ['hue', 'sat', 'val']:
        image_means[channel] = np.mean(image_hsv[:, :, channels[channel]])
    for channel in ['red', 'green', 'blue']:
        image_means[channel] = np.mean(image_bgr[:, :, channels[channel]])

    for i in range(0, image.shape[0], n):
        for j in range(0, image.shape[1], n):
            curr_features = cell_features(
                image_bgr[i:i + n, j:j + n], image_hsv[i:i + n, j:j + n],
                ground_truth[i: i + n, j:j + n], image_means
            )
            for key in curr_features:
                if key not in features:
                    features[key] = [curr_features[key]]
                else:
                    features[key].append(curr_features[key])

    return features


def _str(value):
    if isinstance(value, float):
        return str(round(value, 3))
    return str(value)


def to_csv(csvfile, features, header=False):
    if header is True:
        csv = open(csvfile, 'w+')
        print(','.join(features.keys()), file=csv)
    else:
        csv = open(csvfile, 'a+')
    for i in range(0, len(features.values()[0])):
        print(','.join([_str(features[k][i]) for k in features.keys()]),
              file=csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-i', '--images', nargs='+')
    parser.add_argument('-g', '--ground-truths', nargs='+')
    parser.add_argument('-n', '--cell-size', type=int, default=10)
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    images = sorted(args.images)
    ground_truths = sorted(args.ground_truths)

    for i, image_path in enumerate(images):
        print('Getting features for', image_path)
        image = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_COLOR)
        ground_truth = cv2.imread(ground_truths[i],
                                  cv2.CV_LOAD_IMAGE_GRAYSCALE)
        features = grid_image_features(image, ground_truth, args.cell_size)
        to_csv(args.output, features, (i == 0))
