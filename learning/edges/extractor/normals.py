# vim: set tabstop=4 shiftwidth=4 textwidth=79 cc=72,79:
"""
    feature_extractor: The functions related to feature extraction that
    extract_features.py uses.
    Original Author: Owain Jones [odj@aber.ac.uk]
"""

from __future__ import print_function
import numpy.lib.nanfunctions as nm  # nanfunctions ignore NaN values.
from utils import *
import numpy as np
import traceback


def _get_features(image_bgr, image_hsv, bgr_means, hsv_means, normal,
                  ground_truth, scale=1, ignore_objects=False,
                  extra_args=None):
    """Get statistical features for a single normal.

       This is where all the grunt work happens. Big function."""
    # OpenCV deals with arrays that go y, x instead of x, y
    # But my functions all assume x, y points, so flip the normal around
    # for use with get_point_class and other stuff down below.
    x1, y1 = normal[0]
    y1 = np.clip(y1, 0, image_bgr.shape[0] - 1)
    x1 = np.clip(x1, 0, image_bgr.shape[1] - 1)
    x2, y2 = normal[1]
    y2 = np.clip(y2, 0, image_bgr.shape[0] - 1)
    x2 = np.clip(x2, 0, image_bgr.shape[1] - 1)
    centerx = x1 + ((x2 - x1) / 2.0)
    centery = y1 + ((y2 - y1) / 2.0)
    normal_flipped = ((y1, x1), (y2, x2))

    # Check what this normal is classed as in the ground truth.
    features = {
        'label_class': get_point_class(normal_flipped, ground_truth)
    }
    if ignore_objects and features['label_class'] == 'object':
        features['label_class'] = 'background'

    features['_pos_centerx'] = centerx
    features['_pos_centery'] = centery
    features['img_scale'] = scale
    if extra_args is not None:
        for key, val in extra_args.items():
            features[key] = val

    # Find the line segments that make up either side of the center.
    # Get the HSV and BGR channel values for each (results in a list of
    # 3-element tuples).
    line1 = ((x1, y1), (centerx, centery))
    line2 = ((centerx, centery), (x2, y2))
    edge1_bgr = get_values(image_bgr, line1)
    edge2_bgr = get_values(image_bgr, line2)
    edge1_hsv = get_values(image_hsv, line1)
    edge2_hsv = get_values(image_hsv, line2)
    both_bgr = get_values(image_bgr, normal)
    both_hsv = get_values(image_hsv, normal)

    # Make some internal functions for getting/setting stuff in arrays / the
    # features dict. Makes it easier to change stuff later on if the dictionary
    # labelling changes.
    def _values(array, channel):
        return [x[CHANNELS[channel]] for x in array]

    def _feature(edge, channel, label, feature):
        features[edge + '_' + channel + '_' + label] = feature

    def _olbp(array, channel):
        return olbp(array[:, :, CHANNELS[channel]], (centerx, centery))

    # Get the mean and standard deviation values for either side of the
    # normal, for the hue, saturation and value channels. As well as for
    # the normal as a whole.
    for channel in ['hue', 'sat', 'val']:
        # _feature('edge1', channel, 'stdev',
        #          nm.nanstd(_values(edge1_hsv, channel)))
        _feature('edge1', channel, 'mean',
                 nm.nanmean(_values(edge1_hsv, channel)))
        # _feature('edge2', channel, 'stdev',
        #          nm.nanstd(_values(edge2_hsv, channel)))
        _feature('edge2', channel, 'mean',
                 nm.nanmean(_values(edge2_hsv, channel)))
        # _feature('both', channel, 'stdev',
        #          nm.nanstd(_values(both_hsv, channel)))
        # _feature('both', channel, 'mean',
        #          nm.nanmean(_values(both_hsv, channel)))
        # _feature('both', channel, 'diff', abs(
        #     features['edge1_' + channel + '_mean'] -
        #     features['edge2_' + channel + '_mean']
        # ))
        # _feature('both', channel, 'image_diff', abs(
        #     features['both_' + channel + '_mean'] -
        #     hsv_means[channel]
        # ))
        # _feature('center', channel, 'lbp', _olbp(image_hsv, channel))
        _feature('edge1', channel, 'min',
                 nm.nanmin(_values(edge1_hsv, channel)))
        _feature('edge1', channel, 'max',
                 nm.nanmax(_values(edge1_hsv, channel)))
        _feature('both', channel, 'ratio', (
            nm.nanmin((
                features['edge1_' + channel + '_mean'],
                features['edge2_' + channel + '_mean']
            )) / nm.nanmax((
                features['edge1_' + channel + '_mean'],
                features['edge2_' + channel + '_mean']
            ))
        ))

    # Get value/saturation ratios for both sides as well as the whole
    # normal. Catch possible ZeroDivisonErrors. If one of the attributes is 0,
    # make the answer 0.
    # for edge in ['edge1', 'edge2', 'both']:
    #     if features[edge + '_val_mean'] > 0\
    #     and features[edge + '_sat_mean'] > 0:
    #         features[edge + '_val_sat_ratio'] =\
    #             features[edge + '_val_mean'] / features[edge + '_sat_mean']
    #     else:
    #         features[edge + '_val_sat_ratio'] = 0.0

    # Do the same things as above but for the Red, Green, Blue channels
    # now.
    rgb_channels = set(sorted(['red', 'green', 'blue']))
    for channel in rgb_channels:
        # _feature('edge1', channel, 'stdev',
        #          nm.nanstd(_values(edge1_bgr, channel)))
        _feature('edge1', channel, 'mean',
                 nm.nanmean(_values(edge1_bgr, channel)))
        # _feature('edge2', channel, 'stdev',
        #          nm.nanstd(_values(edge2_bgr, channel)))
        _feature('edge2', channel, 'mean',
                 nm.nanmean(_values(edge2_bgr, channel)))
        # _feature('both', channel, 'stdev',
        #          nm.nanstd(_values(both_bgr, channel)))
        # _feature('both', channel, 'mean',
        #          nm.nanmean(_values(both_bgr, channel)))
        # _feature('both', channel, 'diff', abs(
        #     features['edge1_' + channel + '_mean'] -
        #     features['edge2_' + channel + '_mean']
        # ))
        # _feature('both', channel, 'image_diff', abs(
        #     features['both_' + channel + '_mean'] -
        #     bgr_means[channel]
        # ))
        # _feature('both', channel, 'lbp', _olbp(image_bgr, channel))
        _feature('edge1', channel, 'min',
                 nm.nanmin(_values(edge1_bgr, channel)))
        _feature('edge1', channel, 'max',
                 nm.nanmax(_values(edge1_bgr, channel)))
        _feature('both', channel, 'ratio', (
            nm.nanmin(
                (features['edge1_' + channel + '_mean'],
                 features['edge2_' + channel + '_mean'])
            ) / nm.nanmax(
                (features['edge1_' + channel + '_mean'],
                 features['edge2_' + channel + '_mean'])
            )
        ))

    # Get channel ratios for each of the BGR channels. Try not to duplicate
    # stuff (don't do red/blue then blue/red) to reduce the number of features.
    # for channel, other_channel in [
    #     ('red', 'green'), ('red', 'blue'), ('green', 'blue')
    # ]:
    #     for edge in ['edge1', 'edge2', 'both']:
    #         if features[edge + '_' + channel + '_mean'] > 0\
    #         and features[edge + '_' + other_channel + '_mean'] > 0:
    #             features[edge + '_' + channel + '_' + other_channel + '_ratio'] =\
    #                 features[edge + '_' + channel + '_mean'] /\
    #                 features[edge + '_' + other_channel + '_mean']
    #         # Catch zero divisions, set them to 0.
    #         else:
    #             features[edge + '_' + channel + '_' + other_channel + '_ratio'] =\
    #                 0.0

    return features


def get_features(bgr, hsv, normals, grounds, length=LENGTH,
                 debug=False, scale=1, ignore_objects=False, extra_args=None):
    """Gets statistical features for a collection of contour normals.

       Returns: A dictionary containing all the features calculated by
       this function.

       The dictionary is large and could contain >60 different keys!

       The dictionary can be considered a table of training instances.
       The data is read column-first (loop over each key), with each row
       being an instance.
    """
    bgr_means = {}
    for channel in ['blue', 'green', 'red']:
        bgr_means[channel] = nm.nanmean(bgr[:, :, CHANNELS[channel]])
    hsv_means = {}
    for channel in ['hue', 'sat', 'val']:
        hsv_means[channel] = nm.nanmean(hsv[:, :, CHANNELS[channel]])

    features = {}
    num_contours = len(normals)
    for c_idx, contour in enumerate(normals):
        for n_idx, normal in enumerate(contour):
            try:
                local_features = _get_features(bgr, hsv, bgr_means, hsv_means,
                                               normal, grounds, scale,
                                               ignore_objects, extra_args)

                for key in local_features.keys():
                    f = [local_features[key]]
                    if key not in features.keys():
                        features[key] = f
                    else:
                        features[key].extend(f)

            except:
                traceback.print_exc()

        if debug:
            e_str = '\r'
            if c_idx >= num_contours - 1:
                e_str = '\n'
            d_str = '    Contour %d/%d' % (c_idx, num_contours)
            w_str = ' ' * (79 - len(d_str))
            print(d_str, w_str, end=e_str)

    return features
