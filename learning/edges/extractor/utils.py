#!/usr/bin/env python
# vim: set tabstop=4 shiftwidth=4 textwidth=79 cc=72,79:
"""
    feature_utils: Utilities relating to extract_features.py
    Original Author: Owain Jones [odj@aber.ac.uk]
"""

from __future__ import print_function
import numpy as np
import random
import gzip
import math
import cv2


# Configurable defaults
NORMAL_LENGTH = 30
CLASSES = ['shadow', 'penumbra', 'object']
PN_RATIO = 3  # positive/negative example ratio to enforce
DISTANCE = 3  # minimum pixel distance between contour points
LENGTH = 30   # default length of normals


# Possible names for channels within the HSV/BGR colour spaces.
# mat[:, :, CHANNELS['r']] would get you mat[:, :, 2] -- a 2D array of
# the red channel from the BGR image, mat.
CHANNELS = {
    'h': 0,
    'hue': 0,
    's': 1,
    'sat': 1,
    'saturation': 1,
    'v': 2,
    'val': 2,
    'value': 2,
    'r': 2,
    'red': 2,
    'g': 1,
    'green': 1,
    'b': 0,
    'blue': 0
}


def load_image(path, blur=0, bifilter=0):
    """Loads an image at `path` and also converts it to the HSV colour
       space. Also optionally performs Gaussian blur and Bilateral
       filtering on the input image.

       Returns: two image matrices; one for the original BGR image, and
       one for the image converted to the HSV colour space.
    """
    image = cv2.imread(path, cv2.CV_LOAD_IMAGE_COLOR)
    if bifilter > 0:
        image = cv2.bilateralFilter(image, -1, bifilter, bifilter)
    if blur > 0:
        image = cv2.GaussianBlur(image, (blur, blur), 0)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return (image, image_hsv)


def load_ground_truth(path, penumbra_as_shadow=False):
    """Loads a ground truth image (as a 1-channel grayscale image) and
       extracts the shadow, penumbra and object masks from it.

       Returns: A tuple of 3 image matrices:
           (shadow, penumbra, object)
    """
    image = cv2.imread(path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    shadows = image.copy()
    if penumbra_as_shadow:
        shadows[shadows == 25] = 0
    else:
        shadows[shadows == 25] = 255
    shadows[shadows == 153] = 255
    shadows[shadows < 255] = 0
    penumbra = image.copy()
    if penumbra_as_shadow:
        penumbra.fill(255)
    else:
        penumbra[penumbra != 25] = 255
        penumbra[penumbra == 25] = 0
    objects = image.copy()
    objects[objects != 153] = 255
    objects[objects == 153] = 0
    return (shadows, penumbra, objects)


def get_normal(line, length=0):
    """Gets the normal of a line (flips it 90 degrees).
       Line is a pair of tuples -- ((x, y), (x, y)) -- representing the
       two ends of the line.

       If length is set to a value > 0, the normal is also set to that
       fixed length.
    """
    if type(line) not in [list, tuple]:
        line = (
            line[0][0], line[1][0]        
        )

    if length <= 0:
        return (
            (line[1][0], line[0][1]),
            (line[0][0], line[1][1])
        )
    half_length = length / 2
    center = (
        line[0][0] + ((line[1][0] - line[0][0]) / 2),
        line[0][1] + ((line[1][1] - line[0][1]) / 2)
    )
    diff = (
        line[1][0] - line[0][0],
        line[1][1] - line[0][1]
    )
    angle = math.degrees(math.atan2(diff[0], diff[1]))
    return (
        (center[0] - half_length * math.sin(angle),
         center[1] - half_length * math.cos(angle)),
        (center[0] + half_length * math.sin(angle),
         center[1] + half_length * math.cos(angle))
    )


def round_partial(value, resolution):
    """Rounds a number to a partial interval. Good for rounding things
       up/down to the nearest value of 5 for example.

       Thanks to http://stackoverflow.com/a/8118808 for this neat trick
    """
    return round(value / resolution) * resolution


def anyrange(start, stop, interval, **kwargs):
    """Smarter version of range() -- if the start number > end number,
       inverts interval (subtracts from stop until start is reached).
    """
    if start > stop:
        return np.arange(start, stop, -interval, **kwargs)
    return np.arange(start, stop, interval, **kwargs)


def quantize_contour(contour, bin_size=4):
    """Reduces the resolution (number of points) of a contour by
       treating the image as a grid with square cells of `bin_size`
       pixels width and height. Any point within the square has its
       coordinates set to the center of the square, and only one point
       is stored per square.
       If a line in the contour spans more than one square, a point is
       added for each square it touches.
       This is a simple way of reducing the number of segments in a
       contour, and allows for uniform distribution of normals across
       a contour (to prevent bias when sampling features from a contour)

       Returns: A quantized version of the contour.
    """
    if bin_size <= 1:
        center = bin_size
    else:
        center = bin_size / 2
    bins = []
    idx = 0
    for segment in contour:
        segment = segment.flatten()
        x, y = (
            round_partial(segment[0], center),
            round_partial(segment[1], center)
        )
        if (x, y) not in bins:
            # TODO: Figure out how to get a uniform distribution of
            # points across straight lines (interpolation?)
            # idx = len(bins) - 1
            # if idx > 0:
            #    lastx, lasty = bins[idx - 1]
            #    midx = anyrange(x, lastx, bin_size)
            #    midy = anyrange(y, lasty, bin_size)
            #    print(midx, midy)
            bins.append((x, y))
    return bins


def get_distance(point1, point2):
    """Calculate the distance between two aribtrary points."""
    try:
        d = math.sqrt((abs(point2[0] - point1[0]) ^ 2) +
                      (abs(point2[1] - point1[1]) ^ 2))
    except ValueError:
        d = 0.0
    return d


def minimum_distance_contour(contour, distance=0):
    """Reduce the number of points in a contour by enforcing a minimum
       pixel distance between points.

       Returns: A similar contour (same structure) but with (hopefully)
       less points.
    """
    new_contour = list()
    last_point = contour[0]
    for p_idx, point in enumerate(contour):
        if p_idx == 0:
            continue
        if get_distance(last_point, point) > distance:
            new_contour.append(point)
            last_point = point

    return new_contour


def get_normals(contours, length=40, epsilon=0, array=None):
    """Loops over a list of lists [of lists] of contour points -- like
       the lists generated by cv2.findContours().

       Returns a list of list of lists:
           [contours -> [contour -> [line segment normal (x, y)]]]
    """
    new_contours = list()
    for c_idx, contour in enumerate(contours):
        new_contour = list()
        for s_idx, segment in enumerate(contour):

            # Skip the first point
            if s_idx == 0:
                continue

            # Get the current point and the previous point in the list
            # to make up our line.
            point1 = contour[s_idx - 1]
            point2 = contour[s_idx]
            normal = get_normal((point1, point2), length=length)
            new_contour.append(normal)
        new_contours.append(new_contour)
    return new_contours


def get_values(array, line):
    """Get values across a line in a 2D array.
       Lines can be at any angle. The resulting values will be
       interpolated if the line crosses multiple pixels at each point.
       Lines at the edges of the array are clipped to remain inside the
       array.

       Returns: a 1D array of values.
    """
    x1, y1 = line[0]
    x2, y2 = line[1]
    length = np.hypot(x2 - x1, y2 - y1)
    x = np.clip(np.linspace(x1, y2, length), 0, array.shape[0] - 1)
    y = np.clip(np.linspace(y1, y2, length), 0, array.shape[1] - 1)
    values = []
    for i in range(int(length)):
        values.append(array[x[i], y[i]])
    return np.array(values)



def get_edges(image):
    """Do edge detection on a multi-channel image.

       Returns: An image with the same number of channels as the input
       image, with Canny edge detection done on each channel.
    """
    # output = image.copy()
    # channels = image.shape[2]
    # for channel in range(0, channels):
    #     output[:, :, channel] = cv2.Canny(image[:, :, channel], 0, 1)

    # Take the Lalonde et. al. approach to detecting edges -- get gradient
    # magnitudes, apply watershed on the gradient map, and filter out weak
    # edges using a Canny detector.

    # Smooth out noise! 33 is the magic number
    # new = cv2.bilateralFilter(image, 66, 66, 66)
    new = image.copy()

    Sx = cv2.Sobel(new, cv2.CV_32F, 1, 0, ksize=3)
    Sy = cv2.Sobel(new, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.uint8(cv2.magnitude(Sx, Sy))
    grey = np.int32(cv2.cvtColor(mag, cv2.COLOR_BGR2GRAY))
    cv2.watershed(mag, grey)
    mag_grey = cv2.cvtColor(mag, cv2.COLOR_BGR2GRAY)
    _, mag_edges = cv2.threshold(mag_grey, 0, 255,
                                 cv2.THRESH_OTSU | cv2.THRESH_BINARY)

    # Lalonde paper says they used an upper threshold, empirically derived,
    # of 0.3. Their code had floating point images in the range 0..1, so
    # convert that to 8 bit (0..255)
    mag_edges = cv2.Canny(mag_edges, 0, 255 * 0.3)

    #element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    #mag_edges = cv2.dilate(mag_edges, element)
    #mag_edges = cv2.GaussianBlur(mag_edges, (9, 9), 0)
    #mag_edges[mag_edges < 150] = 0
    #mag_edges[mag_edges >= 150] = 255

    return mag_edges


def flatten_sv_edges(image):
    """Flatten a Hue/Sat/Value edge image into a single channel. Edge
       pixels present in both the saturation and value channels are
       kept.
    """
    output = image[:, :, 0].copy()
    output.fill(0)
    output = image[:, :, 1] & image[:, :, 2]
    return output


def flatten_edges(image):
    """Flatten an image into a single channel. Simply converts it to
       grayscale and thresholds so that anything > 0 == 255.
    """
    output = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output[output > 0] = 255
    return output


def get_contours(edges):
    """Get the contours of the edges in an image. Uses
       cv2.findContours() internally.

       OpenCV's contours lists are a little weird
       (contours -> [contour -> [[segment]]]) -- the segments
       (x, y points) are kept in an array of size 1 which doesn't need
       to be there and isn't an intuitive thing to use. Get odd numpy
       errors.

       So this function also converts the data to the following structure:
       list of contours -> [
            list of contour segments -> [
                segment (x, y point)
            ]
        ]

       Returns: A list of contours.
    """
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    new_contours = []
    for contour in contours:
        new_contour = []
        for segment in contour:
            new_contour.append(segment.flatten())
        new_contours.append(new_contour)
    return new_contours


def olbp(array, point):
    """Perform simple local binary pattern calculation with a fixed
       3x3 neighbourhood. Thanks to:
       http://www.bytefish.de/blog/local_binary_patterns/
       for a really nice explanation of LBP.

       Won't return correct results around the image boundaries.
       Because it's only a 3x3 neighbourhood, probably very susceptible
       to noise.
       TODO: Bigger neighbourhood. Variable size maybe?

       Returns: A single decimal number (the encoded pattern)
    """
    x, y = point

    # Make sure we're within the array bounds.
    if x < 1:
        x = 1
    if x > (array.shape[0] - 2):
        x = array.shape[0] - 2
    if y < 1:
        y = 1
    if y > (array.shape[1] - 2):
        y = array.shape[1] - 2

    center = array[x, y]
    code = 0
    code |= (array[x - 1, y - 1] > center) << 7
    code |= (array[x - 1, y] > center) << 6
    code |= (array[x - 1, y + 1] > center) << 5
    code |= (array[x, y - 1] > center) << 4
    code |= (array[x, y + 1] > center) << 3
    code |= (array[x + 1, y - 1] > center) << 2
    code |= (array[x + 1, y] > center) << 1
    code |= (array[x + 1, y + 1] > center) << 0
    return code


def flip_point(point, array):
    """Flip a point so that X=>Y and Y=>X. Also clip its values so that
       they fit within an array.

       Returns: The same point, but fixed.

       (OpenCV arrays in numpy go row => column, or y => x, which
       confuses me as I'm used to thinking of images in (x, y)
       coordinates.)
    """
    return (
        np.clip(point[1], 0, array.shape[0] - 1),
        np.clip(point[0], 0, array.shape[1] - 1)
    )


def get_point_class(line, ground_truth):
    """Finds the label class of a line in an array, based
       on the ground truth masks.

       Picks the class which has the maximum number of instances across
       the line.

       Returns: A string, indicating the class label.
       (e.g. 'background' or 'shadow')
    """
    x1, y1 = line[0]
    x2, y2 = line[1]
    centerx = x1 + ((x2 - x1) / 2.0)
    centery = y1 + ((y2 - y1) / 2.0)

    labels = {'background': -1, 'shadow': 0, 'penumbra': 0, 'object': 0}
    for i, name in enumerate(['shadow', 'penumbra', 'object']):
        labels['background'] += 1
        if ground_truth[i][x1, y1] == 0:
            labels[name] += 1
            labels['background'] -= 1
        if ground_truth[i][x2, y2] == 0:
            labels[name] += 1
            labels['background'] -= 1
        if ground_truth[i][centerx, centery] == 0:
            labels[name] += 1
            labels['background'] -= 1

    v = labels.values()
    k = labels.keys()
    return k[v.index(max(v))]


def yesno(value):
    """Convert 0/1 or True/False to 'yes'/'no' strings.
       Weka/LibSVM doesn't like labels that are numbers, so this is
       helpful for that.
    """
    if value == 1 or value == True:
        return 'yes'
    return 'no'


def _str(value):
    """Converts numbers to strings, rounding them to 3 decimal places.
       Converts instances of NaN (Not a Number), or infinity, to '0'.
    """
    if value in (np.nan, np.inf, -np.inf):
        return '0'
    if isinstance(value, float):
        return str(round(value, 3))
    return str(value)


def save_features_to_file(filename, features, header=False, ratio=PN_RATIO):
    """Dump features to a CSV file. If the filename ends with .gz, save
       it to a gzipped file automatically.

       If ratio > 0, try and keep this ratio of negative examples to
       positive examples. If ratio > 0, the data rows are shuffled
       randomly to get an unbiased selection of negative examples.
    """
    keys = list(features.keys())
    values = list(features.values())
    label_class_idx = keys.index('label_class')
    if ratio > 0:
        for column in values:
            random.shuffle(column)

    if header is True:
        mode = 'w+'
    else:
        mode = 'a+'

    if filename.endswith('.gz'):
        output = gzip.open(filename, mode)
    else:
        output = open(filename, mode)

    if header is True:
        print(','.join(keys), file=output)

    background = 0.0
    labelled_data = 0.0
    for i in range(0, len(values[0])):
        should_print = True
        if ratio > 0:
            example_class = values[label_class_idx][i]
            if example_class == 'background':
                try:
                    cur_ratio = background / labelled_data
                except ZeroDivisionError:
                    cur_ratio = ratio
                if cur_ratio < ratio:
                    should_print = True
                    background += 1
                else:
                    should_print = False
            else:
                should_print = True
                labelled_data += 1
        if should_print:
            print(','.join([_str(column[i]) for column in values]),
                  file=output)
