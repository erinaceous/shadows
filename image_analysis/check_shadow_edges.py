#!/usr/bin/env python2
# vim: set tabstop=4 shiftwidth=4 textwidth=79 cc=72,79:
"""
    check_shadow_edges: Open an image and a ground truth. Detect the
    shadow/object edges in the ground truth. Find the contours of those edges.
    Generate normals of those edges, sample data across those edges in the
    original image, then plot the results.

    Original Author: Owain Jones [odj@aber.ac.uk]
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import argparse
import random
import math
import cv2
import sys
import os


channels = {
    'hue': 0,
    'saturation': 1,
    'sat': 1,
    'value': 2,
    'val': 2,
    'red': 0,
    'green': 1,
    'blue': 2
}


def load_image(path, blur=0):
    """Loads the image at `path` and also converts it to the HSV colour space.
       Also optionally performs a Gaussian blur on it.
    
       Returns: Two cv2.Mat objects; one for the original RGB image, and one
       for the HSV image.
    """
    image = cv2.imread(path, cv2.CV_LOAD_IMAGE_COLOR)

    if blur > 0:
        image = cv2.GaussianBlur(image, (blur, blur), 0)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    return (image, image_rgb, image_hsv)


def load_image_gray(path):
    return cv2.imread(path, cv2.CV_LOAD_IMAGE_GRAYSCALE)


def overlay(background, foreground):
    """Overlay one array onto another"""
    new = background.copy()

    for channel in range(0, background.shape[2]):
        new[:, :, channel] = np.clip(new[:, :, channel] + foreground, 0, 255)

    return new


def overlay_edges(background, edges, color=None):
    new = background.copy()

    if color is None:
        color = tuple([255] * background.shape[2])

    new[edges == 255] = color
    return new


def get_values(array, line, mode='absolute'):
    """Get values over a line in a 2D array.

       Returns: a 1D array of values
    """
    x1, y1 = line[0]
    x2, y2 = line[1]
    length = int(np.hypot(x2 - x1, y2 - y1))

    x = np.linspace(x1, x2, length).astype('uint8')
    y = np.linspace(y1, y2, length).astype('uint8')

    values = []
    for i in range(length):
        try:
            val = int(array[y[i], x[i]])

            if mode == 'absolute':
                values.append(val)
            elif mode == 'gradient':
                if i > 0:
                    values.append(abs(val - values[0]))
                else:
                    values.append(val)

        except IndexError:
            continue

    return values


def round_partial(value, resolution):
    """Rounds any number to an arbitrary nearest number.
       round_partial(7, 5) would round the value 7 to the closest value
       of 5, therefore returning 5.
    """
    return round(value / resolution) * resolution


def anyrange(start, stop, interval, **kwargs):
    if start > stop:
        return np.arange(start, stop, -interval, **kwargs)
    return np.arange(start, stop, interval, **kwargs)


def quantize_contour(contour, bin_size=4):
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
            idx = len(bins) - 1
            if idx > 0:
                lastx, lasty = bins[idx - 1]
                midx = anyrange(x, lastx, bin_size)
                midy = anyrange(y, lasty, bin_size)

            bins.append((x, y))
    return bins


def simplify_contour(contour, point_every=15):
    new_contour = []
    for p_idx, point in enumerate(contour):
        if p_idx == 0 or p_idx % point_every == 0:
            new_contour.append(point.flatten())

    return new_contour


def get_distance(point1, point2):
    try:
        d = math.sqrt((abs(point2[0] - point1[0]) ^ 2) +
                      (abs(point2[1] - point1[1]) ^ 2))
    except ValueError:
        d = 0.0
    return d


def minimum_distance_contour(contour, distance=3):
    new_contour = []
    last_point = contour[0].flatten()
    new_contour.append(last_point)
    for p_idx, point in enumerate(contour):
        if p_idx == 0:
            continue

        point = point.flatten()

        if get_distance(last_point, point) > distance:
            new_contour.append(point)
            last_point = point

    return new_contour


def get_normals(contours, length=30, epsilon=0, array=None):
    """Loops over a list of lists [of lists] (contour points).
       Calculates the normals of each line segment for each contour (the line
       rotated by 90 degrees). Makes the lines `length` long.

       Returns: A list of list of lists:
            [contours -> [contour -> [line segment normal (x, y)]]]
    """
    normals = list()
    half = length / 2.0
    for c_idx, contour in enumerate(contours):
        new_contour = list()
        approx = contour
        # approx = cv2.approxPolyDP(contour, epsilon, False)
        # approx = quantize_contour(contour)
        # approx = simplify_contour(contour)
        approx = minimum_distance_contour(contour)
        for s_idx, segment in enumerate(approx):
            point1 = approx[s_idx]
            point2 = approx[s_idx - 1]
            normal1 = (point2[0], point1[1])
            normal2 = (point1[0], point2[1])

            if array is not None\
            and array[normal1[1], normal1[0]] > array[normal2[1], normal2[0]]:
                tmp = normal1
                normal1 = normal2
                normal2 = tmp

            center = (
                normal1[0] + ((normal2[0] - normal1[0]) / 2.0),
                normal1[1] + ((normal2[1] - normal1[1]) / 2.0)
            )

            diff = (
                normal2[0] - normal1[0],
                normal2[1] - normal1[1]
            )

            angle = math.degrees(math.atan2(diff[0], diff[1]))

            line1 = (
                center[0] - half * math.sin(angle),
                center[1] - half * math.cos(angle)
            )

            line2 = (
                center[0] + half * math.sin(angle),
                center[1] + half * math.cos(angle)
            )

            line1 = (int(line1[0]), int(line1[1]))
            line2 = (int(line2[0]), int(line2[1]))

            new_contour.append((line1, line2))
        normals.append(new_contour)
    return normals


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-i', '--image', nargs='+')
    parser.add_argument('-g', '--ground-truth', nargs='+')
    parser.add_argument('-b', '--blur', type=int, default=0)
    parser.add_argument('-w', '--window', type=int, default=30)
    parser.add_argument('-c', '--channel', default='value')
    parser.add_argument('-e', '--epsilon', type=float, default=0.0)
    parser.add_argument('-r', '--random-normal', action='store_true',
                        default=False)
    parser.add_argument('-l', '--latex', default=False, action='store_true')
    return parser.parse_args()


def filename_short(filename):
    """Returns a filename with its immediate parent directory"""
    dirname = os.path.basename(os.path.dirname(filename))
    basename = os.path.basename(filename)
    return os.path.join(dirname, basename)


def get_random_normals(normals, count=1):
    new_normals = []
    for i in range(0, count):
        contour = random.choice(normals)
        normal = random.choice(contour)
        new_normals.append([normal])
    return new_normals


def plot_values_from_image(normals, normals_mat, array, img_means,
                           plot, color, random_normal=False,
                           multiple_images=False):
    values = []
    alpha = 0.1
    thickness = 1
    if random_normal:
        normals = get_random_normals(normals, 3)
        alpha = 1.0
        thickness = 3
    for contour in normals:
        for i, normal in enumerate(contour):
            cv2.line(normals_mat, normal[0], normal[1], 255, thickness)
            v = get_values(array, normal)
            side1 = v[len(v) / 2:]
            side2 = v[:len(v) / 2]
            if np.mean(side1) > np.mean(side2):
                v.reverse()
            if not multiple_images:
                plot.plot(v, color, alpha=alpha)
            if random_normal is False:
                values.append(v)
    try:
        length = max([len(x) for x in values])
        means = []
        centers = []
        try:
            for x in range(length):
                means.append(np.mean(values[:][x]))
                centers.append(len(values[x]))
        except IndexError as e:
            print(e)
        if multiple_images:
            plot.plot(means, color)
        else:
            plot.plot(means, color='black')
    except ValueError as e:
        print(e)
    for i, img_mean in enumerate(img_means):
        if not multiple_images:
            plot.axhline(img_mean, color='grey')
            try:
                plot.axvline(centers[0] / 2.0, color='grey')
            except:
                pass


def image_id(path):
    return os.path.splitext(os.path.basename(path))[0].split('_')[0]


def main():
    args = parse_arguments()

    images = {image_id(path): path for path in sorted(args.image)}
    ground_truths = {image_id(path): path for path in sorted(args.ground_truth)}
    comparable_images = {}
    for image in images:
        if image in ground_truths:
            comparable_images[images[image]] = ground_truths[image]


    # If LaTeX mode is on, set up some stuff so fonts are the same
    # as the ones used in my report.
    if args.latex:
        plt.rcParams['text.latex.preamble'] = [
            r"\usepackage{ebgaramond}",
            r"\usepackage{verbatim}"
        ]
        plt.rcParams.update({
            'savefig.dpi': 600,
            'text.usetex': True,
            'font.size': 11,
            'font.family': 'ebgaramond',
            'text.latex.unicode': True
        })

    # Configure the matplot plot. Set up 4 subplots in a grid layout;
    # 3 for the shadow/penumbra/object value line charts, plus 1 for
    # displaying the image. Also sets their titles/labels, and removes
    # axis numbering.
    # If the program is given more than one image, hide the image
    # preview and put the plots into a 1-row, 3-column layout.
    fig = plt.figure()

    if len(comparable_images) > 1:
        gs = matplotlib.gridspec.GridSpec(1, 3)
        shadow_plot = fig.add_subplot(gs[0, 0])
        penumbra_plot = fig.add_subplot(gs[0, 1])
        object_plot = fig.add_subplot(gs[0, 2])
        image_plot = None
    else:
        gs = matplotlib.gridspec.GridSpec(2, 2)
        shadow_plot = fig.add_subplot(gs[0, 0])
        penumbra_plot = fig.add_subplot(gs[1, 1])
        object_plot = fig.add_subplot(gs[1, 0], sharey=shadow_plot)
        image_plot = fig.add_subplot(gs[0, 1])
        image_plot.set_xticks([])
        image_plot.set_yticks([])

    shadow_plot.set_rasterization_zorder(1)  # rasterize this when there's
                                             # lots of lines
    shadow_plot.set_title('Shadow Edges')
    shadow_plot.set_xticks([])
    penumbra_plot.set_rasterization_zorder(1)
    penumbra_plot.set_title('Penumbra Edges')
    penumbra_plot.set_xticks([])
    object_plot.set_rasterization_zorder(1)
    object_plot.set_title('Object Edges')
    object_plot.set_xticks([])
    object_plot.set_ylabel(args.channel.title())
    penumbra_plot.set_ylabel(args.channel.title())
    shadow_plot.set_ylabel(args.channel.title())

    i_means = []

    for image, ground_truth in comparable_images.items():
        print("Analysing", image)
        
        # Load the input image, get the mean value for the channel we're
        # currently looking at from the HSV version of the image.
        i_bgr, i_rgb, i_hsv = load_image(image, args.blur)
        i_means.append(np.mean(i_hsv[:, :, channels[args.channel]]))

        # Load the ground truth image, create three masks from it -
        # one for each of the positive classes.
        # For those classes, pixels marked as that class are set to 0,
        # background is set to 255.
        ground = load_image_gray(ground_truth)
        shadows = ground.copy()
        # shadows[shadows == 25] = 255  # ignore penumbra
        # shadows[shadows == 153] = 255  # ignore objects
        # shadows[shadows < 255] = 0  # everything else is shadow
        shadows[shadows != 0] = 255
        penumbra = ground.copy()
        penumbra[penumbra != 25] = 255
        penumbra[penumbra == 25] = 0
        objects = ground.copy()
        objects[objects != 153] = 255
        objects[objects == 153] = 0

        # Do Canny edge detection on the above classes, with a really
        # low difference in thresholds -- it doesn't really matter for
        # the ground truth images which are essentially binary images.
        # Canny always detects the edges correctly.
        shadow_edges = cv2.Canny(shadows, 0, 1)
        penumbra_edges = cv2.Canny(penumbra, 0, 1)
        object_edges = cv2.Canny(objects, 0, 1)

        # Find the contours of the above edges
        shadow_contours, _ = cv2.findContours(shadow_edges.copy(),
                                              cv2.RETR_LIST,
                                              cv2.CHAIN_APPROX_SIMPLE)
        penumbra_contours, _ = cv2.findContours(penumbra_edges.copy(),
                                                cv2.RETR_LIST,
                                                cv2.CHAIN_APPROX_SIMPLE)
        object_contours, _ = cv2.findContours(object_edges.copy(),
                                              cv2.RETR_LIST,
                                              cv2.CHAIN_APPROX_SIMPLE)

        # Get the normals for the contours detected from each class
        shadow_normals = get_normals(
            shadow_contours, args.window, args.epsilon,
            i_hsv[:, :, channels[args.channel]]
        )
        penumbra_normals = get_normals(
            penumbra_contours, args.window, args.epsilon,
            i_hsv[:, :, channels[args.channel]]
        )
        object_normals = get_normals(
            object_contours, args.window, args.epsilon,
            i_hsv[:, :, channels[args.channel]]
        )

        # Create some new images, which we'll use later to draw the
        # normals onto. Fill them in as black. These are also binary
        # one-channel images.
        shadow_normals_mat = shadow_edges.copy()
        shadow_normals_mat.fill(0)
        penumbra_normals_mat = penumbra_edges.copy()
        penumbra_normals_mat.fill(0)
        object_normals_mat = object_edges.copy()
        object_normals_mat.fill(0)

        multiple_images = len(args.image) > 1

        # Using the detected normals, for each of the classes plot the data
        # from each normal onto the line graphs previously defined.
        plot_values_from_image(shadow_normals, shadow_normals_mat,
                               i_hsv[:, :, channels[args.channel]], i_means,
                               shadow_plot, 'r', args.random_normal,
                               multiple_images)
        plot_values_from_image(penumbra_normals, penumbra_normals_mat,
                               i_hsv[:, :, channels[args.channel]], i_means,
                               penumbra_plot, 'g', args.random_normal,
                               multiple_images)
        plot_values_from_image(object_normals, object_normals_mat,
                               i_hsv[:, :, channels[args.channel]], i_means,
                               object_plot, 'b', args.random_normal,
                               multiple_images)

        # Overlay a bunch of stuff onto the image display -- the normals,
        # and also the original edges of the ground truth classes.
        output = overlay_edges(i_bgr, penumbra_normals_mat, (0, 255, 0))
        output = overlay_edges(output, shadow_normals_mat, (0, 0, 255))
        output = overlay_edges(output, object_normals_mat, (255, 0, 0))
        output = overlay_edges(output, penumbra_edges, (0, 255, 0))
        output = overlay_edges(output, shadow_edges, (0, 0, 255))
        output = overlay_edges(output, object_edges, (255, 0, 0))
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

        if image_plot is not None:
            image_plot.imshow(output_rgb)
            cv2.imwrite('normals.png', output)

    if args.latex:
        plt.tight_layout(0)
        plt.margins(0, 0, tight=True)
    plt.show()


if __name__ == '__main__':
    main()
