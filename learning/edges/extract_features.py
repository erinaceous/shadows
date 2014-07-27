#!/usr/bin/env python2
# vim: set tabstop=4 shiftwidth=4 textwidth=79 cc=72,79:
"""
    Takes images and their associated ground truths, detects all the
    edges in them, and extracts features per edge segment's normal
    (the line, rotated by 90 degrees, and set to a fixed length).

    To reduce the total number of edge instances that features are
    extracted from, the detected edges are reduced in resolution based
    on point distance -- if points are less than --distance value apart,
    then they are omitted from the contour. Empirically, a good value for
    -d/--distance is 3.

    Saves the output in CSV files: one for each of the feature classes
    you have specified. Default is shadow, penumbra, object (all 3).
    Files are output wherever this is ran (the current working
    directory). Their naming convention is as follows:
        [class].[feature_type].[normal_length].features.csv
    Where 'feature_type' currently is just 'edge'. 'normal_length' is
    the length of the normal lines. By default it's 40 (40 pixels - 20 on
    either side of an edge) which has been empirically derived.

    Add '.gz' to the end of the output filename and the file will be
    compressed automatically. Since it can output very large text files
    (over 100MB for my kondo1 image set), this helps shrink the files
    considerably by almost 200% (down to 50MB).

    To get an idea of what the program is doing internally, run it with
    --gui, which will display the image, preprocessed for edge
    detection, alongside the edge-detected image onto which the
    detected contours' normals are overlaid.

    Original Author: Owain Jones [odj@aber.ac.uk]
"""

from __future__ import print_function
from extractor.normals import *
# from lalonde_extractor import *
from extractor.utils import *
import traceback
import argparse
import cv2
import sys
import os


# Configurable defaults
NORMAL_LENGTH = 30
CLASSES = ['shadow', 'penumbra', 'object']
PN_RATIO = 3  # positive/negative example ratio to enforce
DISTANCE = 3  # minimum pixel distance between contour points
LENGTH = 30   # default length of normals


print_header = True  # don't touch this


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-b', '--blur', type=int, default=0)
    parser.add_argument('-f', '--bifilter', type=int, default=0,
                        help='Amount of bilateral filter to apply to images')
    parser.add_argument('-l', '--normal-length', type=int,
                        default=NORMAL_LENGTH,
                        help='Length of the edge normals (default is 30)')
    parser.add_argument('-i', '--inputs', nargs='+',
                        help='Input images')
    parser.add_argument('-g', '--ground-truths', nargs='+',
                        help='Ground truth images')
    parser.add_argument('-o', '--output-file', default=None,
                        help='Name of file to output to. Names ending with ' +
                             '.gz will automatically be compressed with gzip.')
    parser.add_argument('-r', '--ratio', type=float, default=PN_RATIO,
                        help='Ratio of negative to positive examples ' +
                             'to try and enforce')
    parser.add_argument('-p', '--posterize', type=int, default=0,
                        help='Spatial/colour radius for meanshift segmentation')
    parser.add_argument('-d', '--distance', type=float, default=DISTANCE,
                        help='Minimum distance between contour points')
    parser.add_argument('-ps', '--penumbra-as-shadow', action='store_true',
                        default=False,
                        help='Count penumbra as shadow')
    parser.add_argument('--ignore-objects', action='store_true', default=False,
                        help='Don\'t label object edges')
    parser.add_argument('--gui', action='store_true', default=False)
    parser.add_argument('--gui-wait', type=int, default=100)
    return parser.parse_args()


def dirbasename(path):
    """Return an image basename with its immediate parent directory"""
    dirname = os.path.basename(os.path.dirname(path))
    basename = os.path.basename(path)
    return os.path.join(dirname, basename)


def image_id(path):
    """Returns the basename of an image, without any extensions, and
       only the portion of the filename before any underscore ('_')
       characters.
    """
    return os.path.splitext(os.path.basename(path))[0].split('_')[0]


def process_image(image_path, ground_truth_path, args, header=False):
    image_bgr, image_hsv = load_image(image_path, args.blur, args.bifilter)
    ground_truths = load_ground_truth(ground_truth_path,
                                      args.penumbra_as_shadow)
    output_file = args.output_file
    if output_file is None:
        output_file = 'edges.%d.features.csv.gz' % args.normal_length

    bgr = image_bgr.copy()
    hsv = image_hsv.copy()

    # Mean Shift Filtering "posterizes" the colours in 3-channel images
    # somewhat, which is a quicker way of filtering out noise and retaining
    # edges. Similar colours are averaged together, giving you a cartoonish
    # version of the original image.
    # This does, however, remove textural information -- more is lost from
    # doing this than the more expensive bilateral filtering. It seems good for
    # detecting edges however.
    bgr_posterized = cv2.pyrMeanShiftFiltering(bgr, args.posterize, args.posterize)
    grey = cv2.cvtColor(bgr_posterized, cv2.COLOR_BGR2GRAY)
    grey = cv2.equalizeHist(grey)
    grounds = []
    for channel in ground_truths:
        grounds.append(channel.copy())

    for x, scale in enumerate([100, 50, 25]):  # disabling pyramid stuff for now
        s = float(scale) / 100.0
        if s < 1.0:
            bgr = cv2.pyrDown(bgr.copy())
            hsv = cv2.pyrDown(hsv.copy())
            grey = cv2.pyrDown(grey.copy())
            for y, channel in enumerate(grounds):
                # grounds[y] = cv2.pyrDown(channel)
                grounds[y] = cv2.resize(grounds[y].copy(), (0, 0),
                                        fx=0.5, fy=0.5,
                                        interpolation=cv2.INTER_NEAREST)

        print('  Getting edges, contours and normals...')
        edges = get_edges(bgr)

        for ground in grounds:
            ground_edges = cv2.Canny(ground.copy(), 0, 1)
            # ground_edges = 255 - ground_edges
            edges = edges | ground_edges

        contours = [minimum_distance_contour(contour, args.distance)
                    for contour in get_contours(edges)]
        normals = get_normals(contours, args.normal_length)
        # contours = get_contours(edges)
        # normals = []

        if args.gui:
            colours = {
                'background': (200, 0, 200),
                'object': (255, 0, 0),
                'penumbra': (0, 255, 0),
                'shadow': (0, 0, 255)
            }

            # Display the detected normals on the 'edges' image.
            # Also invert the image so we get black edges on a white
            # background with coloured normals.
            edges_display = grounds[0].copy()
            edges_display[edges_display == 0] = 230
            edges_display = edges_display & (255 - edges.copy())
            edges_display = cv2.cvtColor(edges_display, cv2.COLOR_GRAY2BGR)
            for c_id, contour in enumerate(normals):
                for n_id, normal in enumerate(contour):
                    try:
                        normal_flipped = (
                            flip_point(normal[0], bgr),
                            flip_point(normal[1], bgr)
                        )
                        # Convert normal points values to integers
                        # (numpy is fine with all the floating point
                        # arrays, but OpenCV doesn't like them for
                        # the drawing functions)
                        point1 = tuple([int(x) for x in normal[0]])
                        point2 = tuple([int(x) for x in normal[1]])
                        colour = colours[
                            get_point_class(normal_flipped, grounds)
                        ]
                        cv2.line(edges_display, point1, point2, colour)
                    except ValueError:
                        pass

            cv2.imshow('edges', edges_display)
            cv2.imshow('grey', grey)
            cv2.waitKey(args.gui_wait)

        print('  Getting features (%d percent scale)...' % scale)
        features = get_features(bgr, hsv, normals, grounds,
                                length=args.normal_length, debug=True,
                                scale=s,
                                extra_args={
                                    '_img_id': image_id(image_path),
                                    'blur': args.blur
                                },
                                ignore_objects=args.ignore_objects)

        print('  Saving features to file...')
        global print_header
        save_features_to_file(
            output_file, features, header=print_header,
            ratio=args.ratio
        )
        print_header = False  # only print the header for the first file


def main():
    args = parse_arguments()
    inputs = {image_id(path): path for path in sorted(args.inputs)}
    ground_truths = {image_id(path): path
                     for path in sorted(args.ground_truths)}
    comparable_images = {}
    for image in inputs:
        if image in ground_truths:
            comparable_images[inputs[image]] = ground_truths[image]

    for i, image_path in enumerate(comparable_images.keys()):
        print('(%d/%d)' % (i, len(comparable_images)),
              'Processing', dirbasename(image_path))
        ground_truth_path = comparable_images[image_path]

        try:
            process_image(image_path, ground_truth_path, args)
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            traceback.print_exc()
            return 0
        except:
            print('  Failed:')
            traceback.print_exc()

    return 0


if __name__ == '__main__':
    sys.exit(main())
