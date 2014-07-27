#!/usr/bin/env python2
# vim: set tabstop=4 shiftwidth=4 textwidth=79 cc=72,79:
"""
    analyse: Open an image, display on matplotlib graph. Convert to HSV
    colourspace; when user clicks on image area, update charts around image to
    show changes in hue/sat/value.
    Original Author: Owain Jones [odj@aber.ac.uk]
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import numpy
import cv2


im = None
fig = None
rgb = None
hsv = None
hsv_h = None
hsv_v = None
hsv_sq = None
window = 50


def plot(fig, rgb, hsv, pos, no_labels=False):
    """Display the chart.

       Arguments:
       rgb, hsv:    OpenCV Mats to analyse
       pos:         Position of crosshair on image (also used for HSV charts)

       Returns: matplotlib thingy
    """
    global hsv_v, hsv_h, hsv_sq, im
    im = fig.add_subplot(222)
    im.imshow(rgb, picker=True)
    hsv_v = fig.add_subplot(221)
    hsv_h = fig.add_subplot(224)
    hsv_sq = fig.add_subplot(223)

    if no_labels:
        for item in [im, hsv_v, hsv_h, hsv_sq]:
            item.set_xticklabels([])
            item.set_yticklabels([])

    replot_hsv_bars(pos)


def smooth(array, window=window):
    """Smooth a 2D signal (1D signals = hue, sat, value)"""
    w = numpy.ones(window, 'd')
    return numpy.convolve(w / w.sum(), array, mode='valid')


def replot_hsv_bars(pos, window=window):
    """Finds the cross section of pos and window, samples values from the
       HSV image across this cross section in all 3 channels, then plots the
       result.
    """
    hsv_v.clear()
    hsv_h.clear()
    im.clear()
    im.imshow(rgb, picker=True)
    im.vlines(pos[0], pos[1] - window, pos[1] + window, color='black',
              linewidth=3)
    im.vlines(pos[0], pos[1] - window, pos[1] + window, color='white')
    im.hlines(pos[1], pos[0] - window, pos[0] + window, color='black',
              linewidth=3)
    im.hlines(pos[1], pos[0] - window, pos[0] + window, color='white')

    legend = []
    colors = ['b', 'g', 'r']

    for i, _ in enumerate(['hue', 'sat', 'value']):
        color = colors[i % len(colors)]
        horiz_win = hsv[pos[1], pos[0] - window:pos[0] + window, i]
        vert_win = hsv[pos[1] - window:pos[1] + window, pos[0], i]
        horiz_range = range(pos[0] - window, pos[0] + window)
        vert_range = range(pos[1] - window, pos[1] + window)
        hsv_v.plot(vert_win, vert_range, alpha=0.5, color=color)
        hsv_h.plot(horiz_range, horiz_win, alpha=0.5, color=color)
        legend.append(hsv_sq.bar(0, 0, 0, color=color))

    hsv_sq.legend(legend, ['hue', 'sat', 'value'])


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-b', '--blur', type=int, default=0)
    parser.add_argument('-w', '--window', type=int, default=window)
    parser.add_argument('-n', '--no-labels', action='store_true',
                        default=False)
    parser.add_argument('image')
    return parser.parse_args()


def onpick(event):
    replot_hsv_bars(
        (int(event.mouseevent.xdata), int(event.mouseevent.ydata))
    )
    plt.draw()


def main():
    global rgb, hsv, fig, window
    args = parse_arguments()
    image = cv2.imread(args.image)
    if args.blur > 0:
        image = cv2.GaussianBlur(image, (args.blur, args.blur), 0)
    window = args.window
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    fig = plt.figure()
    plot(fig, rgb, hsv, (window, window), no_labels=args.no_labels)
    plt.connect('pick_event', onpick)
    plt.show()


if __name__ == '__main__':
    main()
