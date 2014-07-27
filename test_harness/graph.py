#!/usr/bin/env python2
# vim: set tabstop=4 shiftwidth=4 textwidth=79 cc=72,79:
"""
    Parses a bunch of CSV files generated from the harness.py and ground_truth
    programs, and provides various ways of graphing the results.

    Original Author: Owain Jones [odj@aber.ac.uk]
"""
# TODO: Figure out how to switch between graphs using the arrow keys.
# TODO: Figure out how to do mouse-overs of graphs and get labels to show up.
# TODO: Figure out how to get picker event to get right element

from __future__ import print_function
import matplotlib.pyplot as plt
from utils.graph import *
import numpy as np
import matplotlib
import numpy.lib
import argparse
import sys


DEFAULT_GRAPH_TYPE = 'roc'
DEFAULT_SEPARATION = 'Chain'
state = {}  # since events are in a different scope from the methods they're
            # defined in, we need some global variable in which to store state
            # so that data returned with onclick etc. events makes sense.
colors = ['b', 'r', 'm', 'g', 'y', 'c']
hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
markers = ['o', 'v', '*', 's', '+', 'D']
graph_types = {}


class graph(object):
    """Simple annotation class. Adding this to a function declaration adds the
       function to a global table, graph_types. This is used by the script to
       determine what functions can be used to plot data (what functions can be
       selected on the command line). Functions using the graph annotation must
       accept an argparse Namespace as their first and only positional
       argument. They must also return True if they want the graph to be
       displayed graphically (so if the function returns a text-only table,
       return False or don't return anything at all)."""

    def __init__(self, func):
        graph_types[func.__name__] = func


@graph
def what(args=None):
    """Lists the available graph types."""
    print('Available graph types:\n')
    for graph_type in graph_types:
        print('  -t %s:' % graph_type)
        doc = graph_types[graph_type].__doc__
        if doc is not None:
            doc = doc.strip()
        print('      ', doc)
        print()


def _confusion_matrix(args):
    # TODO: This never got implemented properly. It relies on the ground truth
    # program producing confusion matrices. It can't be done using the data
    # from roc.csv files.

    everything = np_combine_csv_files(args.csvs, args.verbose)
    everything = filter_array(everything, args.filter)

    classes = ['Shadow', 'Object', 'Penumbra']
    max_len = max([len(c) for c in classes])

    print('Class,', ' ' * (max_len - len('class,')), ', '.join(classes))
    for row in classes:
        print(row, end=',')
        print(' ' * (max_len - (len(row) - 1)), end='')
        for i, column in enumerate(classes):
            end = ', '
            if i == len(classes) - 1:
                end = '\n'
            print('0', ' ' * (len(column) - 2), end=end)


_confusion_matrix_latex = _confusion_matrix


@graph
def confusion_matrix(args):
    """Print a simple CSV confusion matrix, with 'Predicted Class' across
       the top and 'Actual Class' along the left side."""
    if args.latex:
        _confusion_matrix_latex(args)
    else:
        _confusion_matrix(args)


@graph
def timing(args):
    """Displays a bar chart of execution times recorded in timing.csv files."""
    everything = np_combine_csv_files(args.csvs, args.verbose)
    everything = filter_array(everything, args.filter)

    chains = list(set(everything['Chain']))
    chain_labels = chains
    image_sets = list(set(everything['Image Set']))
    image_labels = image_sets
    if args.latex:
        image_labels = ['\\verb;%s;' % label for label in image_labels]
        chain_labels = ['\\verb;%s;' % label for label in chain_labels]

    plt.clf()
    plt.close('all')
    fig, ax = plt.subplots()
    ax.set_xlabel('Image Set')
    ax.set_ylabel('Time Taken (microseconds)')

    N = len(image_sets)
    ind = np.arange(N)
    width = 1.0 / len(chains)
    half = len(chains) / 2.0
    ind = numpy.arange((width * half), N + width)
    bars = list()

    ax.set_xticks(ind)
    ax.set_xticklabels(image_labels)

    # Create the graph's legend, assign specific colour to each chain
    bars = []
    bcolors = {}
    for i, chain in enumerate(chains):
        bars.append(ax.bar(0, 0, 0, color=colors[i % len(colors)]))
        bcolors[chain] = colors[i % len(colors)]
    ax.legend(bars, chain_labels)

    # Loop over the image sets, plot the bars for each, give each image set a
    # specific hatching pattern for the bars.
    for i1, image_set in enumerate(image_sets):
        y1 = everything[everything['Image Set'] == image_set]
        for i2, chain in enumerate(chains):
            y2 = y1[y1['Chain'] == chain]
            bars.append(ax.bar(
                i1 + (i2 * width),
                np.mean(y2['Time']),
                width,
                color=bcolors[chain],
                hatch=hatches[i1 % len(hatches)],
                yerr=numpy.std(y2['Time']),
                ecolor=bcolors[chain]
            ))

    return True


def _dice(array):
    """Given an array containing true/false positive/negative columns for the
       'shadow' class, calculates the dice coefficient."""
    v = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    for val in v:
        v[val] = float(np.sum(array['Shadow ' + val]))
    dice = v['TP'] / ((v['FP'] + v['TP']) + (v['TP'] + v['FN']))
    return dice


@graph
def dice_table(args):
    """Calculates the Dice coefficient of a dataset:
       http://sve.bmap.ucla.edu/instructions/metrics/dice/
    """
    everything = np_combine_csv_files(args.csvs, args.verbose)
    everything = filter_array(everything, args.filter)
    separators = sorted(list(set(everything[args.separate_on])))
    separator_labels = separators
    separate_on = args.separate_on
    image_sets = list(set(everything['Image Set']))

    if args.latex:
        separator_labels = [str(x) for x in separators]
        separate_on = '\\verb;%s;' % args.separate_on
    ls = 'l' * (len(separators) + 1)
    print('\n\n\\begin{tabular}{%s}' % ls)
    print('Image Set &', ' & '.join(separator_labels), '\\\\ \hline')
    for i1, image_set in enumerate(image_sets):
        print(image_set, '& ', end='')
        for i2, separator in enumerate(separators):
            filtered = everything[everything['Image Set'] == image_set]
            filtered = filtered[filtered[args.separate_on] == separator]
            dice = _dice(filtered)
            if i2 < len(separators) - 1:
                print('%.3f' % dice, '& ', end='')
            else:
                print('%.3f' % dice, '\\\\', end='')
                if i1 == len(image_sets) - 1:
                    print('\hline')
                else:
                    print()

    print('Combined &', end='')
    for i2, separator in enumerate(separators):
        filtered = everything[everything[args.separate_on] == separator]
        dice = _dice(filtered)
        if i2 < len(separators) - 1:
            print('%.3f' % dice, '& ', end='')
        else:
           print('%.3f' % dice, '\\\\')

    print('\\end{tabular}')
    print('\caption{\label{fig:XXX}')
    print('Comparison of different', separate_on)
    print('}')


@graph
def roc_table(args):
    """Prints ROC statistics in table form"""
    everything = np_combine_csv_files(args.csvs, args.verbose)
    everything = filter_array(everything, args.filter)
    separators = sorted(list(set(everything[args.separate_on])))
    separator_labels = separators
    separate_on = args.separate_on
    image_sets = list(set(everything['Image Set']))
    if args.latex:
        separator_labels = [
            str(x) for x in separators
        ]
        separate_on = '\\verb;%s;' % args.separate_on

    ls = 'l' * (len(separators) + 1)
    print('\n\n\\begin{tabular}{%s}' % ls)
    print('Image Set & Mean TPR; %s=' % separate_on,
          ' & = '.join(separator_labels), '\\\\ \hline')

    for i1, image_set in enumerate(image_sets):
        print(image_set, '& ', end='')
        perf = 0.0
        for i2, separator in enumerate(separators):
            filtered = everything[everything['Image Set'] == image_set]
            filtered = filtered[filtered[args.separate_on] == separator]
            tpr = np.ma.masked_invalid(filtered['Shadow TPR'])
            tpr_mean = round(np.mean(tpr), 2)
            if i2 == 0:
                perf = tpr_mean
                print(tpr_mean, end='')
            else:
                c_perf = round(tpr_mean - perf, 2)
                if c_perf == 0:
                    c_perf = ''
                elif c_perf > 0:
                    c_perf = ' (+%.2f)' % c_perf
                else:
                    c_perf = ' (%.2f)' % c_perf
                print('%.2f%s' % (tpr_mean, c_perf), end='')
            if i2 < len(separators) - 1:
                print(' & ', end='')
            else:
                print(' \\\\')
    print('\\end{tabular}')


@graph
def roc_bars(args):
    """Displays ROC True Positive Rate vs False Positive Rate on a bar chart"""
    everything = np_combine_csv_files(args.csvs, args.verbose)
    everything = filter_array(everything, args.filter)
    separators = sorted(list(set(everything[args.separate_on])))
    separator_labels = separators
    separate_on = args.separate_on
    if args.latex:
        separator_labels = [
            '\\verb;%s;' % x
            for x in separators
        ]
        separate_on = '\\verb;%s;' % args.separate_on

    plt.clf()
    plt.close('all')
    fig, ax = plt.subplots()

    ax.set_xlabel(separate_on)
    ax.set_ylabel('False Positive Rate (Bottom), True Positive Rate (Top)')

    N = len(separators)
    ind = np.arange(N)
    width = 1.0
    half = width / 2.0
    ind = np.arange(half, N + half)

    ax.set_xticks(ind)
    ax.set_xticklabels(separator_labels)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticklabels([1.0, 0.5, 0.0, 0.5, 1.0])

    for i1, separator in enumerate(separators):
        filtered = everything[everything[args.separate_on] == separator]
        top = np.ma.masked_invalid(filtered['Shadow TPR'])
        bottom = np.ma.masked_invalid(filtered['Shadow FPR'])
        color = colors[i1 % len(colors)]

        height_top = np.mean(top)
        err_top = np.std(top)
        ax.bar(
            i1, height_top, width, color=color, yerr=err_top, ecolor=color
        )

        height_bottom = -np.mean(bottom)
        err_bottom = np.std(bottom)
        ax.bar(
            i1, height_bottom, width, color=color, yerr=err_bottom,
            ecolor=color, hatch='/'
        )

    return True


@graph
def roc(args):
    """Displays ROC curve as a scatter plot."""
    everything = np_combine_csv_files(args.csvs, args.verbose)
    everything = filter_array(everything, args.filter)

    separators = sorted(list(set(everything[args.separate_on])))
    separator_labels = list(separators)
    separate_on = args.separate_on
    if args.latex:
        separator_labels = [
            '\\verb;%s;' % str(x)
            for x in separators
        ]
        separate_on = '\\verb;%s;' % args.separate_on
    parameters = list(set(everything['Parameters']))

    plt.clf()
    plt.close('all')
    fig, ax = plt.subplots()

    fig.set_size_inches(3.54, 3.54)

    ax.set_xlabel('Shadow False Positive Rate')
    ax.set_ylabel('Shadow True Positive Rate')
    ax.plot([-0.1, 1.1], [-0.1, 1.1], color='y')
    ax.axis([-0.1, 1.1, -0.1, 1.1])
    ax.set_title('Comparing different %ss' % separate_on)

    legend = list()
    for i1, separator in enumerate(separators):
        legend.append(
            matplotlib.patches.Rectangle((0, 0), 1, 1,
                                         fc=colors[i1 % len(colors)])
        )
        filtered = everything[everything[args.separate_on] == separator]
        parameters = set(filtered['Parameters'])
        for i2, parameter in enumerate(parameters):
            filtered2 = filtered[filtered['Parameters'] == parameter]
            filtered2.sort(order=['Shadow FPR', 'Shadow TPR'])
            ax.scatter(filtered2['Shadow FPR'],
                       filtered2['Shadow TPR'],
                       color=colors[i1 % len(colors)], s=30,
                       marker=markers[i1 % len(markers)], picker=3.0, alpha=1)
    ax.legend(legend, separator_labels, loc=4)  # legend in lower right
    return True


@graph
def flat_roc(args):
    """Slightly quicker way of getting an ROC chart."""
    everything = np_combine_csv_files(args.csvs)
    everything.sort(order=['Shadow TPR'])
    plt.clf()
    plt.close('all')
    plt.xlabel('False Positive Rate')
    plt.ylabel('Shadow TPR')
    plt.plot([-0.1, 1.1], [-0.1, 1.1], color='y')
    plt.axis([-0.1, 1.1, -0.1, 1.1])
    plt.title('Performance of everything')
    plt.plot(everything['Shadow FPR'], everything['Shadow TPR'], marker='.')
    return True


def parse_commandline_arguments(argv):
    """
        Parse arguments from the command line.
        Arguments:
        argv:       argv style list (as you would get from sys.argv)

        Returns:    A Namespace containing parsed arguments
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--black', default=False, action='store_true',
                        help='If set, draw black background and white text')
    parser.add_argument('-v', '--verbose', default=False, action='store_true',
                        help='Print out debug information')
    parser.add_argument('-t', '--type', default=DEFAULT_GRAPH_TYPE,
                        help='Type of graph to display ' +
                             '("--type what" returns list of available types)')
    parser.add_argument('-s', '--separate-on', default=DEFAULT_SEPARATION,
                        help='What property to separate ROC classes ' +
                             '(colours) on')
    parser.add_argument('-f', '--filter', default=None, nargs='+', type=str,
                        help='Filter out data')
    parser.add_argument('-l', '--latex', default=False, action='store_true',
                        help='Set up graphs better to export to report')
    parser.add_argument('csvs', nargs='*',
                        help='Path to .csv files')
    args = parser.parse_args()  # FIXME: sys.argv is being parsed wrong?
    return args


def main():
    args = parse_commandline_arguments(sys.argv)

    if args.latex:
        plt.rcParams['text.latex.preamble'] = [
            r"\usepackage{ebgaramond}",
            r"\usepackage{verbatim}"
        ]
        params = {
            'text.usetex': True,
            'font.size': 11,
            'font.family': 'ebgaramond',
            'text.latex.unicode': True,
        }
        plt.rcParams.update(params)

    if args.black:
        matplotlib.rcParams['patch.facecolor'] = 'black'
        matplotlib.rcParams['patch.edgecolor'] = 'white'
        matplotlib.rcParams['axes.facecolor'] = 'black'
        matplotlib.rcParams['axes.edgecolor'] = 'white'
        matplotlib.rcParams['axes.labelcolor'] = 'white'
        matplotlib.rcParams['figure.facecolor'] = 'black'
        matplotlib.rcParams['figure.edgecolor'] = 'white'
        matplotlib.rcParams['text.color'] = 'white'
    if args.type not in graph_types.keys():
        print('Graph type "%s" not valid.' % args.type)
        graph_types['what'](None)
        return 1
    if len(args.csvs) != 0:
        display = graph_types[args.type](args)
    else:
        print('No input data files given.')
        graph_types['what'](None)
        return 1

    if display is not True:
        return 0

    if args.latex:
        plt.margins(0, 0, tight=True)
        plt.tight_layout(0)

    plt.show()


if __name__ == '__main__':
    sys.exit(main())
