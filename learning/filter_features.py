#!/usr/bin/env python2
# vim: set tabstop=4 shiftwidth=4 textwidth=79 cc=72,79:
"""
    filter_features: Open a CSV file (optionally gzipped) and filter
    results so that there are roughly N times as many negative examples
    as there are positive examples (where N is an adjustable parameter).
    
    This is useful because due to the nature of my images, about 99%
    of the feature instances gathered will be negative examples.
    Some learning algorithms like having lots of negative examples,
    others do not.

    I have *way too much* data to train any learning algorithms with
    good performance in a reasonable amount of time / memory.

    This eliminates examples classed as 'background' until there are
    only (len(positive_examples) * N) negative examples left.
    Then it saves it back out to CSV format.

    As my feature extraction files are pretty big (~150MB of just text
    when uncompressed), this tries to keep memory usage low and streams
    in the file rather than loading it at once.

    Original Author: Owain Jones [odj@aber.ac.uk]
"""

from __future__ import print_function
import argparse
import random
import gzip


N = 3


def open_file(csvfile, mode='r'):
    """Opens a file. If the filename ends in .gz, uses gzip.open."""
    if csvfile.endswith('.gz'):
        return gzip.open(csvfile, mode)
    return open(csvfile, mode)


def get_example_indexes(csvfile):
    """Loop over a CSV file and get the indexes of positive examples and
       negative examples.

       Returns a tuple of three lists,
       (header, positive_examples, negative_examples)
    """
    if type(csvfile) is str:
        csv = open_file(csvfile)
    else:
        csv = csvfile

    positive_examples = []
    negative_examples = []

    i = 0
    idx_label_class = -1
    header = []
    for line in csv:
        values = line.strip().split(',')
        if i == 0:
            header = values
            idx_label_class = header.index('label_class')
            i += 1
            continue
        if values[idx_label_class] == 'background':
            negative_examples.append(i)
        else:
            positive_examples.append(i)
        i += 1

    return (header, i, positive_examples, negative_examples)


def filter_features(csvfile, outfile, n=N, verbose=False):
    """Do what it says in the module docstring."""
    header, length, positives, negatives = get_example_indexes(csvfile)
    if verbose:
        print('Got %d examples - %d positive, %d negative' % (
              length, len(positives), len(negatives)
              ))
    random.shuffle(negatives)
    positives = sorted(positives)
    negatives = sorted(negatives[:(len(positives) * n)])
    if verbose:
        print('Picked %d negatives at random' % len(negatives))

    csv = open_file(csvfile)
    out = open_file(outfile, 'w+')

    print(','.join(header), file=out)
    i = 0
    for line in csv:
        line = line.strip()
        if i in positives or i in negatives:
            print(line, file=out)
        i += 1
        if verbose:
            print('(%d/%d) %s' % (i, length, csvfile), end='\r')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output')
    parser.add_argument('-n', type=int, default=N)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    args = parser.parse_args()

    filter_features(args.input, args.output, args.n, args.verbose)
