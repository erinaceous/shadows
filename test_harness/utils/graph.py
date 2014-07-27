# TODO: File some bugs w/ numpy project to get numpy.lib.npyio to work better
# with Python3
# (numpy.recfromcsv(StringIO(...) in Python3 throws TypeErrors because things
# are byte-type instead of str-type)
"""
    Utilities related to the graph.py script. Mostly to do with reading
    in of CSV files and sorting them and fetching specific columns of
    data from them, etc.

    Original Author: Owain Jones [odj@aber.ac.uk]
"""

from __future__ import print_function
import numpy.lib.recfunctions
import numpy
import os
import re


def np_combine_csv_files(csvpaths, verbose=False):
    """Combine a collection of CSV files into a single numpy record
       array. Can take a while! CSV files with different fields
       (different headers, different number of fields) are merged
       together correctly, data type inferral and promotion takes a
       while.

       Treats the first line as a header, uses to name the fields.
       Giving it files without headers will cause weird things to
       happen.

       Arguments:
       csvpaths:    List of text files to read into the array

       Returns: numpy.recarray
    """
    big_csv = numpy.recfromcsv(
        open(csvpaths[0]), case_sensitive=True, deletechars='',
        replace_space=' ', autostrip=True
    )
    if 'File ID' not in big_csv.dtype.names and big_csv['Input'].size > 1:
        big_csv = numpy.lib.recfunctions.append_fields(
            big_csv, 'File ID',
            [os.path.splitext(os.path.basename(x))[0]
             for x in big_csv['Input']],
            usemask=False, asrecarray=True
        )
    for i, csvpath in enumerate(csvpaths[1:]):
        csv_arr = numpy.recfromcsv(
            open(csvpath), case_sensitive=True, deletechars='',
            replace_space=' ', autostrip=True
        )
        if 'File ID' not in csv_arr.dtype.names and csv_arr['Input'].size > 1:
            csv_arr = numpy.lib.recfunctions.append_fields(
                csv_arr, 'File ID',
                [os.path.splitext(os.path.basename(x))[0]
                 for x in csv_arr['Input']],
                usemask=False, asrecarray=True
            )
        for field_name in csv_arr.dtype.names:
            if field_name not in big_csv.dtype.names:
                big_csv = numpy.lib.recfunctions.append_fields(
                    big_csv, field_name, [], usemask=False, asrecarray=True
                )
        big_csv = numpy.lib.recfunctions.stack_arrays(
            (big_csv, csv_arr), usemask=False, asrecarray=True,
            autoconvert=True
        )
        if verbose:
            print('Loaded %d/%d files' % (i + 1, len(csvpaths)), end='\r')
    return big_csv


def np_search_text(array, regex, column=None):
    """Search through a numpy array using a regular expression to filter
       out rows.

       Arguments:
       array:   Numpy array to search through.
       regex:   Regular Expression pattern
       column:  (Optional) If set, only test regex on this column of
                the array.

       Returns: A view of the array with the rows that match that regex.
    """
    regex = re.compile(regex)
    vmatch = numpy.vectorize(lambda x: bool(regex.match(x)))
    vmatch_column =\
        numpy.vectorize(lambda x, c: bool(regex.match(x[c])))

    if column is None:
        return vmatch(array)
    return vmatch_column(array, column)


def filter_array(array, filters=None):
    if filters is None:
        return array
    for f in filters:
        column, value = f.split('=')
        if value.startswith('r/') and value.endswith('/r'):
            array = np_search_text(array, value[2:-2], column)
        else:
            array = array[array[column] == value]
    return array


# vim: set tabstop=4 shiftwidth=4 textwidth=79 cc=72,79:
