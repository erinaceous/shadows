"""
    Utilities used by the test_harness script. Functions for handling
    various bits of config files.


    Original Author: Owain Jones [odj@aber.ac.uk]
"""

from __future__ import print_function
from collections import OrderedDict
import utils.yaml_ordered_dict as yaml_ordered_dict
import logging as log
import subprocess
import psutil
import shutil
import copy
import glob
import time
import yaml
import sys
import os


# Python2 uses raw_input for text prompts, which doesn't exist in
# Python3. 'input()' in Python2 also does different things.
try:
    prompt = raw_input
except NameError:
    prompt = input


ASK_THRESHOLD = 10
new_errors = 0
shellenv = dict(os.environ)  # copy shell environment
shellenv['SHELL'] = '/bin/bash'  # i'm using fish by default. bits of
                                 # this script use the $? exitcode var
                                 # which exists in bash and not fish.


def yesno(boolean):
    """Boost-based C++ programs can accept 'yes' and 'no' on command line
       arguments mapped to bools."""
    if boolean is True:
        return 'yes'
    return 'no'


def expand_path(path):
    """Expand paths with ${variables} and ~user/ in them.
       Also expands {harness_dir} to the directory path that this file is in.
    """
    return os.path.normpath(os.path.expandvars(os.path.expanduser(
        path.replace('{harness_dir}',
                     os.path.dirname(os.path.realpath(sys.argv[0])))
    )))


def filelist_expand(files):
    """Given a list of globbed paths, text files or explicit lists of
       files, expand those into a list of list containing paths to
       actual files.

       Arguments:
       files:   The list of paths, e.g.:
                ["/path/to/files/*.jpg",         # Accepts bash globs
                 "/path/to/more/files.txt",      # Accepts file lists
                 ["/path/to/another/file.jpg",   # Accepts actual lists
                  "/path/to/one/more/file.jpg"]] # of files.

       Returns: A list of lists, containing paths to actual files that
                were found on the system.
    """
    output = []
    if type(files) != list:
        files = [files]
    for entry in files:
        file_entry = []
        # Expand any shell variables
        entry = expand_path(entry)
        # Expand any shell globs
        entry = glob.glob(entry)
        for sub_entry in entry:
            # If an entry has the '.txt' extension, assume that it is
            # a textfile containing another list of files, one file path
            # per line. Merge it with our current list.
            if sub_entry.endswith('.txt'):
                sub_entry = open(sub_entry, 'r').readlines()
                file_entry.extend(filelist_expand(sub_entry))
            else:
                file_entry.append(sub_entry)
        output.append(file_entry)
    # return list(itertools.chain(output))
    return sorted(output[0])  # FIXME: why is this a list in a list?


def get_base_filename(filename):
    """Takes a file path, get just the basename (filename without path),
       and Returns the part of the name before the first underscore '_'.
    """
    return os.path.splitext(os.path.basename(filename))[0].split('_')[0]


def parse_config_files(names):
    """Parses a YAML config file, expands the lists of input and ground
       truth files, and finds the files that can be matched against said
       ground truths.

       Arguments:
       name:    Path to YAML config file.

       Returns: A dictionary representing the script's configuration.
    """
    config = OrderedDict()

    for name in names:
        config.update(yaml.load(open(name, 'r'),
                                yaml_ordered_dict.OrderedDictYAMLLoader))

    # Parse the input entries, expand any globs and read filenames from any
    # text files encountered, then stick it back into the config.
    # Do the same with ground truths.
    # (Deep-copying the file lists so that things don't go weird from modifying
    # the lists in-place)
    for x in ['inputs', 'ground_truths']:
        for y in config[x]:
            config[x][y] = filelist_expand(copy.deepcopy(config[x][y]))

    # Look for files that actually exist on the system; don't include missing
    # files. And then sort the resulting new list.
    for x in ['inputs', 'ground_truths']:
        for y in config[x]:
            config[x][y] = sorted([z for z in config[x][y]
                                   if os.path.isfile(z)])

    # Strip off suffixes from filenames so we essentially have the image IDs,
    # use this to generate a list of available ground truth images.
    ground_truth_ids = {}
    for x in config['ground_truths']:
        ground_truth_ids[x] = {get_base_filename(y): y
                               for y in config['ground_truths'][x]}

    # Loop through all the image sets, find input images that have
    # corresponding ground truths, add to new list.
    images_that_can_be_compared = {}
    for image_set in config['inputs']:
        compare_set = {}
        for image in config['inputs'][image_set]:
            for ground_truth_id, ground_truth\
            in ground_truth_ids[image_set].items():
                if ground_truth_id in image:
                    compare_set[image] = ground_truth
        if compare_set != {}:
            images_that_can_be_compared[image_set] = compare_set
    config['compare'] = images_that_can_be_compared

    return config


def build_argument_tree(items):
    """
       Given a dictionary of lists, builds a tree structure representing
       all possible combinations of the values from these lists.

       Arguments:
       items:   Dictionary of lists, looking like this:
                {
                    "foo": [1, 2, 3, 4, 5, 6],
                    "bar": ["hello", "there", "world"],
                    "baz": ["test1", "test2", "test3"]
                }

        Returns: A new dictionary representing a tree structure, which
                 might look like this for example:
                 {"foo":
                        {1:
                            {"bar": {"hello":
                                              {"baz": {"test1": None},
                                    {"there": {"baz": {"test1": None},
                                    ...
                            ...
                        ...
                }
    """

    # Various early-termination things to prevent further recursion
    # (e.g. we're at the end of a branch because the values are all None
    # or aren't iterable objects)
    if type(items) not in [dict, list, tuple, set, OrderedDict]:
        return {str(items): None}
    if items is None:
        return None
    if len(items) == 0:
        return None

    # Find out the length of each tree node.
    # Anything iterable has its length counted. Anything not, e.g. strings or
    # integers, equals 1. NoneType == 0.
    item_lengths = {}
    for x in items:
        if items[x] is not None:
            if type(items[x]) in [dict, list, tuple, set, OrderedDict]:
                item_lengths[x] = len(items[x])
            else:
                item_lengths[x] = 1
        else:
            item_lengths[x] = 0

    # Find the tree node with the most items, and make that the new root node
    # of the tree.
    root_key = max(items, key=lambda x: item_lengths[x])
    root = items[root_key]

    # More early-termination stuff.
    if root is None:
        return {root_key: None}
    if type(root) not in [dict, list, tuple, set, OrderedDict]:
        return {str(root): None}

    # Create a new tree. Shift things around so that the list with most items
    # really is the root.
    new_root = {}
    children = copy.deepcopy(items)
    del children[root_key]
    new_root[str(root_key)] = {}

    # Sweep over all the elements and recurse down their branches of the tree.
    for item in root:
        new_root[str(root_key)][str(item)] = build_argument_tree(children)
    return new_root


def _flatten_argument_node(node, input_list, output):
    # Private function used by 'flatten_argument_tree' below.
    # Searches through a tree one branch at a time, until it terminates at a
    # 'None' value. Append all keys discovered, whilst traversing that branch,
    # to input_list. When the 'None' is reached, convert input_list to a string
    # and append it to output, then terminate.
    # This function does magic with side-effects rather than return values.
    # Which is probably bad, but recursion is hurting my head and this way
    # works.
    # To use it:
    # output = list()
    # _flatten_argument_node(tree, [], output)
    # Your desired result will now be stored in output.
    if type(node) != dict:
        for i, x in enumerate(input_list):
            if 1 - (i % 2) == 1:
                input_list[i] = '--' + str(x)
        output.append(' '.join(input_list))
        return input_list
    for values, arguments in node.items():
        output_list = copy.deepcopy(input_list)
        output_list.append(values)
        output_list = _flatten_argument_node(arguments, output_list, output)


def flatten_argument_tree(tree):
    """Takes a tree built by the build_argument_tree function, and
       flattens it down into a list of strings. These strings represent
       each depth-first branch traversal of the tree.
       Since it's for building up arguments for command line programs,
       It prepends '--' to every odd-numbered tree key."""
    output = []
    _flatten_argument_node(tree, [], output)
    return output


def build_argument_combinations(args):
    """
    Given a dictionary of a program's possible arguments:
    {
        'foo': [1, 2, 3],
        'bar': ['hello', 'world'],
        'baz': 'static_item'
    }
    Returns a list of all the possible combinations of those arguments,
    ready to be used on the command line.
    """
    combinations = []
    list_args = {}
    static_args = {}

    # Loop through the lists of arguments -- if any are static (aren't
    # a list/only have one option), add them to static_keys, otherwise
    # add them to list_keys.
    for key, arg in args.items():
        if type(arg) == list:
            list_args[key] = arg
        else:
            static_args[key] = arg

    # Generate the string forms of the static arguments.
    static_args_str = ''.join(['--%s %s ' % (key, val) for key, val
                               in static_args.items()])

    # Build the list of all possible combinations of the dynamic arguments.
    tree = build_argument_tree(list_args)

    # Since static arguments only have one option and never change, we can
    # safely just prepend them to the strings generated from flattening the
    # dynamic argument tree.
    combinations = [static_args_str + x for x
                    in flatten_argument_tree(tree)]

    return combinations


global run_stats
run_stats = {
    'success': 0,
    'error': 0,
    'finished': 0,
    'total': 0,
    'pool': [],
    'started': None
}


def wait_for_processes(maximum=psutil.NUM_CPUS - 1):
    """Checks the pool of currently running processes. If it's full, waits
       until there's a free slot again."""
    global run_stats, new_errors
    if maximum < 1:
        maximum = 1
    while len(run_stats['pool']) >= maximum:
        shouldWait = True
        for i, proc in enumerate(run_stats['pool']):
            if proc.poll() is not None:
                run_stats['pool'].pop(i)
                if proc.returncode == 0:
                    run_stats['success'] += 1
                else:
                    run_stats['error'] += 1
                    new_errors += 1
                run_stats['finished'] += 1
                shouldWait = False
        print(' ' * (shutil.get_terminal_size()[0] - 1), end='\r')
        print(run_stats_str(), end='\r')
        if shouldWait is False:
            break
        time.sleep(0.1)


def run_stats_str():
    global run_stats
    s = run_stats
    elapsed = time.time() - s['started']
    hours = int(elapsed / 3600.0) % 24
    minutes = int(elapsed / 60.0) % 60
    seconds = int(elapsed) % 60
    elapsed_str = '%d:%02d:%02d' % (hours, minutes, seconds)
    print(' ' * (shutil.get_terminal_size()[0] - 1), end='\r')
    return '%s elapsed, Jobs: %d running, %d finished, %d errored, %d total' %\
           (elapsed_str, len(s['pool']), s['finished'],
            s['error'], s['total'])


def run(command, test=False, parallel=False, threshold=ASK_THRESHOLD):
    """Wrapper for calling external processes.

       Arguments:
       command:     Command to run. This string will be passed to <your
                    default shell>, so can accept shell commands with
                    variable names etc.
       test:        If True, the above command is never ran, instead the
                    command is printed to console.
       parallel:    If True, runs program in parallel -- submits it to
                    a job queue and then returns straight away if it
                    can. If the queue is full (same length as
                    psutil.NUM_CPUS), waits until a slot frees up.
                    NOTE: This will make the return values of this
                    function useless, so if you're debugging things,
                    keep parallel set to False if you want any program
                    output or timing information.

       Returns: A Tuple;
                (no. seconds spent running program,
                 standard output as a byte string.)
    """
    start = time.time()
    global run_stats, new_errors, shellenv
    if run_stats['started'] is None:
        run_stats['started'] = start
    stdout = ''
    if test:
        log.info(command + '\n')
    else:
        if threshold > 0 and new_errors >= threshold:
            prompt("Harness halted: Too many errors. Ctrl+C to quit, " +
                   "Enter to continue.")
            new_errors = 0
        try:
            if parallel is True:
                wait_for_processes()
                run_stats['pool'].append(
                    # 'set -e' tells bash to quit as soon as a command errors.
                    subprocess.Popen("set -e; " + command + "; exit $?",
                                     shell=True,
                                     env=shellenv, executable='/bin/bash')
                )
                run_stats['total'] += 1
            else:
                stdout = subprocess.check_output(command, shell=True,
                                                 stderr=subprocess.STDOUT,
                                                 universal_newlines=True,
                                                 env=shellenv,
                                                 executable='/bin/bash')
        except subprocess.CalledProcessError as e:
            stdout = e.output
            run_stats['error'] += 1
            new_errors += 1
    return (time.time() - start, stdout)


# vim: set tabstop=4 shiftwidth=4 textwidth=79 cc=72,79:
