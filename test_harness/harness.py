#!/usr/bin/env python
"""
    Test suite for shadow detection algorithms.
    Provides an agnostic wrapper around algorithms. Given a selection of
    programs (which take the same standard sets of arguments, inputs and
    outputs), can run them in different combinations and compare the final
    outputs of each method chain against a list of ground truth data files.

    Configuration is defined with a YAML file. See 'config.yaml' in this
    script's directory to see what a correct config file looks like.

    Original Author: Owain Jones [odj@aber.ac.uk]
"""

from __future__ import print_function
from utils.harness import *
import logging as log
import argparse
import shutil
import sys
import os


def parse_command_line_arguments(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__
    )
    parser.add_argument('-a', '--ask', type=int, default=ASK_THRESHOLD,
                        help='If set, pause and prompt whether to continue ' +
                             'when programs in the chain throw over this ' +
                             'many errors.')
    parser.add_argument('-c', '--config', default=['config.yaml'], nargs='+',
                        help='YAML configuration Files to use. By default, ' +
                             'looks for a config.yaml in your working dir.')
    parser.add_argument('-t', '--test', default=False, action='store_true',
                        help='If set, don\'t run anything, just print ' +
                             'commands that would be ran to command line.')
    parser.add_argument('-v', '--verbose', default=False, action='store_true',
                        help='If set, be noisy about everything.')
    parser.add_argument('-d', '--delete-scratch', default=False,
                        action='store_true',
                        help='If set, delete + recreate scratch directory')
    parser.add_argument('--chains', nargs='+', default=None,
                        help='Names of processing chains to run - as defined' +
                             'in the chains section of your config.yaml file.')
    parser.add_argument('-i', '--image-sets', nargs='+', default=None,
                        help='Names of image sets to run on.')
    parser.add_argument('-s', '--skip-to', nargs='+', default=None,
                        help='Program to skip to in the chain (useful for ' +
                             'debugging)')
    parser.add_argument('-g', '--ground-truths-only', default=False,
                        action='store_true',
                        help='If set, only run the ground truths comparisons.')
    parser.add_argument('-n', '--no-ground-truth', default=False,
                        action='store_true',
                        help='If set, don\'t run ground truths comparisons.')
    parser.add_argument('-p', '--parallel', default=False, action='store_true',
                        help='If set, run jobs in parallel on all CPUs')
    return parser.parse_args()


def main():
    # FIXME: This function is getting too complex! (for loop in for loop in for
    # loop in...)
    args = parse_command_line_arguments(sys.argv)

    FORMAT = '%(message)s'
    log.basicConfig(format=FORMAT)
    log.getLogger().setLevel(log.ERROR)
    if args.test:
        log.getLogger().setLevel(log.INFO)
    if args.verbose:
        log.getLogger().setLevel(log.DEBUG)

    try:
        config = parse_config_files(args.config)
    except IOError as e:
        log.error("Couldn't open config file %s" % args.config)
        log.exception(e)
        exit(1)
    except ValueError as e:
        log.error("Error decoding YAML in config file %s" % args.config)
        log.exception(e)
        exit(1)

    ground_truths_program = config['ground_truths_program']
    scratch_dir = config['scratch']
    if args.chains is not None:
        chains = args.chains
    else:
        chains = list(config['chains'].keys())

    if args.image_sets is not None:
        image_sets = args.image_sets
    else:
        image_sets = config['compare'].keys()

    for image_set in image_sets:
        images = config['compare'][image_set].keys()
        ground_truths = config['compare'][image_set].values()

        for chain in chains:
            should_ground_truth = True
            if 'ground_truth' in config['chains'][chain]:
                if config['chains'][chain]['ground_truth'] in (False, None):
                    should_ground_truth = False
                    del config['chains'][chain]['ground_truth']

            chain_arguments = {}
            for program_name in config['chains'][chain]:
                program = config['chains'][chain][program_name]
                if program is not None and 'arguments' in program.keys():
                    program_arguments =\
                        build_argument_tree(program['arguments'])
                    chain_arguments[program_name] =\
                        flatten_argument_tree(program_arguments)
            chain_arguments_tree =\
                build_argument_tree(chain_arguments)
            chain_arguments_flat =\
                flatten_argument_tree(chain_arguments_tree)

            for arguments in chain_arguments_flat:
                arguments_dir = arguments.replace('--', '')\
                    .replace(' ', os.path.sep)
                scratch_dir_expanded = os.path.normpath(scratch_dir.format(
                    image_set=image_set, chain=chain,
                    tunables=arguments_dir
                ))
                arguments_csv = arguments.replace('--', '')

                # Delete the scratch directory if asked to on command line
                if args.delete_scratch:
                    try:
                        shutil.rmtree(scratch_dir_expanded)
                    except FileNotFoundError:
                        pass

                # Recreate the scratch directory if necessary
                try:
                    os.makedirs(scratch_dir_expanded)
                except FileExistsError:
                    log.debug('Tried to create directory %s, but it '
                              % scratch_dir_expanded + 'already exists.')

                # This is needed if we're running just the ground truths
                # testing without running the chain, otherwise scratch_images
                # would never exist.
                scratch_images = []
                for image in images:
                    scratch_image = os.path.join(scratch_dir_expanded,
                                                 os.path.basename(image))
                    # shutil.copyfile(image, scratch_image)
                    scratch_images.append(scratch_image)

                if args.ground_truths_only:
                    program_names = []
                elif args.skip_to is not None:
                    program_names = args.skip_to
                else:
                    program_names = config['chains'][chain].keys()

                # Finally, run the programs in the chain.
                scratch_images = images
                whole_command = []
                for i, program_name in enumerate(program_names):

                    if program_name not in config['programs'].keys():
                        program = {'path': '{prefix}/%s' % program_name}
                    else:
                        program = config['programs'][program_name]
                    arguments_csv = arguments_csv.replace(
                        program_name + ' ', ''
                    )
                    binary = expand_path(program['path'].format(
                        prefix=expand_path(config['default_program']['prefix'])
                    ))
                    # program_arguments = chain_arguments[program_name]
                    # FIXME: There is a better way of doing the line below.
                    program_arguments = arguments.split(program_name)[-1]
                    if 'argument_format' not in program.keys():
                        command = '{path} ' + \
                                  config['default_program']['argument_format']
                    else:
                        command = '{path} ' + program['argument_format']
                    command = command.format(
                        path=binary, output_dir=scratch_dir_expanded,
                        inputs=' '.join(scratch_images),
                        ground_truths=' '.join(ground_truths),
                        chain=chain, program_name=program_name,
                        image_set=image_set, original_images=' '.join(images),
                        arguments=program_arguments.strip(),
                        parameters=arguments_dir,
                        verbose=yesno(args.verbose)
                    )

                    # If we're in parallel mode, add this command to a list.
                    # Otherwise, run it directly now.
                    if args.parallel is False:
                        t, msg = run(command, args.test, threshold=args.ask)
                        if msg != '':
                            for line in msg.split('\n'):
                                log.info('%s->%s: %s',
                                         chain, program_name, line)
                    whole_command.append(command)

                    # The first program will operate on the source images and
                    # save them as .png files in the scratch dir.
                    # The successive programs operate on the scratch images.
                    # (The first command could be the ImageMagick 'convert'
                    # command)
                    if i == 0:
                        scratch_images = []
                        for image in images:
                            scratch_image = os.path.join(
                                scratch_dir_expanded, os.path.basename(image)
                            )
                            scratch_image = os.path.splitext(scratch_image)[0]
                            scratch_image += '.png'
                            scratch_images.append(scratch_image)

                # Now, compare the end-of-chain images with the ground truths!
                # Run the ground_truths program on the scratch images first to
                # generate the CSV file.
                output_csv = 'roc.csv'
                output_csv = os.path.join(scratch_dir_expanded, output_csv)
                if 'argument_format' not in ground_truths_program.keys():
                    command = '{path} ' + \
                              config['default_program']['argument_format']
                else:
                    command = '{path} ' + \
                              ground_truths_program['argument_format']
                binary = expand_path(ground_truths_program['path'].format(
                    prefix=expand_path(config['default_program']['prefix'])
                ))
                command = command.format(
                    path=binary,
                    output_dir=scratch_dir_expanded,
                    output_csv=output_csv,
                    inputs=' '.join(scratch_images),
                    ground_truths=' '.join(ground_truths),
                    chain=chain, image_set=image_set,
                    parameters_list=arguments_csv,
                    parameters=arguments_dir,
                    verbose=yesno(args.verbose)
                )
                if should_ground_truth:
                    whole_command.append(command)

                # If in parallel mode, all the chain's commands (including
                # ground truth) can be ran as one shell command.
                if args.parallel:
                    t, msg = run('; '.join(whole_command),
                                 args.test, args.parallel, args.ask)
                elif should_ground_truth:
                    t, msg = run(command, args.test, threshold=args.ask)
                if msg != '':
                    for line in msg.split('\n'):
                        log.info('%s: %s', chain, line)

    # If we were in parallel mode, probably best to get some feedback on how
    # things went.
    if args.parallel:
        wait_for_processes()
        global run_stats
        print(' ' * (shutil.get_terminal_size()[0] - 1), end='\r')
        print(run_stats_str(), end='\r\n')


if __name__ == '__main__':
    main()


# vim: set tabstop=4 shiftwidth=4 textwidth=79 cc=72,79:
