#!/bin/bash
# 
# time.sh: Keep track of how long it takes programs to run, in lieu of
# being able to run the 'time' command outside of bash.
# 
# Usage:
# time.sh $event path/to/output
#
# Appends timestamp to output file in CSV format:
# start, end
# $start_time, $end_time
#
# If first argument == 'start', clears output file first.

output=/dev/stdout
if [ "$2" != "" ]; then
    output=$2
fi

if [ "$1" = "start" ]; then
    echo -n 'Start, End' > $output
    if [ "$3" != "" ]; then
        echo -n ', Chain' >> $output
    fi
    if [ "$4" != "" ]; then
        echo -n ', Image Set' >> $output
    fi
    echo >> $output
fi

echo -n `date -u '+%s.%N, '` >> $output
if [ "$1" = "end" ]; then
    if [ "$3" != "" ]; then
        echo -n $3 >> $output
    fi
    if [ "$4" != "" ]; then
        echo -n ", $4" >> $output
    fi
    echo >> $output
fi
