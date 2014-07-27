#!/bin/bash

HARNESS=../test_harness

function _help() {
    echo 'Usage:'
    echo -n "$0 "
    grep '^function' $0 | awk '{ print $2 }' | sed 's/[^a-zA-Z]//g' | uniq | tr '\n' ' '
    echo
}

function --help() {
    _help
}

function config() {
    mkdir -p build
    cd build
    if [ "$2" = "debug" ]; then
        cmake -DCMAKE_BUILD_TYPE=Debug ..
    else
        cmake ..
    fi
    status=$?
    cd ..
    return $status
}

function compile() {
    cd build
    # Use compcache when compiling to speed things up
    ccache make -j4
    status=$?
    cd ..
    return $status
}

function compile_loop() {
    while true; do
        ls *.cpp *.hpp *.c *.h CMakeLists.txt 2>/dev/null | inotifywait -qq -e modify --fromfile -
        compile
    done
}

function compile_test_loop() {
    while true; do
        ls *.cpp *.hpp *.c *.h CMakeLists.txt 2>/dev/null | inotifywait -qq -e modify --fromfile -
        echo
        compile
        if [ "$?" = "0" ]; then
            $HARNESS/harness.py -c $HARNESS/config.yaml $HARNESS/config-all.yaml --chain ${1} --image-set test -vdp
            $HARNESS/graph.py /tmp/test_harness/test/regions/**/roc.csv
        fi
    done
}

function compile_test_loop_single() {
    prog=$1
    shift
    while true; do
        ls ${prog}.cpp ${prog}.hpp ${prog}.c ${prog}.h CMakeLists.txt 2>/dev/null | inotifywait -qq -e modify --fromfile -
        echo
        compile
        if [ "$?" = "0" ]; then
            ./build/$prog $*
        fi
        sleep 0.5
    done
}

function doc() {
    mkdir -p doc
    doxygen doxyfile.in
}

function cp() {
    if [ "$*" != "" ]; then
        COMMIT_MSG="$*"
    else
        read -p "Commit Message: " COMMIT_MSG
    fi
    git add -A *
    git commit -a -m "$COMMIT_MSG"
    git push pacific master
}

if [ "$1" = "help" ]; then
    _help
elif [ "$1" != "" ]; then
    CMD=$1
    shift
    $CMD $*
else
    _help
fi
