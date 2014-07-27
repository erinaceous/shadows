This is the GitHub copy of code written by myself for my undergraduate
dissertation project for my BSc at Aberystwyth University --
"Shadow Detection for Mobile Robots".
If you're interested in the project, you can read about how I [failed to
produce anything working, because naive undergrad going into serious real
world computer vision and machine learning for the first time] in the report,
of which I've uploaded a copy to this repo:
https://cdn.rawgit.com/erinaceous/shadows/master/odj_cs39440_report.pdf


If you make use of this and want the accompanying data to go with it, you are
best off contacting me via my university email account:
odj@aber.ac.uk


Subdirectories of this repo:
    ground_truth_drawing: Painting program for ground truthing.
    image_analysis:       Plots graphs of values around edges in images.
    learning/edges:       The edge feature extraction method used in this
                          project.
    learning/grid:        Another unfinished method of extracting features from
                          image 'cells', where images are treated as uniform
                          grids.
    methods:              Contains all the programs written in C++ for this
                          project -- all the shadow detection implementations.
    test_harness:         The test harness and graphing tools.


TO COMPILE C++ PROGRAMS ON A *NIX MACHINE:
(Assuming CMake is installed, and OpenCV libraries can be found)
cd methods
./build.sh config debug
./build.sh compile


TO RUN TEST HARNESS (Example):
Python Dependencies: numpy, matplotlib, yaml
cd test_harness
./harness.py -vdp -a 10 -c config.yaml scenes.yaml --chains hypot1
    -v: Verbose flag
    -d: Delete output directories before use (so old data isn't kept around)
    -p: Run chains in parallel using all but one of the CPUs available
    -a 10: After 10 errors are caught, pause the harness and wait for user input
    -c config.yaml scenes.yaml: Load these configuration files.
    --chains hypot1 hypot2: Run the hypot1 and hypot2 chains
Should the harness run successfully (assuming all programs are where they should
be, e.g. the C++ programs have been compiled), it should output some statistics
as it runs:
$ 0:00:09 elapsed, Jobs: 6 running, 12 finished, 0 errored, 18 total

By default (in config.yaml), the harness asks programs to output to
/tmp/test_harness/. Run `tree` on that directory to see the directory structure
it creates.

To check the ROC outputs:
find /tmp/test_harness -name "roc.csv" | xargs ./graph.py -v -t roc
    -t roc: Display a ROC scatter plot. Could also try '-t roc_bars' or '-t dice'


FOR FEATURE EXTRACTION (Example):
Python Dependencies: numpy, matplotlib, Python OpenCV 2 bindings (cv2)
cd learning/edges
./extract_features.py --inputs ../../../data/images/kondo1_bifilter/0020.png \
                      --ground-truths ../../../data/ground_truth/kondo1/0020.png \
                      --gui -o /tmp/kondo1.csv.gz --penumbra-as-shadow \
                      --posterize 33 --distance 3 --ignore-objects
    --inputs: List of input images
    --ground-truths: Corresponding ground truth images (should be in same order)
    -o: Output file
    --penumbra-as-shadow: Remove distinction between 'penumbra' and 'shadow'
                          class in ground truth, treat all penumbra as shadow.
    --posterize: Spatial/Colour radius for Mean-Shift Segmentation function.
    --distance: Minimum pixel distance between consecutive points in contours.
    --ignore-objects: Treat 'object' class in ground truth as 'background'.

This will process a single image from the kondo1 image set. With the --gui
switch, it'll display two windows briefly showing the input image at various
stages of processing. When the program is done, the output should be in
'/tmp/kondo1.csv.gz'. This can be opened directly in Weka.


All Python code is written to be compatible with either Python 3 or Python 2.
Some of the libraries it use behave incorrectly on Python3 however. For best
results use Python2.

Due to the way it's written, the test harness will only run in a *nix
environment as it makes calls to the shell and assumes Bash is installed. This
is for the purposes of 'parallel mode'.
