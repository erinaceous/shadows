// vim: set tabstop=4 shiftwidth=4 textwidth=79 cc=72, 79:
/**
 * @file simple_threshold.cpp
 * @brief Simple thresholding of images in a grayscale image: Anything lower
 * than --threshold in the input images is considered shadow, anything higher
 * than --object in the input images is considered a foreground object. The rest
 * is background.
 * Objects are grey (value 153)
 * Shadows are black (value 0)
 * Background is white (value 255)
 *
 * @author Owain Jones <odj@aber.ac.uk>
 * @version 0.1
 * @date 2014-02-12
 */
// NOTE: Uses some C++11 features (chrono stuff is Cxx11?)
//       so when compiling, make sure -std=c++11
#define PROGRAM_NAME "simple_threshold"

#include "utils/utils.cpp"

using namespace std;
using namespace cv;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

int shadow = DEFAULT_SHADOW_THRESHOLD;
int object = DEFAULT_OBJECT_THRESHOLD;
int input_blur = 0;
string colour_space = "grey";
bool otsu_mode = false;


/**
 * @brief Takes a greyscale (0-255) image and does simple binary thresholding on
 * it: any pixel values higher than object_threshold == object. Any pixel values
 * lower than shadow_threshold == shadow.
 *
 * @param image
 * @param shadow_threshold
 * @param object_threshold
 *
 * @return Mat array, same size as input image, RGB 8-bit.
 */
Mat simple_threshold(Mat image, int shadow_threshold, int object_threshold,
                     bool otsu_mode=false, int input_blur=0) {
    Mat grey, shadows, objects, output;
    image.copyTo(output);
    //cvtColor(image, grey, CV_BGR2GRAY);

    if(input_blur > 0) {
        GaussianBlur(image, image, Size(input_blur, input_blur), 0, 0);
    }

    if(otsu_mode) {
        threshold(image, shadows, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
        threshold(image, objects, 0, 255, THRESH_BINARY | THRESH_OTSU);
    } else {
        threshold(image, shadows, shadow_threshold, 255, THRESH_BINARY_INV);
        threshold(image, objects, object_threshold, 255, THRESH_BINARY);
    }

    output.setTo(SCALAR_C_BACKGROUND);
    output.setTo(SCALAR_C_OBJECT, objects);
    output.setTo(SCALAR_C_SHADOW, shadows);

    return output;
}


/**
 * @brief Takes a sequence of images, does work on them, and spits them out into
 * an output folder specified by --output, as PNG files.
 * (Assumes that output folder already exists and we have write permissions to
 * it, otherwise this program will just fail silently)
 *
 * @param argc
 * @param argv
 *
 * @return 
 */
int main(int argc, char** argv) {
    utils_init();

    add_parameter(int, shadow);
    add_parameter(int, object);
    add_parameter(int, input_blur);
    add_parameter(bool, otsu_mode);
    add_parameter(string, colour_space);
    parse_commands();

    init_timers();

    for_each_string_in_list(inputs, i) {
        Mat input = imread(*i);
        string output_path = to_png_filename(*i, output_dir);

        if(!input.data) {
            log_e("Failed to load " << *i << endl);
            continue;
        }

        Mat grey, hsv, output;
        vector<Mat> hsv_channels;

        if(colour_space == "hsv") {
            input.copyTo(hsv);
            cvtColor(input, hsv, CV_BGR2HSV);
            split(hsv, hsv_channels);
            grey = hsv_channels[2];  // the V of HSV is brightness
        } else if(colour_space == "xyz") {
            input.copyTo(hsv);
            cvtColor(input, hsv, CV_BGR2XYZ);
            split(hsv, hsv_channels);
            grey = hsv_channels[1];  // the Y of XYZ is luminance (brightness)
        } else if(colour_space == "lab") {
            input.copyTo(hsv);
            cvtColor(input, hsv, CV_BGR2Lab);
            split(hsv, hsv_channels);
            grey = hsv_channels[0];
        } else if(colour_space == "luv") {
            input.copyTo(hsv);
            cvtColor(input, hsv, CV_BGR2Luv);
            split(hsv, hsv_channels);
            grey = hsv_channels[0];  // TODO: find out if L* is correct to use
        } else if(colour_space == "hls") {
            input.copyTo(hsv);
            cvtColor(input, hsv, CV_BGR2HLS);
            split(hsv, hsv_channels);
            grey = hsv_channels[1];
        } else {
            cvtColor(input, grey, CV_BGR2GRAY);
        }

        output = simple_threshold(grey, shadow, object, otsu_mode, input_blur);
        timing_file_log();
        
        imwrite_png(output_path, output);

        show_gui(input, output, inputs, i);
    }

    timing_file_end();
    return 0;
}
