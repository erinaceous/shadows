// vim: set tabstop=4 shiftwidth=4 textwidth=79 cc=72, 79:
/**
 * @file chromacity.cpp
 * @brief Simple chromacity-based approach that uses the saturation of
 * an image as well as its value -- as shadows are often less saturated
 * than their surroundings.
 * @author Owain Jones <odj@aber.ac.uk>
 * @version 0.1
 * @date 2014-03-19
 */
// NOTE: Uses some C++11 features (chrono stuff is Cxx11?)
//       so when compiling, make sure -std=c++11
#define PROGRAM_NAME "chromacity"

#include "utils/utils.cpp"

using namespace std;
using namespace cv;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

bool denoise = false;
string thresh_type = "none";
int thresh_value;
int thresh_hue = 10; // minimum difference between respective pixels
int thresh_sat = 10; // " "
int input_blur = 0;


Mat chromacity(const Mat input, int thresh_value,
               int thresh_sat, string thresh_type, int blur, bool denoise) {

    // Extract HSV channels for input image
    Mat hue, saturation, value, hsv;
    vector<Mat> hsv_channels;

    if(denoise) {
        fastNlMeansDenoisingColored(input, hsv);
    } else {
        input.copyTo(hsv);
    }
    cvtColor(hsv, hsv, CV_BGR2HSV);

    // Denoise and blur image if applicable
    if(blur > 0) {
        GaussianBlur(hsv, hsv, Size(blur, blur), 0, 0);
    }

    // Split into Hue, Sat, Value channels
    split(hsv, hsv_channels);
    hue = hsv_channels[0];
    saturation = hsv_channels[1];
    value = hsv_channels[2];

    // If an adaptive threshold type is enabled, apply it to the value
    // channel.
    if(thresh_type == "otsu") {
        threshold(value, value, 0, 255, THRESH_TOZERO | THRESH_OTSU);
    } else if(thresh_type == "adaptive") {
        adaptiveThreshold(value, value, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
                          THRESH_BINARY, 11, 0);
    }

    if(gui) {
        imshow("thresholded", value);
    }

    // Initialize shadow, object masks and output image
    Mat output(hue.rows, hue.cols, CV_8UC1);
    //Mat objects(hue.rows, hue.cols, CV_8UC1);
    Mat shadows(hue.rows, hue.cols, CV_8UC1);

    for(int x=0; x < hue.cols; x++) {
        for(int y=0; y < hue.rows; y++) {

            uchar pixel_value = value.at<uchar>(y, x);
            uchar pixel_sat = saturation.at<uchar>(y, x);
            //uchar pixel_hue = hue.at<uchar>(y, x);

            if((pixel_value <= thresh_value) &&
               (pixel_sat <= thresh_sat)) {
                shadows.at<uchar>(y, x) = 255;
            }
        }
    }

    output.setTo(SCALAR_C_BACKGROUND);
    //output.setTo(SCALAR_C_OBJECT, objects);
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

    add_parameter(bool, denoise);
    add_parameter(string, thresh_type);
    add_parameter(int, thresh_value);
    add_parameter(int, thresh_hue);
    add_parameter(int, thresh_sat);
    add_parameter(int, input_blur);
    parse_commands();

    init_timers();

    for_each_string_in_list(inputs, i) {
        Mat input = imread(*i);
        string output_path = to_png_filename(*i, output_dir);

        if(!input.data) {
            log_e("Failed to load " << *i << endl);
            continue;
        }
        log_v("Detecting shadow chromacity for " << *i);

        Mat output = chromacity(input, thresh_value, thresh_sat,
                                thresh_type, input_blur, denoise);

        timing_file_log();
        
        imwrite_png(output_path, output);

        log_v(" done.\n");
        show_gui(input, output, inputs, i);
    }

    timing_file_end();

    return 0;
}
