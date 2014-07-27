// vim: set tabstop=4 shiftwidth=4 textwidth=79 cc=72, 79:
/**
 * @file contours.cpp
 * @brief Simple contours-based approach that uses the saturation of
 * an image as well as its value -- as shadows are often less saturated
 * than their surroundings.
 * @author Owain Jones <odj@aber.ac.uk>
 * @version 0.1
 * @date 2014-03-19
 */
// NOTE: Uses some C++11 features (chrono stuff is Cxx11?)
//       so when compiling, make sure -std=c++11
#define PROGRAM_NAME "contours"

#include "utils/utils.cpp"

using namespace std;
using namespace cv;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

string thresh_type = "none";
int value_difference = 1;
int hue_difference = 1;
int sat_difference = 1;
int input_blur = 0;
int window_size = 3;


Mat contours(const Mat input, int value_difference, int hue_difference,
             int sat_difference, int window_size=3, int blur=0) {

    // Extract HSV channels for input image
    Mat hsv;
    cvtColor(input, hsv, CV_BGR2HSV);

    // Denoise and blur image if applicable
    if(blur > 0) {
        GaussianBlur(hsv, hsv, Size(blur, blur), 0, 0);
    }

    // Initialize shadow, object masks and output image
    Mat output(hsv.rows, hsv.cols, CV_8UC3);
    Mat objects(hsv.rows, hsv.cols, CV_8UC1);
    Mat shadows(hsv.rows, hsv.cols, CV_8UC1);
    Mat edges(hsv.rows, hsv.cols, CV_8UC1);

    // Figure out where the center of sliding window should be.
    int half_window;
    if(window_size % 2) {
        half_window = 1 + (window_size / 2);
    } else {
        half_window = window_size / 2;
    }

    Scalar averages = mean(hsv);

    for(int i=0; i < hsv.rows; i++) {
        for(int j=0; j < hsv.cols; j++) {
            int i_range_start = i - half_window, i_range_end = i + half_window,
                j_range_start = j - half_window, j_range_end = j + half_window;
            if(i_range_start < 0) i_range_start = 0;
            if(i_range_end > hsv.rows - 1) i_range_end = hsv.rows - 1;
            if(j_range_start < 0) j_range_start = 0;
            if(j_range_end > hsv.cols - 1) j_range_end = hsv.cols - 1;
            int i_range = i_range_end - i_range_start,
                j_range = j_range_end - j_range_start;

            Mat window(hsv, Rect(
                j_range_start, i_range_start, j_range, i_range            
            ));

            Scalar frame = mean(window);
            Vec<uchar, 3> center = hsv.at<Vec<uchar, 3> >(i, j);

            //float fdc = ((float)frame[2] / (float)center[2]) * 100;
            //int fdc = abs(frame[2] - center[2]);

            if((frame[2] <= averages[2]) &&
               (abs(frame[1] - center[1]) >= sat_difference) &&
               (abs(frame[0] - center[0]) >= hue_difference)) {
                shadows.at<uchar>(i, j) = 255;
            }

            //hsv.at<Vec<uchar, 3> >(i, j) = Vec<uchar, 3>(255, i, j);
        }
    }

    //shadows.at<uchar>(10, 10) = 255;
    //shadows.at<uchar>(15, 15) = 255;
    //shadows.at<uchar>(200, 10) = 255;

    //int from_to[] = {0, 0};
    //mixChannels(&hsv, 3, &shadows, 1, from_to, 1);

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

    add_parameter(int, value_difference);
    add_parameter(int, hue_difference);
    add_parameter(int, sat_difference);
    add_parameter(int, input_blur);
    add_parameter(int, window_size);
    parse_commands();

    init_timers();

    for_each_string_in_list(inputs, i) {
        Mat input = imread(*i);
        string output_path = to_png_filename(*i, output_dir);

        if(!input.data) {
            log_e("Failed to load " << *i << endl);
            continue;
        }

        log_v("Detecting shadow contours for " << *i);

        Mat output = contours(input, value_difference, hue_difference,
                              sat_difference, window_size, input_blur);

        timing_file_log();
        
        imwrite_png(output_path, output);

        log_v(" done.\n");
        show_gui(input, output, inputs, i);
    }

    timing_file_end();

    return 0;
}
