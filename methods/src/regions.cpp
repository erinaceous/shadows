// vim: set tabstop=4 shiftwidth=4 textwidth=79 cc=72, 79:
/**
 * @file regions.cpp
 * @brief Using a local regions approach, checks the pixels surrounding each
 * pixel for changes in hue (edge of an object?) or changes in value (edge of
 * a shadow?). Sliding window size is an adjustable parameter, as is colour
 * space used, and thresholds for edge detections.
 * @author Owain Jones <odj@aber.ac.uk>
 * @version 0.1
 * @date 2014-03-13
 */
// NOTE: Uses some C++11 features (chrono stuff is Cxx11?)
//       so when compiling, make sure -std=c++11
#define PROGRAM_NAME "regions"
#include "utils/utils.cpp"

using namespace std;
using namespace cv;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

bool denoise = false;
string thresh_type = "none";
int beta1 = 10;
int beta2 = 10;
int thresh_hue = 10; // minimum difference between respective pixels
int thresh_sat = 10; // " "
int input_blur = 0;
int window_size = 3; // 3x3 window by default.
float rescale = 1.0;


Mat regions(const Mat input, int beta1, int beta2,
            int thresh_sat, int thresh_hue, int window_size, int blur=0,
            string thresh_type="none", bool denoise=false) {

    // Extract HSV channels for input image, blurring / denoising the
    // image in the process.
    Mat hsv;
    vector<Mat> hsv_channels;
    input.copyTo(hsv);
    Mat hue(hsv.rows, hsv.cols, CV_8UC1);
    Mat saturation(hsv.rows, hsv.cols, CV_8UC1);
    Mat value(hsv.rows, hsv.cols, CV_8UC1);
    cvtColor(input, hsv, CV_BGR2HSV);

    if(denoise) {
        fastNlMeansDenoisingColored(hsv, hsv);
    }
    if(blur > 0) {
        GaussianBlur(hsv, hsv, Size(blur, blur), 0, 0);
    }

    split(hsv, hsv_channels);
    hue = hsv_channels[0];
    saturation = hsv_channels[1];
    value = hsv_channels[2];

    if(thresh_type == "otsu") {
        threshold(value, value, 0, 255, THRESH_TOZERO | THRESH_OTSU);
    } else if(thresh_type == "adaptive") {
        adaptiveThreshold(value, value, 255, ADAPTIVE_THRESH_GAUSSIAN_C,
                          THRESH_BINARY, 11, 0);
    }

    if(blur > 0 && thresh_type != "none") {
        GaussianBlur(value, value, Size(blur, blur), 0, 0);
    }

    // Initialize shadow, object masks and output image
    Mat output(value.rows, value.cols, CV_8UC1);
    Mat objects(value.rows, value.cols, CV_8UC1);
    Mat shadows(value.rows, value.cols, CV_8UC1);

    // Work out where the center of the sliding window should be
    int half_window;
    if(window_size % 2) {  // if window_size is an odd number
        half_window = 1 + (window_size / 2);
    } else {
        half_window = window_size / 2;
    }

    for(int y=0; y < hue.rows; y++) {
        for(int x=0; x < hue.cols; x++) {

            int x_range_start = x - half_window, x_range_end = x + half_window,
                y_range_start = y - half_window, y_range_end = y + half_window;
            if(x_range_start < 0) x_range_start = 0;
            if(y_range_start < 0) y_range_start = 0;
            if(x_range_end >= hue.cols) x_range_end = hue.cols - 1;
            if(y_range_end >= hue.rows) y_range_end = hue.rows - 1;

            Mat value_window = value(Range(y_range_start, y_range_end),
                                     Range(x_range_start, x_range_end));
            Mat sat_window = saturation(Range(y_range_start, y_range_end),
                                        Range(x_range_start, x_range_end));
            Mat hue_window = hue(Range(y_range_start, y_range_end),
                                 Range(x_range_start, x_range_end));

            uchar center_value = value.at<uchar>(y, x);
            uchar center_sat = saturation.at<uchar>(y, x);
            uchar center_hue = hue.at<uchar>(y, x);

            uchar frame_value = mean(value_window).val[0];
            uchar frame_sat = mean(sat_window).val[0];
            uchar frame_hue = mean(hue_window).val[0];

            float fdc = ((float)frame_value / (float)center_value) * 255;
            //float fdc = ((float)center_value / (float)frame_value) * 100;
 
            if(((beta1 <= fdc) && (fdc <= beta2)) &&
               (abs(frame_sat - center_sat) <= thresh_sat) &&
               (abs(frame_hue - center_hue) <= thresh_hue)) {
                shadows.at<uchar>(y, x) = 255;
            }
        }
    }

    /*vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(shadows, contours, hierarchy, CV_RETR_TREE,
                 CV_CHAIN_APPROX_SIMPLE, Point(-1, -1));

    bitwise_or(shadows.row(0), shadows.row(1), shadows.row(1));
    bitwise_or(shadows.col(0), shadows.col(1), shadows.col(1));
    bitwise_or(shadows.row(shadows.rows - 1), shadows.row(shadows.rows - 2),
               shadows.row(shadows.rows - 2));
    bitwise_or(shadows.col(shadows.cols - 1), shadows.col(shadows.cols - 2),
               shadows.col(shadows.cols - 2));

    vector<Point> approxShape;
    for(size_t i=0; i < contours.size(); i++) {
        //approxPolyDP(contours[i], approxShape,
        //             arcLength(Mat(contours[i]), true) * 0.04, true);
        drawContours(shadows, contours, i, Scalar(255), CV_FILLED);
    }*/

    //shadows = 255 - shadows;

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

    add_parameter(bool, denoise);
    add_parameter(string, thresh_type);
    add_parameter(int, beta1);
    add_parameter(int, beta2);
    add_parameter(int, thresh_hue);
    add_parameter(int, thresh_sat);
    add_parameter(int, input_blur);
    add_parameter(int, window_size);
    add_parameter(float, rescale);
    parse_commands();

    init_timers();

    for_each_string_in_list(inputs, input_path) {
        Mat input = imread(*input_path);
        string output_path = to_png_filename(*input_path, output_dir);

        if(!input.data) {
            log_e("Failed to load " << *input_path << endl);
            continue;
        }

        log_v("Detecting shadow regions for " << *input_path);

        int input_w = input.cols, input_h = input.rows;
        if(rescale < 1.0) {
            resize(input, input, Size(input_w * rescale, input_h * rescale),
                   0, 0, INTER_AREA);
        }

        Mat output = regions(input, beta1, beta2,
                         thresh_sat, thresh_hue, window_size, input_blur,
                         thresh_type, denoise);
        
        if(rescale < 1.0) {
            resize(output, output, Size(input_w, input_h),
                   0, 0, INTER_NEAREST);
        }

        timing_file_log();

        imwrite_png(output_path, output);

        log_v(" done.\n");
        show_gui(input, output, inputs, input_path);
    }

    timing_file_end();
    return 0;
}
