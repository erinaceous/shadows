/**
 * @file edges.cpp
 * @brief Detect potential shadow and object edges. Then detect those contours,
 * and figure out where the insides and outsides of shadows/objects are by
 * moving a sliding window over the contour and comparing each side of the
 * edge. Use this information to fill in areas with shadows etc.
 * @author Owain Jones <odj@aber.ac.uk>
 * @version 0.1
 * @date 2014-03-25
 */
// NOTE: Uses some C++11 features (chrono stuff is Cxx11?)
//       so when compiling, make sure -std=c++11
#define PROGRAM_NAME "edges"
#include "utils/utils.cpp"

using namespace std;
using namespace cv;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

int kernel_size = 3;
int input_blur = 0;
double canny_threshold = 1.0;
double canny_ratio = 1.0;
bool equalize = false;


Mat edges(const Mat input, double threshold, double ratio,
          int kernel_size, int blur, bool equalize) {

    // Extract HSV channels for input image
    Mat hsv, hue, saturation, value, hue_sat;
    vector<Mat> hsv_channels;
    input.copyTo(hsv);
    cvtColor(hsv, hsv, CV_BGR2HSV);

    // Blur image if applicable
    if(blur > 0) {
        GaussianBlur(hsv, hsv, Size(blur, blur), 0, 0);
    }

    // Split into Hue, Sat, Value channels
    split(hsv, hsv_channels);
    hue = hsv_channels[0];
    saturation = hsv_channels[1];
    value = hsv_channels[2];
    hue.copyTo(hue_sat);
    hue_sat = (hue + saturation) / 2.0;

    // Equalize histograms of channels independently, if applicable
    if(equalize) {
        equalizeHist(hue, hue);
        equalizeHist(saturation, saturation);
        equalizeHist(value, value);
    }

    // Initialize shadow, object masks and output image
    Mat output(hue.rows, hue.cols, CV_8UC1);
    Mat objects(hue.rows, hue.cols, CV_8UC1);
    Mat shadows(hue.rows, hue.cols, CV_8UC1);

    // Do Canny edge detection
    Canny(value, shadows, threshold, threshold*ratio, kernel_size);
    Canny(saturation, objects, threshold, threshold*ratio, kernel_size);

    /*for(int x=0; x < hue.cols; x++) {
        for(int y=0; y < hue.rows; y++) {

            uchar pixel_value = value.at<uchar>(y, x);
            uchar pixel_sat = saturation.at<uchar>(y, x);
            //uchar pixel_hue = hue.at<uchar>(y, x);

            if((pixel_value <= thresh_value) &&
               (pixel_sat <= thresh_sat)) {
                shadows.at<uchar>(y, x) = 255;
            }
        }
    }*/

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(shadows, contours, hierarchy, CV_RETR_TREE,
                 CV_CHAIN_APPROX_SIMPLE, Point(-1, -1));
    Mat markers;
    shadows.copyTo(markers);
    Mat approxShape;
    for(size_t i=0; i < contours.size(); i++) {
        approxPolyDP(contours[i], approxShape,
                     arcLength(Mat(contours[i]), true) * 0.04, true);
        drawContours(shadows, contours, i, Scalar(255), CV_FILLED);
    }
    /*findContours(objects, contours, hierarchy, CV_RETR_TREE,
                 CV_CHAIN_APPROX_SIMPLE, Point(-1, -1));*/
    //watershed(shadows, approxShape);


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

    add_parameter(int, input_blur);
    add_parameter(double, canny_threshold);
    add_parameter(double, canny_ratio);
    add_parameter(int, kernel_size);
    add_parameter(bool, equalize);
    parse_commands();

    init_timers();

    for(vector<string>::iterator i = inputs.begin(); i < inputs.end(); i++) {
        Mat input = imread(*i);
        string output_path = to_png_filename(*i, output_dir);

        if(!input.data) {
            log_e("Failed to load " << *i << endl);
            continue;
        }
        log_v("Detecting shadow edges for " << *i);

        Mat output = edges(input, canny_threshold, canny_ratio,
                           kernel_size, input_blur, equalize);
        imwrite_png(output_path, output);

        log_v(" done." << endl);
        timing_file_log();
        show_gui(input, output, inputs, i);

    }

    timing_file_end();

    return 0;
}


// vim: set tabstop=4 shiftwidth=4 textwidth=79 cc=72, 79:
