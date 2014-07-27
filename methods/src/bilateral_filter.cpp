// vim: set tabstop=4 shiftwidth=4 textwidth=79 cc=72, 79:
/**
 * @file bilateral_filter.cpp
 * @brief Apply a bilateral filter to images. Useful for removing noise from
 * natural images; removes some texture information also.
 * @author Owain Jones <odj@aber.ac.uk>
 * @version 0.1
 * @date 2014-03-19
 */
// NOTE: Uses some C++11 features (chrono stuff is Cxx11?)
//       so when compiling, make sure -std=c++11
#define PROGRAM_NAME "bilateralfilter"

#include "utils/utils.cpp"

using namespace std;
using namespace cv;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

int diameter = 5;
double sigma_color = 100.0;
double sigma_space = 100.0;
bool adaptive = false;
int kernel_size = 3;


Mat bilateral_filter(const Mat input, int diameter, double sigma_color,
                     double sigma_space, bool adaptive, int kernel_size) {

    Mat output;
    Mat tmp;
    input.copyTo(tmp);

    if(adaptive) {
        adaptiveBilateralFilter(tmp, output, Size(kernel_size, kernel_size),
                                sigma_space);
    } else {
        bilateralFilter(tmp, output, diameter, sigma_color, sigma_space);
    }

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

    add_parameter(int, diameter);
    add_parameter(double, sigma_color);
    add_parameter(double, sigma_space);
    add_parameter(bool, adaptive);
    add_parameter(int, kernel_size);    
    parse_commands();

    log_v("Starting...");

    init_timers();

    for_each_string_in_list(inputs, i) {
        Mat input = imread(*i);
        string output_path = to_png_filename(*i, output_dir);

        if(!input.data) {
            log_e("Failed to load " << *i << endl);
            continue;
        }
        log_v("Running bilateral filter on " << *i);

        Mat output = bilateral_filter(input, diameter, sigma_color,
                                      sigma_space, adaptive, kernel_size);

        timing_file_log();
        
        imwrite_png(output_path, output);

        log_v(" done.\n");
        show_gui(input, output, inputs, i);
    }

    timing_file_end();

    return 0;
}
