// vim: set tabstop=4 shiftwidth=4 textwidth=79 cc=72, 79:
/**
 * @file dumber_lalonde.cpp
 * @brief Attempt to follow the methodology presented by Lalonde et al. in
 * their paper, "Detecting Ground Shadows in Outdoor Consumer Photographs".
 * (The first step, anyway)
 * @author Owain Jones <odj@aber.ac.uk>
 * @version 0.1
 * @date 2014-04-02
 */
// NOTE: Uses some C++11 features (chrono stuff is Cxx11?)
//       so when compiling, make sure -std=c++11
#define PROGRAM_NAME "dumberlalonde"

#include "utils/utils.cpp"

using namespace std;
using namespace cv;
namespace po = boost::program_options;
namespace fs = boost::filesystem;


Mat gradient_boundaries(Mat input, int blur=23) {
    Mat mag, mag_32bit, mag_uint, Sx, Sy, tmp;
   
    if(blur > 0) { 
        GaussianBlur(input, tmp, Size(blur, blur), 0); // filter out noise
    } else {
        input.copyTo(tmp);
    }

    // Thankyou to http://stackoverflow.com/a/11157426 for the nice simple
    // explanation on how to calculate gradient magnitudes :)
    Sobel(tmp, Sx, CV_32F, 1, 0, 3);
    Sobel(tmp, Sy, CV_32F, 0, 1, 3);
    magnitude(Sx, Sy, mag);

    // "then apply the watershed segmentation algorithm on the gradient map"
    mag.convertTo(mag_uint, CV_8UC3);
    cvtColor(mag, mag_32bit, CV_BGR2GRAY);
    mag_32bit.convertTo(mag_32bit, CV_32S);
    watershed(mag_uint, mag_32bit);

    if(gui) {
        imshow("magnitude", mag);
        imshow("watershed", mag_uint);
    }

    return mag_uint;
}


Mat dumber_lalonde(const Mat input) {
    Mat hsv, output;

    // Convert to HSV colour space
    cvtColor(input, hsv, CV_BGR2HSV);

    // Calculate gradient boundaries first (lots of over-segmentation here)
    // IDEA OF MINE: Get boundaries at different "resolutions" aka blurriness,
    // and then get the mean of the combined images.
    // TODO: Replace this stuff with simple image pyramid downsampling --
    // cv2::pyrDown() and cv2::pyrUp()
    Mat boundary_image_mean = gradient_boundaries(hsv, 0);
    /* boundary_image_mean.convertTo(boundary_image_mean, CV_32FC3);
    Mat boundary_images[] = {
        gradient_boundaries(hsv, 5),
        gradient_boundaries(hsv, 15),
        gradient_boundaries(hsv, 45),
    };
    for(int i=2; i>0; i--) {
        Mat boundary_image;
        boundary_images[i].convertTo(boundary_image, CV_32FC3);
        // boundary_image_mean = min(boundary_image, boundary_image_mean);
        boundary_image_mean += boundary_image;
    }
    boundary_image_mean *= (1.0 / 3.0); */
    boundary_image_mean.convertTo(output, CV_8UC3);

    vector<Mat> channels;
    split(output, channels);

    // "we use the canny edge detector at 4 scales to account for blurry
    // shadow edges (sigma^2 = {1, 2, 4, 8}, with a high threshold set
    // empirically to 0.3
    // THIS DOESN'T WORK IN OPENCV. The window sizes have to be an odd number
    // (possibly multiple of 3) because gaussian matrix and all that.
    // Also I think their images were in floating point ranges (0.0 to 1.0)
    // rather than unsigned int, so must scale 0.3 to 255 * 0.3.
    // Also, this still doesn't work. I'm doing it my own way.
    // int sigma_2[] = { 1, 2, 4, 8 };
    // equalizeHist(channels[2], channels[2]);
    threshold(channels[2], channels[2], 20, 255, THRESH_TOZERO);
    double sigma_2[] = { 3, 5, 7 };
    double t1 = 255 * 0.0;
    double t2 = 255 * 0.3;
    Mat value_canny;
    for(int i=0; i<3; i++) {
        Mat value_canny;
        Canny(channels[2], value_canny, t1, t2, sigma_2[i]);
        channels[2] = max(channels[2], value_canny);
    }
    threshold(channels[2], channels[2], 1, 255, THRESH_BINARY);
    Canny(channels[2], channels[2], 0, 0.1);

    output.setTo(SCALAR_C_BACKGROUND);
    output.setTo(SCALAR_C_PENUMBRA, channels[2]);
    //merge(channels, output);
    
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

    parse_commands();

    init_timers();

    for_each_string_in_list(inputs, i) {
        Mat input = imread(*i);
        string output_path = to_png_filename(*i, output_dir);

        if(!input.data) {
            log_e("Failed to load " << *i << endl);
            continue;
        }
        log_v("Running vague version of Lalonde & co's method on " << *i);

        Mat output = dumber_lalonde(input);

        timing_file_log();
        
        imwrite_png(output_path, output);

        log_v(" done.\n");
        show_gui(input, output, inputs, i);
    }

    timing_file_end();

    return 0;
}
