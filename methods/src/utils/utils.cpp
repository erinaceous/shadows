/**
 * @file utils.cpp
 * @brief Utility functions and macros shared between my programs. Including
 * this file makes a bunch of global variables available, which command line
 * arguments are copied to.
 * It also defines some helpful macros to reduce 'boiler plate' code. I don't
 * know if this is a good idea or bad C++ practise since I'm still a newbie to
 * C/C++ style programming. It's certainly made it easier for me to read my own
 * files.
 *
 * Since this initializes things for you, the main() function for a program is
 * as simple as:
 *
 * void main(int argc, char** argv) {
 *      utils_init();
 *      add_parameter(int, example_parameter_1);
 *      add_parameter(int, example_parameter_2);
 *      add_parameter_multi(vector<string>, multi_token_option);
 *      parse_commands();
 *      init_timers();
 *
 *      for_each_input_file_as(input_path) {
 *          Mat input = imread(*input_path);
 *          string output_path = to_png_filename(*input_path, output_dir);
 *          if(!input.data) {
 *              cerr << "Failed to load " << *input_path << endl;
 *              continue;
 *          }
 *          Mat output = image_processing_function(
 *              input, example_parameter_1, example_parameter_2
 *          );
 *          timing_file_log();
 *          imwrite_png(output_path, output);
 *          show_gui();  // if "--gui yes" specified on command line, show input
 *                       // and output images
 *      }
 *      timing_file_end();
 *      return 0;
 * }
 *
 *
 * @author Owain Jones <odj@aber.ac.uk>
 * @version 0.1
 * @date 2014-03-12
 */

#include "shadows.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

using namespace std;
namespace fs = boost::filesystem;
namespace ch = std::chrono;
namespace po = boost::program_options;


po::options_description desc("Options");
bool gui = false;
bool gui_loop = false;
bool gui_wait = false;
bool test_mode = false;
string output_dir;
string info_image_set;
string info_chain;
string info_parameters;
vector<string> inputs;
bool verbose = false;


/** All my programs are going to be run in parallel - and that's handled by
 * the external Python script. Having 8 threads per program trips over the
 * 8 programs running at a time and generally doesn't help; on my laptop at
 * least, timing tests suggest that 8 processes are better than 8 threads --
 * I'm guessing due to waiting on IO or something.
 * So all the programs that run this function will have their OpenCV code
 * ran on a single thread only. **/
#define utils_init() cv::setNumThreads(0)


/**
 * @brief Creates an output stream for a file. If that file exists, then
 * simply returns it. If it doesn't exist yet, opens the file, and if
 * you've specified anything in the header argument, appends that to
 * the file, then returns it.
 * Always opens files in output + append mode.
 *
 * @param path Path to file to be opened
 * @param header Optional header string to append to file if it is new
 *
 * @return Pointer to an output stream that can be written to.
 */
ofstream* get_file(const char* path, const char* header=NULL) {
    if(fs::exists(path)) {
        ofstream* got_file =
            new ofstream(path, ofstream::out | ofstream::app);
        return got_file;
    }
    ofstream* got_file = new ofstream(path, ofstream::out | ofstream::app);
    if(header != NULL) {
        *got_file << header << endl;
    }
    return got_file;
}


/**
 * @brief Uses the get_file function to access a file / open a new one,
 * appends a header suitable for a timing.csv file if the file is new.
 *
 * @param path Path to the file
 *
 * @return Pointer to an output stream that can be written to.
 */
ofstream* get_timing_file(const char* path) {
    return get_file(
        path,
        "Chain, Image Set, Parameters, Program, Input, Time"
    );
}


// Macros for timing information
#define get_timing_file(path) \
get_file(path, "Chain, Image Set, Parameters, Program, Input, Time")

// Put this at the start of your main()
#define init_timers() \
std::stringstream timing_path; \
timing_path << output_dir.c_str() << "/timing.csv"; \
ofstream* timing = get_timing_file(timing_path.str().c_str()); \
chrono::steady_clock::time_point time_curr = std::chrono::steady_clock::now(); \
chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();

#define timing_file_write(output_path, time_last, time_now) \
*timing << info_chain.c_str() << ", " << info_image_set.c_str() << ", "; \
*timing << info_parameters.c_str() << ", "; \
*timing << PROGRAM_NAME << ", " << output_path << ", "; \
*timing << chrono::duration_cast<chrono::microseconds>(time_now - time_last).count(); \
*timing << endl;

// Put this somewhere in your image processing loop -- after you've processed an
// image is probably a good place
#define timing_file_log() \
time_end = std::chrono::steady_clock::now(); \
timing_file_write(output_path, time_curr, time_end) \
time_curr = std::chrono::steady_clock::now()

// Put this after your loop
#define timing_file_end() \
/*timing_file_write("total", time_start, time_end);*/ \
timing->close()


// Shortcuts for adding parameters to a Boost program_options::options_description
#define add_parameter(type, name)\
desc.add_options()(#name, po::value<type>(&name))

#define add_parameter_multi(type, name)\
desc.add_options()(#name, po::value<type>(&name)->multitoken())

#define parse_commands() \
    po::variables_map options = parse_command_line(desc, argc, argv);


/**
 * @brief Parses command line using options_description passed by user, adding
 * default options used by all programs along the way.
 *
 * @param desc An initialized options_description containing the extra arguments
 * your program requires
 * @param argc
 * @param argv
 *
 * @return A variables_map with the parsed command line
 */
po::variables_map parse_command_line(po::options_description desc,
                                     int argc, char** argv) {
    desc.add_options()("help", "Print help message");
    add_parameter(bool, test_mode);
    add_parameter(bool, verbose);
    add_parameter(bool, gui);
    add_parameter(bool, gui_loop);
    add_parameter(bool, gui_wait);
    add_parameter(string, info_image_set);
    add_parameter(string, info_chain);
    add_parameter(string, info_parameters);
    add_parameter(string, output_dir);
    add_parameter_multi(vector<string>, inputs);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc)
            .allow_unregistered().run(), vm);

    if(vm.count("help")) {
        cout << desc << endl;
    }

    if(vm.count("test_mode")) {
        verbose = true;
        gui = true;
        gui_loop = true;
        gui_wait = true;
        stringstream sb;
        sb << "/tmp/" << PROGRAM_NAME;
        output_dir = sb.str();
    }

    po::notify(vm);

    return vm;
}


/**
 * @brief Take a filename from anywhere and replace its extension with .png and
 * say that it's in the directory specified by output_dir.
 *
 * @param input_path Path to input file
 * @param output_dir Directory to prepend filename with
 *
 * @return 
 */
string to_png_filename(string input_path, string output_dir) {
    stringstream output_path;
    fs::path filepath(input_path);
    output_path << output_dir << '/' << fs::basename(input_path) << ".png";
    return output_path.str();
}


/**
 * @brief Write a file to output_dir/ with the .png extension.
 * Uses opencv's imwrite function internally, configured to use the best PNG
 * compression available.
 *
 * @param input_path Path to input file (used for base filename)
 * @param output_dir Directory in which to save file
 * @param output Mat containing image data
 */
void imwrite_png(string path, cv::Mat output) {
    vector<int> imwrite_params;
    imwrite_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    imwrite_params.push_back(9);
    cv::imwrite(path, output, imwrite_params);
}


#define show_gui(input, output, file_list, current_input_path) \
if(gui) { \
    imshow("input", input); \
    imshow("output", output); \
    if(gui_wait) { \
        waitKey(0); \
    } else { \
        waitKey(1); \
    } \
    if(gui_loop) { \
        if(current_input_path >= file_list.end() - 1) { \
           current_input_path = file_list.begin() - 1; \
        } \
    } \
}


#define for_each_string_in_list(l, i) \
for(std::vector<std::string>::iterator i = l.begin(); i < l.end(); i++)


#define _log(os, msg)\
if(verbose) {\
    os << msg;\
    os.flush();\
}

#define log_v(msg)\
_log(std::cout, msg)

#define log_e(msg)\
_log(std::cerr, msg)
