/**
 * @file ground_truth.cpp
 * @brief Given a list of inputs and ground truths, compare the values of the
 * inputs with the values of the ground truth.
 * (Ground truth and input must be the same size and use the same values to
 * represent shadow, penumbra, foreground object, background and unknown)
 * Produces two files in output_dir -- roc.csv and confusion.csv.
 * @author Owain Jones <odj@aber.ac.uk>
 * @version 0.1
 * @date 2014-02-14 (Happy Valentine's Day!)
 */
#define PROGRAM_NAME "ground_truth"

#include "utils/utils.cpp"

using namespace std;
using namespace cv;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

string output_file;
vector<string> parameters_list;
vector<string> ground_truths;
bool replace_penumbra = false;


/**
 * @brief Structure to hold data for constructing a confusion matrix with.
 */
class Confusion{
    public:
        int predicted;
        int actual;
};


/**
 * @brief Structure to hold true/false positive/negative counts for anything.
 */
class ROC {
    public:
        double true_positive;
        double true_negative;
        double false_positive;
        double false_negative;

        double positive() {
            return this->true_positive + this->false_positive;
        }

        double negative() {
            return this->true_negative + this->false_negative;
        }

        double total() {
            return this->positive() + this->negative();
        }

        /* Following functions are using the derivations listed in
         * "Terminology and derivations from a confusion matrix", from
         * http://en.wikipedia.org/wiki/Receiver_operating_characteristic
         */

        double sensitivity() {  // AKA true positive rate
            return this->true_positive /
                   (this->true_positive + this->false_negative);
        };

        double specifity() {  // AKA true negative rate
            return this->true_negative /
                   (this->false_positive + this->true_negative);
        };

        double precision() {  // AKA positive predictive value
            return this->true_positive /
                   this->positive();
        };

        double negative_predictive_value() {
            return this->true_negative /
                   this->negative();
        };

        double fallout() {  // AKA false positive rate
            return this->false_positive /
                   (this->false_positive + this->true_negative);
        };

        double false_discovery_rate() {
            return 1.0 - this->precision();
        };

        double miss_rate() {  // AKA false negative rate
            return this->false_negative /
                   (this->false_negative + this->true_positive);
        };

        double accuracy() {
            return (this->true_positive + this->true_negative) /
                   this->total();
        };

        double f1() {
            return (2 * this->true_positive) /
                   ((2 * this->true_positive)
                            + this->false_positive + this->false_negative);
        }

        /*double matthews() {
         * TODO: be less lazy and finish this one maybe
            return ((this->true_positive * this->true_negative)
                            - (this->false_positive * this->false_negative)) /
                   ()
        }*/

        double informedness() {
            return this->sensitivity() + this->specifity() - 1.0;
        }

        double markedness() {
            return this->precision() + this->negative_predictive_value() - 1.0;
        }
};

/*const char* confusion_header =
    "Chain,Image Set,Input,Ground Truth,"\
    "T,P,N,TP,TN,FP,FN,Sensitivity,Specifity,PPV,NPV,FPR,FDR,FNR,Accuracy"; */
const char* roc_header =
    "Chain,Parameters,Image Set,Input,Ground Truth,"\
    "Shadow Total,Shadow P,Shadow N,Shadow TP,Shadow TN,Shadow FP,Shadow FN,"\
    "Shadow TPR,Shadow FPR,"\
    "Object Total,Object P,Object N,Object TP,Object TN,Object FP,Object FN,"\
    "Object TPR,Object FPR,"\
    "Background Total,Background P,Background N,Background TP,Background TN,"\
    "Background FP,Background FN,Background TPR,Background FPR,"\
    "Penumbra Total,Penumbra P,Penumbra N,Penumbra TP,Penumbra TN,"\
    "Penumbra FP,Penumbra FN,Penumbra TPR,Penumbra FPR";

ostream& operator<<(ostream &strm, ROC &c) {
    /*return strm << c.total() << ',' << c.positive() << ',' << c.negative() << ','
                << c.true_positive << ',' << c.true_negative << ','
                << c.false_positive << ',' << c.false_negative << ','
                << c.sensitivity() << ',' << c.specifity() << ','
                << c.precision() << ','
                << c.negative_predictive_value() << ','
                << c.fallout() << ','
                << c.false_discovery_rate() << ','
                << c.miss_rate() << ','
                << c.accuracy();*/
    return strm << c.total() << ',' << c.positive() << ',' << c.negative() << ','
                << c.true_positive << ',' << c.true_negative << ','
                << c.false_positive << ',' << c.false_negative << ','
                << c.sensitivity() << ',' << (1.0 - c.specifity());
}

const char* confusion_header = "Predicted,Actual";

ostream& operator<<(ostream &strm, Confusion &c) {
    return strm << c.predicted << ',' << c.actual;
}


/**
 * @brief Structure to hold the true/false positive/negative counts for each
 * shadow/object classification when an image is compared with its ground
 * truth.
 */
class Statistics {
    public:
        ROC objects;
        ROC shadows;
        ROC penumbras;
        ROC background;
        vector<Confusion> confusions;
};


/**
 * @brief Fixes ground truth image. Any pixel values not belonging to shadow,
 * penumbra, object or background classes are replaced with a preset value. By
 * default, anything that != white, != #99999 / value 153, != black, is
 * replaced with black, signifying shadow.
 *
 * @param original Original ground truth image.
 * @param replace_value Value to replace unknown values with. (Default is black
 * aka shadow class)
 * @param replace_penumbra If true, replace penumbra value with replace_value.
 *
 * @return Copy of the ground truth image with updated values.
 */
Mat fix_ground_truth(const cv::Mat original,
                     uchar replace_value=C_SHADOW,
                     bool replace_penumbra=false) {
    cv::Mat updated;
    original.copyTo(updated);
    uchar iv;

    for(int i=0; i<updated.rows; i++) {
        for(int j=0; j<updated.cols; j++) {
            iv = updated.at<uchar>(i, j);
            if(iv != C_OBJECT && iv != C_BACKGROUND
               && iv != C_PENUMBRA && iv != C_SHADOW) {
                updated.at<uchar>(i, j) = replace_value;
            }
            if(replace_penumbra == true && iv == C_PENUMBRA) {
                updated.at<uchar>(i, j) = replace_value;
            }
        }
    }

    return updated;
}


/**
 * @brief Compare an image with its ground truth. Both images must be greyscale
 * and an 8-bit uchar array.
 *
 * @param image The Mat array containing classification values
 * @param ground_truth The Mat array containing ground truth values
 * @param shadow_value The value which should be considered positive for
 *        shadows.
 * @param penumbra_value Positive value for penumbras.
 * @param object_value Positive value for objects.
 * @param background_value Positive value for background.
 *
 * @return A Statistics object storing the true/false positive/negative
 * counts for shadows, objects, penumbras and backgrounds.
 */
Statistics compare_images(const cv::Mat image, const cv::Mat ground_truth,
                          uchar shadow_value=C_SHADOW,
                          uchar penumbra_value=C_PENUMBRA,
                          uchar object_value=C_OBJECT,
                          uchar background_value=C_BACKGROUND) {
    Statistics s = Statistics();

    /* Weird bug with some PNG images means that the last two rows and columns
     * are inaccessible (cause the program to segfault). Simplest fix was to
     * ignore a 2-pixel border of the image. */
    uchar iv, gt;
    for(int i=0; i<image.rows; i++) {
        for(int j=0; j<image.cols; j++) {
            iv = image.at<uchar>(i, j);
            gt = ground_truth.at<uchar>(i, j);
            Confusion c = Confusion();
            c.actual = gt;
            c.predicted = iv;

            s.confusions.push_back(c);

            // Shadows
            if(iv == shadow_value && gt == shadow_value)
                s.shadows.true_positive += 1;
            if(iv != shadow_value && gt != shadow_value)
                s.shadows.true_negative += 1;
            if(iv == shadow_value && gt != shadow_value)
                s.shadows.false_positive += 1;
            if(iv != shadow_value && gt == shadow_value)
                s.shadows.false_negative += 1;

            // Penumbras
            if(iv == penumbra_value && gt == penumbra_value)
                s.penumbras.true_positive += 1;
            if(iv != penumbra_value && gt != penumbra_value)
                s.penumbras.true_negative += 1;
            if(iv == penumbra_value && gt != penumbra_value)
                s.penumbras.false_positive += 1;
            if(iv != penumbra_value && gt == penumbra_value)
                s.penumbras.false_negative += 1;


            // Objects
            if(iv == object_value && gt == object_value)
                s.objects.true_positive += 1;
            if(iv != object_value && gt != object_value)
                s.objects.true_negative += 1;
            if(iv == object_value && gt != object_value)
                s.objects.false_positive += 1;
            if(iv != object_value && gt == object_value)
                s.objects.false_positive += 1;

            // Background
            if(iv == background_value && gt == background_value)
                s.background.true_positive += 1;
            if(iv != background_value && gt != background_value)
                s.background.true_negative += 1;
            if(iv == background_value && gt != background_value)
                s.background.false_positive += 1;
            if(iv != background_value && gt == background_value)
                s.background.false_negative += 1;

        }
    }
    return s;
}


/**
 * @brief Takes a sequence of images, does work on them, and spits them out into
 * an output folder specified by --output.
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

    add_parameter(string, output_file);
    add_parameter_multi(vector<string>, parameters_list);
    add_parameter_multi(vector<string>, ground_truths);
    add_parameter(bool, replace_penumbra);
    parse_commands();

    Mat input;
    Mat ground_truth;
    ofstream roc;
    ofstream confusion;
    Statistics stats;
    stringstream confusion_path;
    confusion_path << output_dir.c_str() << "/confusion.csv";
    
    roc.open(output_file.c_str());
    roc.precision(32);
    roc << roc_header;
    for(vector<string>::size_type i = 0; i != parameters_list.size(); i++) {
        if(i % 2 == 0) {
            roc << ',' << parameters_list[i];
        }
    }
    roc << endl;

    for(vector<string>::size_type i = 0; i != inputs.size(); i++) {
        if(verbose) {
            cout << "input=" << inputs[i] << ';' << endl;
            cout << "ground_truth=" << ground_truths[i] << ';' << endl;
        }
        input = imread(inputs[i], CV_LOAD_IMAGE_GRAYSCALE);
        ground_truth = imread(ground_truths[i], CV_LOAD_IMAGE_GRAYSCALE);
        if(verbose) cout << "  opened both images successfully." << endl;

        // Make sure both images are definitely 8-bit uchar matrices
        input.convertTo(input, CV_8U);
        ground_truth.convertTo(ground_truth, CV_8U);
        if(verbose) cout << "  converted both to 8-bit greyscale." << endl;

        // Fix ground truth
        ground_truth = fix_ground_truth(ground_truth, C_SHADOW,
                                        replace_penumbra);
        if(verbose) cout << "  fixed ground truth image." << endl;

        if((input.rows != ground_truth.rows)
            || (input.cols != ground_truth.cols)) {
            if(verbose) cerr << "  images are not same dimensions!" << endl;
        }

        stats = compare_images(input, ground_truth);
        if(verbose) cout << "  collected stats on both." << endl;

        roc << info_chain << ',' << info_parameters << ',' << info_image_set << ',';
        roc << inputs[i] << ',' << ground_truths[i] << ',';
        roc << stats.shadows << ',' << stats.objects << ',';
        roc << stats.background << ',' << stats.penumbras;
        
        for(vector<string>::size_type x = 0; x != parameters_list.size(); x++) {
            if(x % 2 == 1) {
                roc << ',' << parameters_list[x];
            }
        }

        roc << endl;
        if(verbose) cout << "  wrote rows to ROC and Confusion file." << endl;
    }

    roc.close();
    return 0;
}


// vim: set tabstop=4 shiftwidth=4 textwidth=79 cc=72, 79:
