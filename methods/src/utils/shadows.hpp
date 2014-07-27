/**
 * @file shadows.hpp
 * @brief Header containing definitions for shadow/object/background things.
 * @author Owain Jones <odj@aber.ac.uk>
 * @version 0.1
 * @date 2014-02-12
 */

#include <opencv2/opencv.hpp>

#define DEFAULT_SHADOW_THRESHOLD    50
#define DEFAULT_PENUMBRA_THRESHOLD  55
#define DEFAULT_OBJECT_THRESHOLD    150

typedef enum pixelClass {
    C_SHADOW      = 0,
    C_PENUMBRA    = 25,
    C_OBJECT      = 153,
    C_UNKNOWN     = 250,
    C_BACKGROUND  = 255
} pixelClass;

static const int pixelClasses[] = {
    C_SHADOW, C_PENUMBRA, C_OBJECT, C_UNKNOWN, C_BACKGROUND
};

static const cv::Scalar SCALAR_C_BACKGROUND
    = cv::Scalar(C_BACKGROUND, C_BACKGROUND, C_BACKGROUND);                                                                             
static const cv::Scalar SCALAR_C_OBJECT =
    cv::Scalar(C_OBJECT, C_OBJECT, C_OBJECT);      
static const cv::Scalar SCALAR_C_SHADOW =
    cv::Scalar(C_SHADOW, C_SHADOW, C_SHADOW);
static const cv::Scalar SCALAR_C_PENUMBRA =
    cv::Scalar(C_PENUMBRA, C_PENUMBRA, C_PENUMBRA);
static const cv::Scalar SCALAR_C_EDGE =
    cv::Scalar(255, 0, 0);
