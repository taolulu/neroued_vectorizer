#pragma once

#include <opencv2/core.hpp>

#include <vector>

namespace neroued::vectorizer::detail {

struct SlicConfig {
    int target_superpixels = 256;
    int region_size        = 0; ///< If > 0, overrides target_superpixels via H*W/(region_size^2).
    float compactness      = 10.0f;
    int iterations         = 10;
    float min_region_ratio = 0.25f;

    /// Optional edge magnitude map (CV_32FC1, same size as target, values in [0,1]).
    /// When provided, SLIC reduces spatial weight near strong edges so that
    /// superpixel boundaries align with image edges.
    cv::Mat edge_map;
    float edge_sensitivity = 0.5f; ///< How strongly edges reduce spatial weight [0,1].
};

struct SlicResult {
    cv::Mat labels; // H x W, CV_32SC1, -1 means invalid pixel.
    std::vector<cv::Vec3f> centers;
};

SlicResult SegmentBySlic(const cv::Mat& target, const cv::Mat& mask, const SlicConfig& cfg);

} // namespace neroued::vectorizer::detail
