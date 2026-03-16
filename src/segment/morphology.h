#pragma once

/// \file morphology.h
/// \brief Shared morphological operations (Zhang-Suen thinning).

#include <opencv2/core.hpp>

namespace neroued::vectorizer::detail {

/// Zhang-Suen thinning to produce a 1-pixel wide skeleton from a binary mask.
/// Input must be CV_8UC1 with foreground = 255, background = 0.
cv::Mat ZhangSuenThinning(const cv::Mat& binary_mask);

} // namespace neroued::vectorizer::detail
