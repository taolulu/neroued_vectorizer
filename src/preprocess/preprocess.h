#pragma once

/// \file preprocess.h
/// \brief Preprocessing for vectorization (auto downscale/upscale + Mean Shift smoothing).

#include <opencv2/core.hpp>

namespace neroued::vectorizer::detail {

struct PreprocessResult {
    cv::Mat bgr;            ///< Scaled and smoothed BGR image (fed to SLIC).
    cv::Mat unsmoothed_bgr; ///< Scaled but NOT smoothed BGR (for pixel-level refinement).
    float scale = 1.0f;
};

/// Conditionally downscale large images, upscale small images, and apply color smoothing.
///
/// Area downscale triggers when total_pixels exceeds \p max_working_pixels.
/// Lanczos 2x upscale triggers only when short_edge < \p upscale_short_edge and total_pixels < 1MP.
/// Mean Shift filtering is applied when \p enable_color_smoothing is true (skip for binary mode).
PreprocessResult PreprocessForVectorize(const cv::Mat& bgr, bool enable_color_smoothing = true,
                                        float smoothing_spatial = 15.0f,
                                        float smoothing_color = 25.0f, int upscale_short_edge = 600,
                                        int max_working_pixels = 3000000);

} // namespace neroued::vectorizer::detail
