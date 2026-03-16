#pragma once

/// \file vectorizer.h
/// \brief Public API for raster-to-SVG vectorization.

#include <neroued/vectorizer/config.h>
#include <neroued/vectorizer/result.h>

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <string>

namespace neroued::vectorizer {

/// Vectorize a raster image file to SVG.
///
/// \param image_path  Path to input image (PNG, JPG, BMP, etc.).
/// \param config      Pipeline configuration.
/// \return            Vectorization result with SVG content.
VectorizerResult Vectorize(const std::string& image_path, const VectorizerConfig& config = {});

/// Vectorize an in-memory image buffer to SVG (ICC-aware).
///
/// \param image_data  Pointer to encoded image bytes (JPEG, PNG, etc.).
/// \param image_size  Number of bytes.
/// \param config      Pipeline configuration.
/// \return            Vectorization result with SVG content.
VectorizerResult Vectorize(const uint8_t* image_data, size_t image_size,
                           const VectorizerConfig& config = {});

/// Vectorize a BGR cv::Mat to SVG.
///
/// \param bgr_image   Input image in BGR/BGRA/GRAY format.
/// \note              For BGRA input, pixels with alpha==0 are excluded from vectorization.
/// \param config      Pipeline configuration.
/// \return            Vectorization result with SVG content.
VectorizerResult Vectorize(const cv::Mat& bgr_image, const VectorizerConfig& config = {});

} // namespace neroued::vectorizer
