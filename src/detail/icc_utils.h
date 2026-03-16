/// \file detail/icc_utils.h
/// \brief ICC color profile extraction and conversion using lcms2.
///
/// Handles CMYK JPEGs and non-sRGB color profiles by extracting the embedded
/// ICC profile and converting pixel data to sRGB using Little CMS 2.

#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace neroued::vectorizer::detail {

/// Load an image from file with ICC color management.
///
/// If the file contains an embedded ICC profile (e.g. CMYK JPEG with a print
/// profile), the pixel data is converted to sRGB. For CMYK JPEGs, the raw
/// CMYK channels are read via libjpeg and transformed through lcms2.
/// For files without an ICC profile or with an sRGB profile, this behaves
/// like cv::imread(..., IMREAD_UNCHANGED).
///
/// \param path  Path to the image file.
/// \return BGR CV_8UC3 image in sRGB color space.
/// \throws std::runtime_error if the file cannot be loaded.
cv::Mat LoadImageIcc(const std::string& path);

/// Load an image from an in-memory buffer with ICC color management.
///
/// Same behavior as the file-path overload but operates on raw bytes
/// (e.g. from an HTTP multipart upload).
///
/// \param data  Pointer to the image bytes.
/// \param size  Number of bytes.
/// \return BGR CV_8UC3 image in sRGB color space.
/// \throws std::runtime_error if the buffer cannot be decoded.
cv::Mat LoadImageIcc(const uint8_t* data, size_t size);

} // namespace neroued::vectorizer::detail
