#pragma once

/// \file result.h
/// \brief Result of the vectorization pipeline.

#include <neroued/vectorizer/color.h>

#include <string>
#include <vector>

namespace neroued::vectorizer {

/// Result of the vectorization pipeline.
struct VectorizerResult {
    std::string svg_content;     ///< Complete SVG document.
    int width               = 0; ///< Image width in pixels.
    int height              = 0; ///< Image height in pixels.
    int num_shapes          = 0; ///< Number of shapes in the SVG.
    int resolved_num_colors = 0; ///< Actual color count used (from auto-detection or config).
    std::vector<Rgb> palette;    ///< Color palette used.
};

} // namespace neroued::vectorizer
