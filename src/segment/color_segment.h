#pragma once

/// \file color_segment.h
/// \brief Color segmentation: binary/multi-color clustering, label refinement, morphology cleanup.

#include <neroued/vectorizer/color.h>

#include <opencv2/core.hpp>

#include <vector>

namespace neroued::vectorizer::detail {

struct SegmentationResult {
    cv::Mat labels;
    cv::Mat lab;
    std::vector<cv::Vec3f> centers_lab;
};

SegmentationResult SegmentBinary(const cv::Mat& bgr, const cv::Mat& lab);

SegmentationResult SegmentMultiColor(const cv::Mat& lab, int num_colors, int slic_region_size,
                                     float slic_compactness, const cv::Mat& edge_map,
                                     float edge_sensitivity);

cv::Mat ComputeEdgeMap(const cv::Mat& bgr);

void RefineLabelsBoundary(cv::Mat& labels, const cv::Mat& unsmoothed_lab,
                          const std::vector<cv::Vec3f>& centers_lab, int passes);

void MergeSmallComponents(cv::Mat& labels, const cv::Mat& lab, std::vector<cv::Vec3f>& centers_lab,
                          int min_region_area, float max_merge_color_dist);

void MorphologicalCleanup(cv::Mat& labels, int num_labels, int close_radius);

int CompactLabels(cv::Mat& labels, std::vector<cv::Vec3f>& centers_lab);

std::vector<Rgb> ComputePalette(const cv::Mat& bgr, const cv::Mat& labels, int num_labels);

int EstimateOptimalColors(const cv::Mat& bgr);

} // namespace neroued::vectorizer::detail
