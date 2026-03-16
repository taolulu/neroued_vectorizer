#include "slic.h"

#include <neroued/vectorizer/error.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <unordered_map>
#include <vector>

namespace neroued::vectorizer::detail {
namespace {

struct SlicCenter {
    cv::Vec3f color;
    float x = 0.0f;
    float y = 0.0f;
};

bool IsValidPixel(const cv::Mat& mask, int r, int c) {
    return mask.empty() || mask.at<uint8_t>(r, c) != 0;
}

int EstimateGridStep(std::size_t valid_count, int target_superpixels) {
    if (valid_count == 0 || target_superpixels <= 0) { return 1; }
    const double area_per_cluster =
        static_cast<double>(valid_count) / static_cast<double>(target_superpixels);
    return std::max(1, static_cast<int>(std::round(std::sqrt(area_per_cluster))));
}

bool FindClosestValid(const cv::Mat& mask, int r, int c, int radius, int& out_r, int& out_c) {
    if (mask.empty()) {
        out_r = r;
        out_c = c;
        return true;
    }

    const int rows = mask.rows;
    const int cols = mask.cols;
    float best_d2  = std::numeric_limits<float>::max();
    bool found     = false;

    const int r0 = std::max(0, r - radius);
    const int r1 = std::min(rows - 1, r + radius);
    const int c0 = std::max(0, c - radius);
    const int c1 = std::min(cols - 1, c + radius);

    for (int rr = r0; rr <= r1; ++rr) {
        const uint8_t* mask_row = mask.ptr<uint8_t>(rr);
        for (int cc = c0; cc <= c1; ++cc) {
            if (mask_row[cc] == 0) { continue; }
            const float dr = static_cast<float>(rr - r);
            const float dc = static_cast<float>(cc - c);
            const float d2 = dr * dr + dc * dc;
            if (!found || d2 < best_d2) {
                best_d2 = d2;
                out_r   = rr;
                out_c   = cc;
                found   = true;
            }
        }
    }
    return found;
}

std::vector<SlicCenter> InitializeCenters(const cv::Mat& target, const cv::Mat& mask,
                                          int target_superpixels, int step) {
    const int rows = target.rows;
    const int cols = target.cols;
    std::vector<SlicCenter> centers;
    centers.reserve(static_cast<std::size_t>(target_superpixels));

    for (int r = step / 2; r < rows; r += step) {
        for (int c = step / 2; c < cols; c += step) {
            int rr = r;
            int cc = c;
            if (!FindClosestValid(mask, r, c, step, rr, cc)) { continue; }
            const cv::Vec3f color = target.at<cv::Vec3f>(rr, cc);
            centers.push_back(SlicCenter{color, static_cast<float>(cc), static_cast<float>(rr)});
        }
    }

    if (!centers.empty() && static_cast<int>(centers.size()) > target_superpixels) {
        std::vector<SlicCenter> reduced;
        reduced.reserve(static_cast<std::size_t>(target_superpixels));
        const double stride =
            static_cast<double>(centers.size()) / static_cast<double>(target_superpixels);
        for (int i = 0; i < target_superpixels; ++i) {
            const int idx = std::min(static_cast<int>(std::floor(i * stride)),
                                     static_cast<int>(centers.size()) - 1);
            reduced.push_back(centers[static_cast<std::size_t>(idx)]);
        }
        centers.swap(reduced);
    }

    if (!centers.empty()) { return centers; }

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (!IsValidPixel(mask, r, c)) { continue; }
            centers.push_back(SlicCenter{target.at<cv::Vec3f>(r, c), static_cast<float>(c),
                                         static_cast<float>(r)});
            return centers;
        }
    }
    return centers;
}

void PerturbCentersToMinGradient(const cv::Mat& target, const cv::Mat& mask,
                                 std::vector<SlicCenter>& centers) {
    if (target.empty() || centers.empty()) return;

    cv::Mat channel;
    cv::extractChannel(target, channel, 0);
    cv::Mat gx, gy;
    cv::Sobel(channel, gx, CV_32F, 1, 0, 3);
    cv::Sobel(channel, gy, CV_32F, 0, 1, 3);
    cv::Mat grad;
    cv::magnitude(gx, gy, grad);

    for (auto& center : centers) {
        int cy = static_cast<int>(std::round(center.y));
        int cx = static_cast<int>(std::round(center.x));

        float best_grad = std::numeric_limits<float>::max();
        int best_r = cy, best_c = cx;
        for (int dr = -1; dr <= 1; ++dr) {
            for (int dc = -1; dc <= 1; ++dc) {
                int nr = cy + dr;
                int nc = cx + dc;
                if (nr < 0 || nr >= target.rows || nc < 0 || nc >= target.cols) continue;
                if (!mask.empty() && mask.at<uint8_t>(nr, nc) == 0) continue;
                float g = grad.at<float>(nr, nc);
                if (g < best_grad) {
                    best_grad = g;
                    best_r    = nr;
                    best_c    = nc;
                }
            }
        }
        center.x     = static_cast<float>(best_c);
        center.y     = static_cast<float>(best_r);
        center.color = target.at<cv::Vec3f>(best_r, best_c);
    }
}

void AssignLabels(const cv::Mat& target, const cv::Mat& mask, const cv::Mat& edge_map,
                  float edge_sensitivity, const std::vector<SlicCenter>& centers, int step,
                  float compactness, cv::Mat& labels, cv::Mat& distance) {
    labels.setTo(cv::Scalar(-1));
    distance.setTo(cv::Scalar(std::numeric_limits<float>::max()));

    const float inv_step = 1.0f / static_cast<float>(std::max(1, step));
    const float lambda   = (compactness * inv_step) * (compactness * inv_step);
    const bool use_edges =
        !edge_map.empty() && edge_sensitivity > 0.0f && edge_map.size() == target.size();

    for (int k = 0; k < static_cast<int>(centers.size()); ++k) {
        const SlicCenter& center = centers[static_cast<std::size_t>(k)];
        const int cx             = static_cast<int>(std::round(center.x));
        const int cy             = static_cast<int>(std::round(center.y));

        const int r0 = std::max(0, cy - 2 * step);
        const int r1 = std::min(target.rows - 1, cy + 2 * step);
        const int c0 = std::max(0, cx - 2 * step);
        const int c1 = std::min(target.cols - 1, cx + 2 * step);

        for (int r = r0; r <= r1; ++r) {
            const cv::Vec3f* target_row = target.ptr<cv::Vec3f>(r);
            const uint8_t* mask_row     = mask.empty() ? nullptr : mask.ptr<uint8_t>(r);
            const float* edge_row       = use_edges ? edge_map.ptr<float>(r) : nullptr;
            int* label_row              = labels.ptr<int>(r);
            float* distance_row         = distance.ptr<float>(r);
            for (int c = c0; c <= c1; ++c) {
                if (mask_row && mask_row[c] == 0) { continue; }
                const cv::Vec3f& pixel = target_row[c];
                const float d0         = pixel[0] - center.color[0];
                const float d1         = pixel[1] - center.color[1];
                const float d2         = pixel[2] - center.color[2];
                const float dc2        = d0 * d0 + d1 * d1 + d2 * d2;
                const float dr         = static_cast<float>(r) - center.y;
                const float dc_val     = static_cast<float>(c) - center.x;
                const float ds2        = dr * dr + dc_val * dc_val;
                float spatial_lambda   = lambda;
                if (edge_row) {
                    float ef       = 1.0f - edge_sensitivity * edge_row[c];
                    spatial_lambda = lambda * std::max(0.1f, ef);
                }
                const float d = dc2 + spatial_lambda * ds2;
                if (d < distance_row[c]) {
                    distance_row[c] = d;
                    label_row[c]    = k;
                }
            }
        }
    }
}

void FillUnassignedPixels(const cv::Mat& target, const cv::Mat& mask,
                          const std::vector<SlicCenter>& centers, int step, float compactness,
                          cv::Mat& labels) {
    const float inv_step = 1.0f / static_cast<float>(std::max(1, step));
    const float lambda   = (compactness * inv_step) * (compactness * inv_step);

    for (int r = 0; r < labels.rows; ++r) {
        const cv::Vec3f* target_row = target.ptr<cv::Vec3f>(r);
        const uint8_t* mask_row     = mask.empty() ? nullptr : mask.ptr<uint8_t>(r);
        int* label_row              = labels.ptr<int>(r);
        for (int c = 0; c < labels.cols; ++c) {
            if (mask_row && mask_row[c] == 0) { continue; }
            if (label_row[c] >= 0) { continue; }

            float best_d           = std::numeric_limits<float>::max();
            int best_k             = 0;
            const cv::Vec3f& pixel = target_row[c];
            for (int k = 0; k < static_cast<int>(centers.size()); ++k) {
                const SlicCenter& center = centers[static_cast<std::size_t>(k)];
                const float d0           = pixel[0] - center.color[0];
                const float d1           = pixel[1] - center.color[1];
                const float d2           = pixel[2] - center.color[2];
                const float dc2          = d0 * d0 + d1 * d1 + d2 * d2;
                const float dr           = static_cast<float>(r) - center.y;
                const float dc           = static_cast<float>(c) - center.x;
                const float ds2          = dr * dr + dc * dc;
                const float d            = dc2 + lambda * ds2;
                if (d < best_d) {
                    best_d = d;
                    best_k = k;
                }
            }
            label_row[c] = best_k;
        }
    }
}

void UpdateCenters(const cv::Mat& target, const cv::Mat& mask, cv::Mat& labels,
                   std::vector<SlicCenter>& centers) {
    struct CenterAccum {
        cv::Vec3d color{0.0, 0.0, 0.0};
        double x  = 0.0;
        double y  = 0.0;
        int count = 0;
    };

    std::vector<CenterAccum> sums(centers.size());
    for (int r = 0; r < target.rows; ++r) {
        const cv::Vec3f* target_row = target.ptr<cv::Vec3f>(r);
        const uint8_t* mask_row     = mask.empty() ? nullptr : mask.ptr<uint8_t>(r);
        const int* label_row        = labels.ptr<int>(r);
        for (int c = 0; c < target.cols; ++c) {
            if (mask_row && mask_row[c] == 0) { continue; }
            const int label = label_row[c];
            if (label < 0 || label >= static_cast<int>(centers.size())) { continue; }
            auto& acc              = sums[static_cast<std::size_t>(label)];
            const cv::Vec3f& color = target_row[c];
            acc.color[0] += color[0];
            acc.color[1] += color[1];
            acc.color[2] += color[2];
            acc.x += static_cast<double>(c);
            acc.y += static_cast<double>(r);
            ++acc.count;
        }
    }

    for (int i = 0; i < static_cast<int>(centers.size()); ++i) {
        const CenterAccum& acc = sums[static_cast<std::size_t>(i)];
        if (acc.count <= 0) { continue; }
        const double inv                           = 1.0 / static_cast<double>(acc.count);
        centers[static_cast<std::size_t>(i)].color = cv::Vec3f(
            static_cast<float>(acc.color[0] * inv), static_cast<float>(acc.color[1] * inv),
            static_cast<float>(acc.color[2] * inv));
        centers[static_cast<std::size_t>(i)].x = static_cast<float>(acc.x * inv);
        centers[static_cast<std::size_t>(i)].y = static_cast<float>(acc.y * inv);
    }
}

void EnforceConnectivity(cv::Mat& labels, const cv::Mat& mask, int min_region_area) {
    if (min_region_area <= 1 || labels.empty()) { return; }

    const int rows = labels.rows;
    const int cols = labels.cols;
    cv::Mat visited(rows, cols, CV_8UC1, cv::Scalar(0));

    std::queue<cv::Point> q;
    std::vector<cv::Point> component;
    component.reserve(256);
    constexpr std::array<int, 4> kDr = {1, -1, 0, 0};
    constexpr std::array<int, 4> kDc = {0, 0, 1, -1};

    for (int sr = 0; sr < rows; ++sr) {
        for (int sc = 0; sc < cols; ++sc) {
            if (!IsValidPixel(mask, sr, sc) || visited.at<uint8_t>(sr, sc) != 0) { continue; }
            const int seed_label = labels.at<int>(sr, sc);
            if (seed_label < 0) { continue; }

            q.push({sc, sr});
            visited.at<uint8_t>(sr, sc) = 1;
            component.clear();
            component.push_back({sc, sr});
            std::unordered_map<int, int> neighbor_hist;

            while (!q.empty()) {
                const cv::Point p = q.front();
                q.pop();
                for (int k = 0; k < 4; ++k) {
                    const int nr = p.y + kDr[static_cast<std::size_t>(k)];
                    const int nc = p.x + kDc[static_cast<std::size_t>(k)];
                    if (nr < 0 || nr >= rows || nc < 0 || nc >= cols ||
                        !IsValidPixel(mask, nr, nc)) {
                        continue;
                    }
                    const int nl = labels.at<int>(nr, nc);
                    if (nl == seed_label) {
                        if (visited.at<uint8_t>(nr, nc) == 0) {
                            visited.at<uint8_t>(nr, nc) = 1;
                            q.push({nc, nr});
                            component.push_back({nc, nr});
                        }
                    } else if (nl >= 0) {
                        neighbor_hist[nl]++;
                    }
                }
            }

            if (static_cast<int>(component.size()) >= min_region_area || neighbor_hist.empty()) {
                continue;
            }

            int best_label = seed_label;
            int best_votes = -1;
            for (const auto& [candidate, votes] : neighbor_hist) {
                if (votes > best_votes) {
                    best_votes = votes;
                    best_label = candidate;
                }
            }
            if (best_label == seed_label) { continue; }
            for (const cv::Point& p : component) { labels.at<int>(p.y, p.x) = best_label; }
        }
    }
}

std::vector<cv::Vec3f> CompactAndComputeCenters(const cv::Mat& target, const cv::Mat& mask,
                                                cv::Mat& labels) {
    int max_label = -1;
    for (int r = 0; r < labels.rows; ++r) {
        const uint8_t* mask_row = mask.empty() ? nullptr : mask.ptr<uint8_t>(r);
        const int* row          = labels.ptr<int>(r);
        for (int c = 0; c < labels.cols; ++c) {
            if (mask_row && mask_row[c] == 0) { continue; }
            max_label = std::max(max_label, row[c]);
        }
    }
    if (max_label < 0) { return {}; }

    std::vector<int> remap(static_cast<std::size_t>(max_label + 1), -1);
    int next_label = 0;
    for (int r = 0; r < labels.rows; ++r) {
        const uint8_t* mask_row = mask.empty() ? nullptr : mask.ptr<uint8_t>(r);
        int* row                = labels.ptr<int>(r);
        for (int c = 0; c < labels.cols; ++c) {
            if (mask_row && mask_row[c] == 0) {
                row[c] = -1;
                continue;
            }
            const int old_label = row[c];
            if (old_label < 0) {
                row[c] = -1;
                continue;
            }
            int& mapped = remap[static_cast<std::size_t>(old_label)];
            if (mapped < 0) { mapped = next_label++; }
            row[c] = mapped;
        }
    }

    std::vector<cv::Vec3d> sums(static_cast<std::size_t>(next_label), cv::Vec3d(0.0, 0.0, 0.0));
    std::vector<int> counts(static_cast<std::size_t>(next_label), 0);
    for (int r = 0; r < labels.rows; ++r) {
        const int* label_row        = labels.ptr<int>(r);
        const cv::Vec3f* target_row = target.ptr<cv::Vec3f>(r);
        for (int c = 0; c < labels.cols; ++c) {
            const int label = label_row[c];
            if (label < 0) { continue; }
            auto& sum              = sums[static_cast<std::size_t>(label)];
            const cv::Vec3f& pixel = target_row[c];
            sum[0] += pixel[0];
            sum[1] += pixel[1];
            sum[2] += pixel[2];
            counts[static_cast<std::size_t>(label)]++;
        }
    }

    std::vector<cv::Vec3f> centers(static_cast<std::size_t>(next_label), cv::Vec3f(0, 0, 0));
    for (int i = 0; i < next_label; ++i) {
        const int count = counts[static_cast<std::size_t>(i)];
        if (count <= 0) { continue; }
        const double inv     = 1.0 / static_cast<double>(count);
        const cv::Vec3d& sum = sums[static_cast<std::size_t>(i)];
        centers[static_cast<std::size_t>(i)] =
            cv::Vec3f(static_cast<float>(sum[0] * inv), static_cast<float>(sum[1] * inv),
                      static_cast<float>(sum[2] * inv));
    }
    return centers;
}

} // namespace

SlicResult SegmentBySlic(const cv::Mat& target, const cv::Mat& mask, const SlicConfig& cfg) {
    if (target.empty()) { throw InputError("SLIC target image is empty"); }
    if (target.type() != CV_32FC3) { throw InputError("SLIC target image must be CV_32FC3"); }
    if (!mask.empty()) {
        if (mask.type() != CV_8UC1) { throw InputError("SLIC mask must be CV_8UC1"); }
        if (mask.rows != target.rows || mask.cols != target.cols) {
            throw InputError("SLIC mask size mismatch");
        }
    }

    SlicResult out;
    out.labels = cv::Mat(target.rows, target.cols, CV_32SC1, cv::Scalar(-1));

    std::size_t valid_count = 0;
    for (int r = 0; r < target.rows; ++r) {
        const uint8_t* mask_row = mask.empty() ? nullptr : mask.ptr<uint8_t>(r);
        for (int c = 0; c < target.cols; ++c) {
            if (!mask_row || mask_row[c] != 0) { ++valid_count; }
        }
    }
    if (valid_count == 0) { return out; }

    const int target_superpixels =
        cfg.region_size > 0
            ? std::max(1, static_cast<int>(valid_count) / (cfg.region_size * cfg.region_size))
            : std::min(std::max(1, cfg.target_superpixels), static_cast<int>(valid_count));
    const int step = EstimateGridStep(valid_count, target_superpixels);

    std::vector<SlicCenter> centers = InitializeCenters(target, mask, target_superpixels, step);
    if (centers.empty()) { return out; }
    PerturbCentersToMinGradient(target, mask, centers);

    const int iterations    = std::max(1, cfg.iterations);
    const float compactness = std::max(0.001f, cfg.compactness);

    cv::Mat distance(target.rows, target.cols, CV_32FC1);
    for (int iter = 0; iter < iterations; ++iter) {
        AssignLabels(target, mask, cfg.edge_map, cfg.edge_sensitivity, centers, step, compactness,
                     out.labels, distance);
        FillUnassignedPixels(target, mask, centers, step, compactness, out.labels);
        UpdateCenters(target, mask, out.labels, centers);
    }

    const float min_ratio = std::clamp(cfg.min_region_ratio, 0.0f, 1.0f);
    if (min_ratio > 0.0f && !centers.empty()) {
        const float expected_area =
            static_cast<float>(valid_count) / static_cast<float>(centers.size());
        const int min_region_area =
            std::max(1, static_cast<int>(std::round(expected_area * min_ratio)));
        EnforceConnectivity(out.labels, mask, min_region_area);
    }

    out.centers = CompactAndComputeCenters(target, mask, out.labels);
    return out;
}

} // namespace neroued::vectorizer::detail
