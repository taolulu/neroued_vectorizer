#pragma once

/// \file boundary_graph.h
/// \brief Shared boundary graph from a label map — the core watertight mechanism.
///
/// Scans every 2x2 pixel window to detect junctions (>= 3 distinct labels)
/// and traces boundary edges between junctions along label transitions.
/// Each edge is shared exactly by two labels, guaranteeing watertight boundaries.

#include <neroued/vectorizer/vec2.h>

#include <opencv2/core.hpp>

#include <vector>

namespace neroued::vectorizer::detail {

struct BoundaryNode {
    Vec2f position;
    std::vector<int> edge_ids;
};

struct BoundaryEdge {
    int node_start  = -1;
    int node_end    = -1;
    int label_left  = -1;
    int label_right = -1;
    std::vector<Vec2f> points;
};

struct BoundaryGraph {
    std::vector<BoundaryNode> nodes;
    std::vector<BoundaryEdge> edges;
};

/// Build a shared boundary graph from a label map.
///
/// \param labels  CV_32SC1 label map; -1 means invalid/transparent pixel.
/// \return        Graph with junction nodes and shared boundary edges.
BoundaryGraph BuildBoundaryGraph(const cv::Mat& labels);

} // namespace neroued::vectorizer::detail
