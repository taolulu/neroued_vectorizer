#include "topology.h"

#include <clipper2/clipper.h>

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace neroued::vectorizer::detail {

namespace {

constexpr double kScale = 1024.0;

Clipper2Lib::Path64 RingToPath(const std::vector<Vec2f>& ring) {
    Clipper2Lib::Path64 out;
    out.reserve(ring.size());
    for (const auto& p : ring) {
        out.emplace_back(static_cast<int64_t>(std::llround(p.x * kScale)),
                         static_cast<int64_t>(std::llround(p.y * kScale)));
    }
    if (!out.empty() && out.front() == out.back()) out.pop_back();
    return out;
}

std::vector<Vec2f> PathToRing(const Clipper2Lib::Path64& path) {
    std::vector<Vec2f> out;
    out.reserve(path.size());
    for (const auto& p : path) {
        out.push_back({static_cast<float>(static_cast<double>(p.x) / kScale),
                       static_cast<float>(static_cast<double>(p.y) / kScale)});
    }
    if (out.size() > 1 && (out.front() - out.back()).LengthSquared() < 1e-6f) out.pop_back();
    return out;
}

struct RingInfo {
    Clipper2Lib::Path64 path;
    double area     = 0.0;
    double abs_area = 0.0;
    int parent      = -1;
    int depth       = 0;
};

} // namespace

std::vector<TracedPolygonGroup> RepairTopology(const std::vector<TracedPolygonGroup>& groups,
                                               float simplify_epsilon, float min_outer_area,
                                               float min_hole_area) {
    Clipper2Lib::Paths64 raw;
    for (const auto& g : groups) {
        if (g.outer.size() >= 3) raw.push_back(RingToPath(g.outer));
        for (const auto& h : g.holes) {
            if (h.size() >= 3) raw.push_back(RingToPath(h));
        }
    }
    if (raw.empty()) return {};

    double simplify = std::max(1.0, static_cast<double>(simplify_epsilon) * kScale);
    auto simplified = Clipper2Lib::SimplifyPaths(raw, simplify, true);
    auto unified    = Clipper2Lib::Union(simplified, Clipper2Lib::FillRule::EvenOdd);
    if (unified.empty()) return {};

    std::vector<RingInfo> rings;
    rings.reserve(unified.size());
    double min_keep_area =
        std::max(0.1, static_cast<double>(std::min(min_outer_area, min_hole_area)));
    for (const auto& p : unified) {
        if (p.size() < 3) continue;
        RingInfo info;
        info.path     = p;
        info.area     = Clipper2Lib::Area(p) / (kScale * kScale);
        info.abs_area = std::abs(info.area);
        if (info.abs_area < min_keep_area) continue;
        rings.push_back(std::move(info));
    }
    if (rings.empty()) return {};

    std::vector<int> order(rings.size());
    for (int i = 0; i < static_cast<int>(order.size()); ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return rings[a].abs_area > rings[b].abs_area; });

    for (size_t oi = 0; oi < order.size(); ++oi) {
        int i = order[oi];
        for (size_t oj = 0; oj < oi; ++oj) {
            int j = order[oj];
            if (Clipper2Lib::Path2ContainsPath1(rings[i].path, rings[j].path)) {
                rings[i].parent = j;
                break;
            }
        }
    }

    for (int idx : order) {
        int d = 0;
        int p = rings[idx].parent;
        while (p != -1) {
            ++d;
            p = rings[p].parent;
        }
        rings[idx].depth = d;
    }

    std::vector<TracedPolygonGroup> repaired;
    std::unordered_map<int, int> outer_to_group;
    for (int idx : order) {
        if ((rings[idx].depth % 2) != 0) continue;

        auto outer = PathToRing(rings[idx].path);
        if (outer.size() < 3) continue;
        if (SignedArea(outer) < 0.0) std::reverse(outer.begin(), outer.end());
        double area = std::abs(SignedArea(outer));
        if (area < static_cast<double>(min_outer_area)) continue;

        TracedPolygonGroup g;
        g.outer             = std::move(outer);
        g.area              = area;
        outer_to_group[idx] = static_cast<int>(repaired.size());
        repaired.push_back(std::move(g));
    }

    for (int idx : order) {
        if ((rings[idx].depth % 2) == 0) continue;

        int p = rings[idx].parent;
        while (p != -1 && (rings[p].depth % 2) != 0) p = rings[p].parent;
        if (p == -1) continue;
        auto git = outer_to_group.find(p);
        if (git == outer_to_group.end()) continue;

        auto hole = PathToRing(rings[idx].path);
        if (hole.size() < 3) continue;
        if (SignedArea(hole) > 0.0) std::reverse(hole.begin(), hole.end());
        double area = std::abs(SignedArea(hole));
        if (area < static_cast<double>(min_hole_area)) continue;
        repaired[git->second].holes.push_back(std::move(hole));
    }

    std::sort(repaired.begin(), repaired.end(),
              [](const auto& a, const auto& b) { return a.area > b.area; });
    return repaired;
}

} // namespace neroued::vectorizer::detail
