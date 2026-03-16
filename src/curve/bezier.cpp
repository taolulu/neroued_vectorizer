#include "bezier.h"

#include <algorithm>
#include <cmath>

namespace neroued::vectorizer::detail {

Vec2f EvalBezier(const CubicBezier& b, float t) {
    float s = 1.0f - t;
    return b.p0 * (s * s * s) + b.p1 * (3 * s * s * t) + b.p2 * (3 * s * t * t) +
           b.p3 * (t * t * t);
}

Vec2f EvalBezierDeriv(const CubicBezier& b, float t) {
    float s  = 1.0f - t;
    Vec2f d1 = (b.p1 - b.p0) * 3.0f;
    Vec2f d2 = (b.p2 - b.p1) * 3.0f;
    Vec2f d3 = (b.p3 - b.p2) * 3.0f;
    return d1 * (s * s) + d2 * (2 * s * t) + d3 * (t * t);
}

Vec2f EvalBezierSecondDeriv(const CubicBezier& b, float t) {
    float s  = 1.0f - t;
    Vec2f d1 = (b.p2 - b.p1 * 2.0f + b.p0) * 6.0f;
    Vec2f d2 = (b.p3 - b.p2 * 2.0f + b.p1) * 6.0f;
    return d1 * s + d2 * t;
}

void FlattenCubicBezier(Vec2f p0, Vec2f p1, Vec2f p2, Vec2f p3, float tolerance,
                        std::vector<Vec2f>& out) {
    Vec2f baseline = p3 - p0;
    float d2       = std::abs((p1 - p3).Cross(baseline));
    float d3       = std::abs((p2 - p3).Cross(baseline));

    float len_sq = baseline.LengthSquared();
    float tol_sq = tolerance * tolerance * len_sq;

    if ((d2 + d3) * (d2 + d3) <= tol_sq) {
        out.push_back(p3);
        return;
    }

    Vec2f p01  = (p0 + p1) * 0.5f;
    Vec2f p12  = (p1 + p2) * 0.5f;
    Vec2f p23  = (p2 + p3) * 0.5f;
    Vec2f p012 = (p01 + p12) * 0.5f;
    Vec2f p123 = (p12 + p23) * 0.5f;
    Vec2f mid  = (p012 + p123) * 0.5f;

    FlattenCubicBezier(p0, p01, p012, mid, tolerance, out);
    FlattenCubicBezier(mid, p123, p23, p3, tolerance, out);
}

double BezierContourSignedArea(const BezierContour& contour) {
    if (contour.segments.empty()) return 0.0;
    std::vector<Vec2f> pts;
    pts.push_back(contour.segments[0].p0);
    for (const auto& seg : contour.segments) { FlattenCubicBezier(seg, 0.25f, pts); }
    double acc = 0.0;
    for (size_t i = 0; i < pts.size(); ++i) {
        const Vec2f& a = pts[i];
        const Vec2f& b = pts[(i + 1) % pts.size()];
        acc += static_cast<double>(a.x) * b.y - static_cast<double>(b.x) * a.y;
    }
    return 0.5 * acc;
}

void ReverseBezierContour(BezierContour& contour) {
    std::reverse(contour.segments.begin(), contour.segments.end());
    for (auto& seg : contour.segments) {
        std::swap(seg.p0, seg.p3);
        std::swap(seg.p1, seg.p2);
    }
}

} // namespace neroued::vectorizer::detail
