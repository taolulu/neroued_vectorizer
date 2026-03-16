#pragma once

/// \file topology.h
/// \brief Topology cleanup for traced polygon groups.

#include "potrace.h"

#include <vector>

namespace neroued::vectorizer::detail {

/// Clean traced polygons with boolean operations and rebuild outer/hole hierarchy.
std::vector<TracedPolygonGroup> RepairTopology(const std::vector<TracedPolygonGroup>& groups,
                                               float simplify_epsilon, float min_outer_area,
                                               float min_hole_area);

} // namespace neroued::vectorizer::detail
