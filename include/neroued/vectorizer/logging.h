#pragma once

/// \file logging.h
/// \brief Logging initialization utilities.

#include <spdlog/spdlog.h>

#include <algorithm>
#include <string>

namespace neroued::vectorizer {

void InitLogging(spdlog::level::level_enum level = spdlog::level::info);

spdlog::level::level_enum ParseLogLevel(const std::string& str);

} // namespace neroued::vectorizer
