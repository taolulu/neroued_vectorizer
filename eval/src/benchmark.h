#pragma once

#include <neroued/vectorizer/eval.h>

namespace neroued::vectorizer::eval {

std::string MakeRunId();
std::string GitShortHash();
std::string CurrentTimestamp();

} // namespace neroued::vectorizer::eval
