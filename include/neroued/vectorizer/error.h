#pragma once

/// \file error.h
/// \brief Error types for the vectorizer library.

#include <stdexcept>
#include <string>

namespace neroued::vectorizer {

/// Error categories.
enum class ErrorCode : int {
    Ok = 0,
    InvalidInput,
    IOError,
    InternalError,
};

/// Base exception for all vectorizer errors.
class Error : public std::runtime_error {
public:
    Error(ErrorCode code, const std::string& msg) : std::runtime_error(msg), code_(code) {}

    ErrorCode code() const noexcept { return code_; }

private:
    ErrorCode code_;
};

/// Invalid input arguments or data.
class InputError : public Error {
public:
    explicit InputError(const std::string& msg) : Error(ErrorCode::InvalidInput, msg) {}
};

/// File / stream I/O failure.
class IOError : public Error {
public:
    explicit IOError(const std::string& msg) : Error(ErrorCode::IOError, msg) {}
};

/// Unexpected internal logic failure.
class InternalError : public Error {
public:
    explicit InternalError(const std::string& msg) : Error(ErrorCode::InternalError, msg) {}
};

} // namespace neroued::vectorizer
