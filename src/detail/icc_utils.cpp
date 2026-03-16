/// \file detail/icc_utils.cpp
/// \brief ICC color profile extraction and conversion implementation.

#include "icc_utils.h"

#include <opencv2/imgcodecs.hpp>

#include <fstream>
#include <stdexcept>

#ifdef NV_HAS_LCMS2

#    include <jpeglib.h>
#    include <lcms2.h>

#    include <algorithm>
#    include <csetjmp>
#    include <cstdio>

namespace neroued::vectorizer::detail {

namespace {

// ── JPEG ICC profile extraction ─────────────────────────────────────────────

constexpr uint8_t kApp2Marker = 0xE2;
constexpr char kIccSig[]      = "ICC_PROFILE";
constexpr size_t kIccSigLen   = 12; // includes null terminator

struct IccChunk {
    int seq             = 0;
    int total           = 0;
    const uint8_t* data = nullptr;
    size_t size         = 0;
};

std::vector<uint8_t> ExtractIccFromJpegBytes(const std::vector<uint8_t>& buf) {
    std::vector<IccChunk> chunks;
    size_t pos = 0;
    size_t len = buf.size();

    if (len < 2 || buf[0] != 0xFF || buf[1] != 0xD8) return {};

    pos = 2;
    while (pos + 4 < len) {
        if (buf[pos] != 0xFF) {
            ++pos;
            continue;
        }
        uint8_t marker = buf[pos + 1];
        if (marker == 0xD9) break; // EOI
        if (marker == 0x00 || (marker >= 0xD0 && marker <= 0xD7) || marker == 0xD8) {
            pos += 2;
            continue;
        }
        uint16_t seg_len = static_cast<uint16_t>((buf[pos + 2] << 8) | buf[pos + 3]);
        if (marker == kApp2Marker && seg_len > kIccSigLen + 2) {
            const uint8_t* seg = buf.data() + pos + 4;
            if (std::equal(kIccSig, kIccSig + kIccSigLen, seg)) {
                IccChunk c;
                c.seq   = seg[kIccSigLen];
                c.total = seg[kIccSigLen + 1];
                c.data  = seg + kIccSigLen + 2;
                c.size  = seg_len - 2 - kIccSigLen - 2;
                chunks.push_back(c);
            }
        }
        // SOS marker: image data starts, stop scanning markers
        if (marker == 0xDA) break;
        pos += 2 + seg_len;
    }

    if (chunks.empty()) return {};
    std::sort(chunks.begin(), chunks.end(),
              [](const IccChunk& a, const IccChunk& b) { return a.seq < b.seq; });

    std::vector<uint8_t> profile;
    for (auto& c : chunks) profile.insert(profile.end(), c.data, c.data + c.size);
    return profile;
}

// ── RAII wrappers for lcms2 handles ─────────────────────────────────────────

struct ProfileDeleter {
    void operator()(void* p) const {
        if (p) cmsCloseProfile(p);
    }
};

struct TransformDeleter {
    void operator()(void* p) const {
        if (p) cmsDeleteTransform(p);
    }
};

using ProfilePtr   = std::unique_ptr<void, ProfileDeleter>;
using TransformPtr = std::unique_ptr<void, TransformDeleter>;

// ── libjpeg CMYK reading ────────────────────────────────────────────────────

struct JpegErrorMgr {
    jpeg_error_mgr pub;
    jmp_buf jmp;
};

[[noreturn]] void JpegErrorExit(j_common_ptr cinfo) {
    auto* err = reinterpret_cast<JpegErrorMgr*>(cinfo->err);
    std::longjmp(err->jmp, 1);
}

/// Decode a JPEG from memory keeping its native color space (CMYK/YCCK -> CMYK).
/// Returns a CV_8UC4 Mat (C, M, Y, K channels) or empty on failure.
cv::Mat ReadJpegCmykFromMem(const uint8_t* data, size_t size) {
    jpeg_decompress_struct cinfo{};
    JpegErrorMgr jerr{};
    cinfo.err           = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = JpegErrorExit;

    if (setjmp(jerr.jmp)) {
        jpeg_destroy_decompress(&cinfo);
        return {};
    }

    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, data, static_cast<unsigned long>(size));
    jpeg_read_header(&cinfo, TRUE);

    bool is_cmyk = (cinfo.jpeg_color_space == JCS_CMYK || cinfo.jpeg_color_space == JCS_YCCK);

    if (!is_cmyk) {
        jpeg_destroy_decompress(&cinfo);
        return {};
    }

    cinfo.out_color_space = JCS_CMYK;
    jpeg_start_decompress(&cinfo);

    int w  = static_cast<int>(cinfo.output_width);
    int h  = static_cast<int>(cinfo.output_height);
    int ch = static_cast<int>(cinfo.output_components);
    if (ch != 4) {
        jpeg_destroy_decompress(&cinfo);
        return {};
    }

    cv::Mat cmyk(h, w, CV_8UC4);
    while (cinfo.output_scanline < cinfo.output_height) {
        uint8_t* row = cmyk.ptr<uint8_t>(static_cast<int>(cinfo.output_scanline));
        jpeg_read_scanlines(&cinfo, &row, 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    return cmyk;
}

// ── ICC -> sRGB conversion via lcms2 ────────────────────────────────────────

cv::Mat ConvertCmykToSrgb(const cv::Mat& cmyk, const std::vector<uint8_t>& icc) {
    ProfilePtr src_profile(
        cmsOpenProfileFromMem(icc.data(), static_cast<cmsUInt32Number>(icc.size())));
    if (!src_profile) return {};

    cmsColorSpaceSignature cs = cmsGetColorSpace(src_profile.get());
    if (cs != cmsSigCmykData) return {};

    ProfilePtr dst_profile(cmsCreate_sRGBProfile());
    if (!dst_profile) return {};

    TransformPtr xform(cmsCreateTransform(src_profile.get(), TYPE_CMYK_8, dst_profile.get(),
                                          TYPE_BGR_8, INTENT_PERCEPTUAL, 0));
    if (!xform) return {};

    int w = cmyk.cols, h = cmyk.rows;
    cv::Mat bgr(h, w, CV_8UC3);

    // Adobe/libjpeg CMYK uses inverted convention (0=full ink, 255=no ink).
    // lcms2 TYPE_CMYK_8 expects standard convention (0=no ink, 255=full ink).
    // Invert each byte before transforming.
    std::vector<uint8_t> row_buf(static_cast<size_t>(w) * 4);
    for (int r = 0; r < h; ++r) {
        const uint8_t* src = cmyk.ptr<uint8_t>(r);
        for (int i = 0; i < w * 4; ++i)
            row_buf[static_cast<size_t>(i)] = static_cast<uint8_t>(255 - src[i]);
        cmsDoTransform(xform.get(), row_buf.data(), bgr.ptr(r), static_cast<cmsUInt32Number>(w));
    }
    return bgr;
}

cv::Mat ConvertRgbProfileToSrgb(const cv::Mat& img, const std::vector<uint8_t>& icc) {
    ProfilePtr src_profile(
        cmsOpenProfileFromMem(icc.data(), static_cast<cmsUInt32Number>(icc.size())));
    if (!src_profile) return {};

    cmsColorSpaceSignature cs = cmsGetColorSpace(src_profile.get());
    if (cs != cmsSigRgbData) return {};

    ProfilePtr dst_profile(cmsCreate_sRGBProfile());
    if (!dst_profile) return {};

    TransformPtr xform(cmsCreateTransform(src_profile.get(), TYPE_BGR_8, dst_profile.get(),
                                          TYPE_BGR_8, INTENT_PERCEPTUAL, 0));
    if (!xform) return {};

    cv::Mat out = img.clone();
    for (int r = 0; r < out.rows; ++r) {
        cmsDoTransform(xform.get(), img.ptr(r), out.ptr(r), static_cast<cmsUInt32Number>(out.cols));
    }
    return out;
}

bool IsSrgbProfile(const std::vector<uint8_t>& icc) {
    if (icc.size() < 128) return false;

    ProfilePtr prof(cmsOpenProfileFromMem(icc.data(), static_cast<cmsUInt32Number>(icc.size())));
    if (!prof) return false;

    cmsColorSpaceSignature cs = cmsGetColorSpace(prof.get());
    if (cs != cmsSigRgbData) return false;

    // Check profile description for common sRGB names
    cmsMLU* desc = static_cast<cmsMLU*>(cmsReadTag(prof.get(), cmsSigProfileDescriptionTag));
    if (desc) {
        char buf[256] = {};
        cmsMLUgetASCII(desc, "en", "US", buf, sizeof(buf));
        std::string name(buf);
        if (name.find("sRGB") != std::string::npos || name.find("srgb") != std::string::npos)
            return true;
    }

    return false;
}

} // namespace

// ── Public API ──────────────────────────────────────────────────────────────

cv::Mat LoadImageIcc(const uint8_t* data, size_t size) {
    if (!data || size == 0) throw std::runtime_error("Empty image buffer");

    // Only JPEG files can have APP2 ICC markers
    std::vector<uint8_t> buf_vec(data, data + size);
    bool is_jpeg = (size >= 2 && data[0] == 0xFF && data[1] == 0xD8);
    std::vector<uint8_t> icc;
    if (is_jpeg) { icc = ExtractIccFromJpegBytes(buf_vec); }

    if (!icc.empty()) {
        ProfilePtr prof(
            cmsOpenProfileFromMem(icc.data(), static_cast<cmsUInt32Number>(icc.size())));
        if (prof) {
            cmsColorSpaceSignature cs = cmsGetColorSpace(prof.get());
            prof.reset();

            if (cs == cmsSigCmykData) {
                cv::Mat cmyk = ReadJpegCmykFromMem(data, size);
                if (!cmyk.empty()) {
                    cv::Mat bgr = ConvertCmykToSrgb(cmyk, icc);
                    if (!bgr.empty()) return bgr;
                }
            } else if (cs == cmsSigRgbData && !IsSrgbProfile(icc)) {
                cv::Mat img = cv::imdecode(
                    cv::Mat(1, static_cast<int>(size), CV_8UC1, const_cast<uint8_t*>(data)),
                    cv::IMREAD_COLOR);
                if (!img.empty()) {
                    cv::Mat bgr = ConvertRgbProfileToSrgb(img, icc);
                    if (!bgr.empty()) return bgr;
                }
            }
        }
    }

    cv::Mat img =
        cv::imdecode(cv::Mat(1, static_cast<int>(size), CV_8UC1, const_cast<uint8_t*>(data)),
                     cv::IMREAD_UNCHANGED);
    if (img.empty()) throw std::runtime_error("Failed to decode image buffer");
    return img;
}

cv::Mat LoadImageIcc(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) throw std::runtime_error("Cannot open file: " + path);

    auto file_size = ifs.tellg();
    ifs.seekg(0);
    std::vector<uint8_t> file_buf(static_cast<size_t>(file_size));
    ifs.read(reinterpret_cast<char*>(file_buf.data()), file_size);
    ifs.close();

    return LoadImageIcc(file_buf.data(), file_buf.size());
}

} // namespace neroued::vectorizer::detail

#else // !NV_HAS_LCMS2

namespace neroued::vectorizer::detail {

cv::Mat LoadImageIcc(const uint8_t* data, size_t size) {
    if (!data || size == 0) throw std::runtime_error("Empty image buffer");
    cv::Mat img =
        cv::imdecode(cv::Mat(1, static_cast<int>(size), CV_8UC1, const_cast<uint8_t*>(data)),
                     cv::IMREAD_UNCHANGED);
    if (img.empty()) throw std::runtime_error("Failed to decode image buffer");
    return img;
}

cv::Mat LoadImageIcc(const std::string& path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (img.empty()) throw std::runtime_error("Failed to load image: " + path);
    return img;
}

} // namespace neroued::vectorizer::detail

#endif // NV_HAS_LCMS2
