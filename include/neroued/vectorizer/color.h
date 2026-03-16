#pragma once

/// \file color.h
/// \brief Color types (Rgb, Lab) and color-space conversion functions.

#include <neroued/vectorizer/vec3.h>

#include <cmath>
#include <cstdint>

namespace neroued::vectorizer {

// ── sRGB gamma ──────────────────────────────────────────────────────────────

inline float SrgbToLinear(float c) {
    if (c <= 0.04045f) { return c / 12.92f; }
    return std::pow((c + 0.055f) / 1.055f, 2.4f);
}

inline float LinearToSrgb(float c) {
    if (c <= 0.0031308f) { return 12.92f * c; }
    return 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}

// ── Color-space matrices (D65 illuminant) ───────────────────────────────────

inline Vec3f RGBToXYZ(const Vec3f& rgb) {
    float r = rgb.x, g = rgb.y, b = rgb.z;
    return {0.4124564f * r + 0.3575761f * g + 0.1804375f * b,
            0.2126729f * r + 0.7151522f * g + 0.0721750f * b,
            0.0193339f * r + 0.1191920f * g + 0.9503041f * b};
}

inline Vec3f XYZToRGB(const Vec3f& xyz) {
    return {3.2404542f * xyz.x - 1.5371385f * xyz.y - 0.4985314f * xyz.z,
            -0.9692660f * xyz.x + 1.8760108f * xyz.y + 0.0415560f * xyz.z,
            0.0556434f * xyz.x - 0.2040259f * xyz.y + 1.0572252f * xyz.z};
}

inline float LabF(float t) {
    const float delta  = 6.0f / 29.0f;
    const float delta3 = delta * delta * delta;
    if (t > delta3) { return std::cbrt(t); }
    return t / (3.0f * delta * delta) + 4.0f / 29.0f;
}

inline float LabInvF(float t) {
    const float delta = 6.0f / 29.0f;
    if (t > delta) { return t * t * t; }
    return 3.0f * delta * delta * (t - 4.0f / 29.0f);
}

inline Vec3f XYZToLab(const Vec3f& xyz) {
    constexpr float Xn = 0.95047f, Yn = 1.00000f, Zn = 1.08883f;
    float fx = LabF(xyz.x / Xn), fy = LabF(xyz.y / Yn), fz = LabF(xyz.z / Zn);
    return {116.0f * fy - 16.0f, 500.0f * (fx - fy), 200.0f * (fy - fz)};
}

inline Vec3f LabToXYZ(const Vec3f& lab) {
    constexpr float Xn = 0.95047f, Yn = 1.00000f, Zn = 1.08883f;
    float fy = (lab.x + 16.0f) / 116.0f;
    float fx = fy + lab.y / 500.0f;
    float fz = fy - lab.z / 200.0f;
    return {Xn * LabInvF(fx), Yn * LabInvF(fy), Zn * LabInvF(fz)};
}

// ── CRTP base for typed color vectors ───────────────────────────────────────

template <typename Derived>
struct ColorBase : public Vec3f {
    using Vec3f::Vec3f;

    explicit constexpr ColorBase(const Vec3f& v) : Vec3f(v) {}

    Derived operator+(const Derived& o) const { return {x + o.x, y + o.y, z + o.z}; }

    Derived operator-(const Derived& o) const { return {x - o.x, y - o.y, z - o.z}; }

    Derived operator*(float s) const { return {x * s, y * s, z * s}; }

    Derived operator/(float s) const { return {x / s, y / s, z / s}; }

    Derived& operator+=(const Derived& o) {
        x += o.x;
        y += o.y;
        z += o.z;
        return self();
    }

    Derived& operator-=(const Derived& o) {
        x -= o.x;
        y -= o.y;
        z -= o.z;
        return self();
    }

    Derived& operator*=(float s) {
        x *= s;
        y *= s;
        z *= s;
        return self();
    }

    Derived& operator/=(float s) {
        x /= s;
        y /= s;
        z /= s;
        return self();
    }

    static Derived Lerp(const Derived& a, const Derived& b, float t) { return a + (b - a) * t; }

    static float Distance(const Derived& a, const Derived& b) { return Vec3f::Distance(a, b); }

private:
    Derived& self() { return static_cast<Derived&>(*this); }
};

// ── Forward declarations ────────────────────────────────────────────────────

struct Lab;

// ── Rgb (linear sRGB, [0,1]) ────────────────────────────────────────────────

struct Rgb : public ColorBase<Rgb> {
    using ColorBase::ColorBase;
    constexpr Rgb() = default;

    constexpr Rgb(float r_, float g_, float b_) : ColorBase(Vec3f(r_, g_, b_)) {}

    explicit constexpr Rgb(const Vec3f& v) : ColorBase(v) {}

    float& r() { return x; }

    float& g() { return y; }

    float& b() { return z; }

    const float& r() const { return x; }

    const float& g() const { return y; }

    const float& b() const { return z; }

    static constexpr Rgb FromRgb(float r, float g, float b) { return {r, g, b}; }

    static Rgb FromRgb255(uint8_t r, uint8_t g, uint8_t b) {
        return {SrgbToLinear(r / 255.0f), SrgbToLinear(g / 255.0f), SrgbToLinear(b / 255.0f)};
    }

    void ToRgb255(uint8_t& r8, uint8_t& g8, uint8_t& b8) const {
        Vec3f c = Vec3f::Clamp01(*this);
        r8      = static_cast<uint8_t>(std::round(LinearToSrgb(c.x) * 255.0f));
        g8      = static_cast<uint8_t>(std::round(LinearToSrgb(c.y) * 255.0f));
        b8      = static_cast<uint8_t>(std::round(LinearToSrgb(c.z) * 255.0f));
    }

    Lab ToLab() const;
    static Rgb FromLab(const Lab& lab);

    static Rgb Clamp(const Rgb& v, float lo, float hi) { return Rgb(Vec3f::Clamp(v, lo, hi)); }

    static Rgb Clamp01(const Rgb& v) { return Clamp(v, 0.0f, 1.0f); }
};

inline Rgb operator*(float s, const Rgb& v) { return v * s; }

// ── Lab (CIE L*a*b*) ───────────────────────────────────────────────────────

struct Lab : public ColorBase<Lab> {
    using ColorBase::ColorBase;
    constexpr Lab() = default;

    constexpr Lab(float l_, float a_, float b_) : ColorBase(Vec3f(l_, a_, b_)) {}

    explicit constexpr Lab(const Vec3f& v) : ColorBase(v) {}

    float& l() { return x; }

    float& a() { return y; }

    float& b() { return z; }

    const float& l() const { return x; }

    const float& a() const { return y; }

    const float& b() const { return z; }

    static constexpr Lab FromLab(float l, float a, float b) { return {l, a, b}; }

    Rgb ToRgb() const;
    static Lab FromRgb(const Rgb& rgb);

    static float DeltaE76(const Lab& lab1, const Lab& lab2) { return Distance(lab1, lab2); }
};

inline Lab operator*(float s, const Lab& v) { return v * s; }

// ── Inline conversions ──────────────────────────────────────────────────────

inline Lab Rgb::ToLab() const {
    Vec3f lab = XYZToLab(RGBToXYZ(*this));
    return Lab(lab.x, lab.y, lab.z);
}

inline Rgb Rgb::FromLab(const Lab& lab) {
    Vec3f rgb = XYZToRGB(LabToXYZ(lab));
    return Rgb(rgb.x, rgb.y, rgb.z);
}

inline Rgb Lab::ToRgb() const {
    Vec3f rgb = XYZToRGB(LabToXYZ(*this));
    return Rgb(rgb.x, rgb.y, rgb.z);
}

inline Lab Lab::FromRgb(const Rgb& rgb) { return rgb.ToLab(); }

} // namespace neroued::vectorizer
