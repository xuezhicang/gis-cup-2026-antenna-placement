#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

inline constexpr double EPS = 1e-12;

struct Pt {
    double x, y;
};

struct BBox {
    double minx;
    double miny;
    double maxx;
    double maxy;

    static BBox empty() {
        double inf = std::numeric_limits<double>::infinity();
        return {inf, inf, -inf, -inf};
    }

    bool valid() const {
        return minx <= maxx && miny <= maxy;
    }

    void expand(Pt p) {
        minx = std::min(minx, p.x);
        miny = std::min(miny, p.y);
        maxx = std::max(maxx, p.x);
        maxy = std::max(maxy, p.y);
    }

    void expand(BBox other) {
        if (!other.valid()) {
            return;
        }
        minx = std::min(minx, other.minx);
        miny = std::min(miny, other.miny);
        maxx = std::max(maxx, other.maxx);
        maxy = std::max(maxy, other.maxy);
    }

    bool contains(Pt p) const {
        return valid()
            && minx - EPS <= p.x && p.x <= maxx + EPS
            && miny - EPS <= p.y && p.y <= maxy + EPS;
    }
};

struct Building {
    Pt center;
    BBox bbox;
};

inline BBox polygon_bbox(const std::vector<Pt>& polygon) {
    BBox bbox = BBox::empty();
    for (Pt p : polygon) {
        bbox.expand(p);
    }
    return bbox;
}

inline Pt polygon_center(const std::vector<Pt>& polygon, BBox bbox) {
    if (polygon.empty()) {
        return {0.0, 0.0};
    }

    double twice_area = 0.0;
    double cx_sum = 0.0;
    double cy_sum = 0.0;
    int n = (int)polygon.size();

    for (int i = 0; i < n; i++) {
        Pt a = polygon[i];
        Pt b = polygon[(i + 1) % n];
        double cross = a.x * b.y - b.x * a.y;
        twice_area += cross;
        cx_sum += (a.x + b.x) * cross;
        cy_sum += (a.y + b.y) * cross;
    }

    if (std::abs(twice_area) > EPS) {
        return {cx_sum / (3.0 * twice_area), cy_sum / (3.0 * twice_area)};
    }

    return {
        (bbox.minx + bbox.maxx) * 0.5,
        (bbox.miny + bbox.maxy) * 0.5
    };
}
