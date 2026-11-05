#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "geometry.hpp"

namespace py = pybind11;

constexpr double VIS_EPS = 1e-9;
constexpr double VIS_ANGLE_EPS = 1e-9;
constexpr double VIS_POINT_EPS = 1e-7;
constexpr double VIS_TWO_PI = 6.283185307179586476925286766559;

struct TreeNode {
    int id = -1;
    std::array<int, 4> children{{-1, -1, -1, -1}};
    int child_count = 0;
    int start = 0;
    int end = 0;
    int depth = 0;
    double split_x = 0.0;
    double split_y = 0.0;
    BBox center_bbox = BBox::empty();
    BBox cell_bbox = BBox::empty();
    BBox bbox = BBox::empty();

    bool is_leaf() const {
        return child_count == 0;
    }
};

struct Edge {
    int building_index = -1;
    int edge_index = -1;
    Pt a{0.0, 0.0};
    Pt b{0.0, 0.0};
    double source_s1 = 0.0;
    double source_s2 = 1.0;
};

struct Event {
    double angle = 0.0;
    int type = 0; // -1=end, 1=start
    int edge = -1;
};

struct RayHit {
    bool hit = false;
    double t = 0.0;
    double s = 0.0;
    Pt point{0.0, 0.0};
};

struct VisibleSegment {
    int building_index = -1;
    int edge_index = -1;
    double s1 = 0.0;
    double s2 = 0.0;
    double angle1 = 0.0;
    double angle2 = 0.0;
    Pt from{0.0, 0.0};
    Pt to{0.0, 0.0};
};

struct PointVisibilityResult {
    std::vector<VisibleSegment> segments;
    std::vector<std::pair<double, double>> remaining_angles;
    int visited_node_count = 0;
    int processed_node_count = 0;
    int ring_count = 0;
};

struct OwnerEdge {
    int building_index = -1;
    int edge_index = -1;
};

struct SnapResult {
    Pt point{0.0, 0.0};
    double distance2 = std::numeric_limits<double>::infinity();
    int building_index = -1;
    int edge_index = -1;
    double edge_s = 0.0;
    std::array<OwnerEdge, 2> owner_edges;
    int owner_edge_count = 0;
};

static Pt sub(Pt a, Pt b) {
    return {a.x - b.x, a.y - b.y};
}

static Pt add(Pt a, Pt b) {
    return {a.x + b.x, a.y + b.y};
}

static Pt mul(Pt a, double k) {
    return {a.x * k, a.y * k};
}

static double cross(Pt a, Pt b) {
    return a.x * b.y - a.y * b.x;
}

static double dot(Pt a, Pt b) {
    return a.x * b.x + a.y * b.y;
}

static double norm_angle(double theta) {
    theta = std::fmod(theta, VIS_TWO_PI);
    if (theta < 0.0) {
        theta += VIS_TWO_PI;
    }
    return theta;
}

static double angle_of(Pt p, Pt q) {
    return norm_angle(std::atan2(q.y - p.y, q.x - p.x));
}

static bool bbox_intersects(BBox a, BBox b) {
    if (!a.valid() || !b.valid()) {
        return false;
    }
    return a.minx <= b.maxx + VIS_EPS
        && a.maxx + VIS_EPS >= b.minx
        && a.miny <= b.maxy + VIS_EPS
        && a.maxy + VIS_EPS >= b.miny;
}

static bool bbox_contains_bbox(BBox outer, BBox inner) {
    if (!outer.valid() || !inner.valid()) {
        return false;
    }
    return outer.minx <= inner.minx + VIS_EPS
        && outer.miny <= inner.miny + VIS_EPS
        && outer.maxx + VIS_EPS >= inner.maxx
        && outer.maxy + VIS_EPS >= inner.maxy;
}

static BBox bbox_expanded(BBox bbox, double distance) {
    if (!bbox.valid()) {
        return bbox;
    }
    return {
        bbox.minx - distance,
        bbox.miny - distance,
        bbox.maxx + distance,
        bbox.maxy + distance
    };
}

static BBox point_bbox(Pt p) {
    return {p.x, p.y, p.x, p.y};
}

static double point_bbox_distance2(Pt p, BBox bbox) {
    if (!bbox.valid()) {
        return std::numeric_limits<double>::infinity();
    }
    double dx = std::max({bbox.minx - p.x, 0.0, p.x - bbox.maxx});
    double dy = std::max({bbox.miny - p.y, 0.0, p.y - bbox.maxy});
    return dx * dx + dy * dy;
}

static double project_point_to_segment(Pt p, Pt a, Pt b, Pt& projected) {
    Pt ab = sub(b, a);
    double length2 = dot(ab, ab);
    if (length2 <= VIS_EPS * VIS_EPS) {
        projected = a;
        return 0.0;
    }
    double s = dot(sub(p, a), ab) / length2;
    s = std::max(0.0, std::min(1.0, s));
    projected = add(a, mul(ab, s));
    return s;
}

static bool clip_segment_to_bbox(
    Pt a,
    Pt b,
    BBox bbox,
    double& s1,
    double& s2
) {
    if (!bbox.valid()) {
        return false;
    }

    Pt d = sub(b, a);
    s1 = 0.0;
    s2 = 1.0;

    auto clip_half_plane = [&](double p, double q) {
        if (std::abs(p) <= VIS_EPS) {
            return q >= -VIS_POINT_EPS;
        }

        double r = q / p;
        if (p < 0.0) {
            s1 = std::max(s1, r);
        } else {
            s2 = std::min(s2, r);
        }
        return s1 <= s2 + VIS_POINT_EPS;
    };

    if (!clip_half_plane(-d.x, a.x - bbox.minx) ||
        !clip_half_plane( d.x, bbox.maxx - a.x) ||
        !clip_half_plane(-d.y, a.y - bbox.miny) ||
        !clip_half_plane( d.y, bbox.maxy - a.y)) {
        return false;
    }

    s1 = std::max(0.0, std::min(1.0, s1));
    s2 = std::max(0.0, std::min(1.0, s2));
    return s2 > s1 + VIS_EPS;
}

static bool point_on_segment(Pt p, Pt a, Pt b) {
    Pt ab = sub(b, a);
    Pt ap = sub(p, a);
    double length = std::hypot(ab.x, ab.y);
    if (length <= VIS_POINT_EPS) {
        return std::hypot(p.x - a.x, p.y - a.y) <= VIS_POINT_EPS;
    }

    double area = std::abs(cross(ab, ap));
    if (area > VIS_POINT_EPS * std::max(1.0, length)) {
        return false;
    }

    double minx = std::min(a.x, b.x) - VIS_POINT_EPS;
    double maxx = std::max(a.x, b.x) + VIS_POINT_EPS;
    double miny = std::min(a.y, b.y) - VIS_POINT_EPS;
    double maxy = std::max(a.y, b.y) + VIS_POINT_EPS;
    return minx <= p.x && p.x <= maxx && miny <= p.y && p.y <= maxy;
}

static bool point_in_polygon_inclusive(Pt p, const std::vector<Pt>& polygon) {
    int n = (int)polygon.size();
    if (n < 3) {
        return false;
    }

    bool inside = false;
    for (int i = 0; i < n; i++) {
        Pt a = polygon[(size_t)i];
        Pt b = polygon[(size_t)((i + 1) % n)];
        if (point_on_segment(p, a, b)) {
            return true;
        }

        bool crosses_y = (a.y > p.y) != (b.y > p.y);
        if (!crosses_y) {
            continue;
        }

        double x_at_y = a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y);
        if (p.x < x_at_y) {
            inside = !inside;
        }
    }
    return inside;
}

static bool point_in_polygon_strict(Pt p, const std::vector<Pt>& polygon) {
    int n = (int)polygon.size();
    if (n < 3) {
        return false;
    }

    bool inside = false;
    for (int i = 0; i < n; i++) {
        Pt a = polygon[(size_t)i];
        Pt b = polygon[(size_t)((i + 1) % n)];
        if (point_on_segment(p, a, b)) {
            return false;
        }

        bool crosses_y = (a.y > p.y) != (b.y > p.y);
        if (!crosses_y) {
            continue;
        }

        double x_at_y = a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y);
        if (p.x < x_at_y) {
            inside = !inside;
        }
    }
    return inside;
}

static void add_segment_intersection_params(
    Pt a,
    Pt b,
    Pt c,
    Pt d,
    std::vector<double>& params
) {
    Pt r = sub(b, a);
    Pt s = sub(d, c);
    Pt ca = sub(c, a);
    double denom = cross(r, s);
    double rr = dot(r, r);
    if (rr <= VIS_EPS) {
        return;
    }

    if (std::abs(denom) <= VIS_EPS) {
        if (std::abs(cross(ca, r)) > VIS_POINT_EPS * std::max(1.0, std::sqrt(rr))) {
            return;
        }

        double t0 = dot(sub(c, a), r) / rr;
        double t1 = dot(sub(d, a), r) / rr;
        if (t0 > t1) {
            std::swap(t0, t1);
        }
        double start = std::max(0.0, t0);
        double end = std::min(1.0, t1);
        if (end < start - VIS_POINT_EPS) {
            return;
        }
        params.push_back(std::max(0.0, std::min(1.0, start)));
        params.push_back(std::max(0.0, std::min(1.0, end)));
        return;
    }

    double t = cross(ca, s) / denom;
    double u = cross(ca, r) / denom;
    if (-VIS_POINT_EPS <= t && t <= 1.0 + VIS_POINT_EPS &&
        -VIS_POINT_EPS <= u && u <= 1.0 + VIS_POINT_EPS) {
        params.push_back(std::max(0.0, std::min(1.0, t)));
    }
}

static bool segment_crosses_polygon_interior(Pt a, Pt b, const std::vector<Pt>& polygon) {
    if (std::hypot(b.x - a.x, b.y - a.y) <= VIS_POINT_EPS) {
        return false;
    }

    std::vector<double> params{0.0, 1.0};
    int n = (int)polygon.size();
    for (int i = 0; i < n; i++) {
        Pt c = polygon[(size_t)i];
        Pt d = polygon[(size_t)((i + 1) % n)];
        add_segment_intersection_params(a, b, c, d, params);
    }

    std::sort(params.begin(), params.end());
    std::vector<double> unique_params;
    for (double t : params) {
        if (unique_params.empty() || std::abs(t - unique_params.back()) > 1e-8) {
            unique_params.push_back(t);
        }
    }

    for (size_t i = 0; i + 1 < unique_params.size(); i++) {
        double t0 = unique_params[i];
        double t1 = unique_params[i + 1];
        if (t1 <= t0 + 1e-8) {
            continue;
        }
        double mid_t = 0.5 * (t0 + t1);
        Pt mid = add(a, mul(sub(b, a), mid_t));
        if (point_in_polygon_strict(mid, polygon)) {
            return true;
        }
    }
    return false;
}

static bool intervals_intersect(
    const std::vector<std::pair<double, double>>& lhs,
    const std::vector<std::pair<double, double>>& rhs
) {
    for (auto a : lhs) {
        for (auto b : rhs) {
            double start = std::max(a.first, b.first);
            double end = std::min(a.second, b.second);
            if (end > start + VIS_ANGLE_EPS) {
                return true;
            }
        }
    }
    return false;
}

static std::vector<std::pair<double, double>> intersect_intervals(
    const std::vector<std::pair<double, double>>& intervals,
    std::pair<double, double> target
) {
    std::vector<std::pair<double, double>> result;
    for (auto interval : intervals) {
        double start = std::max(interval.first, target.first);
        double end = std::min(interval.second, target.second);
        if (end > start + VIS_ANGLE_EPS) {
            result.push_back({start, end});
        }
    }
    return result;
}

static void subtract_interval(
    std::vector<std::pair<double, double>>& intervals,
    std::pair<double, double> remove
) {
    std::vector<std::pair<double, double>> result;
    for (auto interval : intervals) {
        double start = interval.first;
        double end = interval.second;
        if (remove.second <= start || remove.first >= end) {
            result.push_back(interval);
            continue;
        }
        if (remove.first > start + VIS_ANGLE_EPS) {
            result.push_back({start, remove.first});
        }
        if (remove.second < end - VIS_ANGLE_EPS) {
            result.push_back({remove.second, end});
        }
    }
    intervals.swap(result);
}

static std::vector<std::pair<double, double>> bbox_angle_intervals(Pt p, BBox bbox) {
    if (!bbox.valid()) {
        return {};
    }

    if (bbox.contains(p)) {
        return {{0.0, VIS_TWO_PI}};
    }

    std::array<double, 4> angles{{
        angle_of(p, {bbox.minx, bbox.miny}),
        angle_of(p, {bbox.minx, bbox.maxy}),
        angle_of(p, {bbox.maxx, bbox.miny}),
        angle_of(p, {bbox.maxx, bbox.maxy})
    }};
    std::sort(angles.begin(), angles.end());

    double largest_gap = -1.0;
    int gap_index = 0;
    for (int i = 0; i < 4; i++) {
        double cur = angles[i];
        double next = (i == 3) ? angles[0] + VIS_TWO_PI : angles[i + 1];
        double gap = next - cur;
        if (gap > largest_gap) {
            largest_gap = gap;
            gap_index = i;
        }
    }

    double start = angles[(gap_index + 1) % 4];
    double end = angles[gap_index];
    double arc_len = VIS_TWO_PI - largest_gap;
    if (arc_len >= VIS_TWO_PI - VIS_ANGLE_EPS) {
        return {{0.0, VIS_TWO_PI}};
    }

    if (start <= end) {
        return {{start, end}};
    }
    return {{start, VIS_TWO_PI}, {0.0, end}};
}

static RayHit no_hit() {
    return {false, 0.0, 0.0, {0.0, 0.0}};
}

static RayHit ray_segment_intersection(Pt p, double theta, Pt a, Pt b) {
    theta = norm_angle(theta);
    Pt d{std::cos(theta), std::sin(theta)};
    Pt v = sub(b, a);
    Pt w = sub(a, p);
    double denom = cross(d, v);

    if (std::abs(denom) < VIS_EPS) {
        if (std::abs(cross(w, d)) >= VIS_EPS ||
            std::abs(cross(sub(b, p), d)) >= VIS_EPS) {
            return no_hit();
        }

        double t_a = dot(sub(a, p), d);
        double t_b = dot(sub(b, p), d);
        double t_min = std::min(t_a, t_b);
        double t_max = std::max(t_a, t_b);
        if (t_max < -VIS_EPS) {
            return no_hit();
        }

        double t_hit = std::max(0.0, t_min);
        double s_hit = 0.0;
        if (std::abs(t_b - t_a) >= VIS_EPS) {
            s_hit = (t_hit - t_a) / (t_b - t_a);
        }
        s_hit = std::max(0.0, std::min(1.0, s_hit));
        return {true, t_hit, s_hit, add(p, mul(d, t_hit))};
    }

    double t = cross(w, v) / denom;
    double s = cross(w, d) / denom;
    if (t >= -VIS_EPS && -VIS_EPS <= s && s <= 1.0 + VIS_EPS) {
        return {true, t, s, add(p, mul(d, t))};
    }
    return no_hit();
}

static std::vector<std::pair<double, double>> edge_angle_intervals(Pt p, Pt a, Pt b) {
    double ang_a = angle_of(p, a);
    double ang_b = angle_of(p, b);
    std::vector<std::pair<double, double>> intervals;

    Pt ap = sub(a, p);
    Pt bp = sub(b, p);
    if (std::abs(cross(ap, bp)) < VIS_EPS && dot(ap, bp) < -VIS_EPS) {
        return intervals;
    }

    auto split_interval = [&](double start, double end) {
        if (start <= end) {
            intervals.push_back({start, end});
        } else {
            intervals.push_back({start, VIS_TWO_PI});
            intervals.push_back({0.0, end});
        }
    };

    auto arc_mid_ccw = [](double start, double end) {
        double length = std::fmod(end - start + VIS_TWO_PI, VIS_TWO_PI);
        return norm_angle(start + 0.5 * length);
    };

    for (auto arc : {std::make_pair(ang_a, ang_b), std::make_pair(ang_b, ang_a)}) {
        double start = arc.first;
        double end = arc.second;
        double arc_len = std::fmod(end - start + VIS_TWO_PI, VIS_TWO_PI);
        if (arc_len < VIS_EPS) {
            continue;
        }

        double mid = arc_mid_ccw(start, end);
        RayHit hit = ray_segment_intersection(p, mid, a, b);
        if (!hit.hit || hit.t < -VIS_EPS) {
            continue;
        }
        split_interval(start, end);
    }

    std::vector<std::pair<double, double>> uniq;
    for (auto interval : intervals) {
        bool exists = false;
        for (auto old : uniq) {
            if (std::abs(interval.first - old.first) < VIS_EPS &&
                std::abs(interval.second - old.second) < VIS_EPS) {
                exists = true;
                break;
            }
        }
        if (!exists) {
            uniq.push_back(interval);
        }
    }
    return uniq;
}

class SpatialTree {
public:
    SpatialTree() = default;

    SpatialTree(
        const std::vector<std::vector<Pt>>& polygons,
        int max_leaf_size
    ) : max_leaf_size_(std::max(1, max_leaf_size)) {
        buildings_.reserve(polygons.size());
        for (const auto& polygon : polygons) {
            BBox bbox = polygon_bbox(polygon);
            buildings_.push_back({polygon_center(polygon, bbox), bbox});
        }

        indices_.resize(buildings_.size());
        std::iota(indices_.begin(), indices_.end(), 0);
        if (!indices_.empty()) {
            build_node(0, (int)indices_.size(), 0, BBox::empty());
        }
    }

    int num_buildings() const {
        return (int)buildings_.size();
    }

    int num_nodes() const {
        return (int)nodes_.size();
    }

    int num_leaves() const {
        int count = 0;
        for (const TreeNode& node : nodes_) {
            if (node.is_leaf()) {
                count++;
            }
        }
        return count;
    }

    BBox root_bbox() const {
        if (nodes_.empty()) {
            return BBox::empty();
        }
        return nodes_[0].bbox;
    }

    BBox node_bbox(int node_id) const {
        return nodes_.at(node_id).bbox;
    }

    const TreeNode& node(int node_id) const {
        return nodes_.at(node_id);
    }

    std::vector<int> node_buildings(int node_id) const {
        const TreeNode& n = nodes_.at(node_id);
        return std::vector<int>(indices_.begin() + n.start, indices_.begin() + n.end);
    }

    void query_leaf_nodes_intersecting_bbox(BBox query, std::vector<int>& out) const {
        if (!nodes_.empty()) {
            query_leaf_nodes_intersecting_bbox_recursive(0, query, out);
        }
    }

    std::vector<std::vector<double>> boxes(bool leaves_only, bool modified_bbox) const {
        std::vector<std::vector<double>> result;
        for (const TreeNode& n : nodes_) {
            if (leaves_only && !n.is_leaf()) {
                continue;
            }
            BBox bbox = modified_bbox ? n.bbox : n.cell_bbox;
            result.push_back({
                (double)n.id,
                (double)n.children[0],
                (double)n.children[1],
                (double)n.children[2],
                (double)n.children[3],
                (double)n.start,
                (double)n.end,
                (double)n.depth,
                n.split_x,
                n.split_y,
                bbox.minx,
                bbox.miny,
                bbox.maxx,
                bbox.maxy
            });
        }
        return result;
    }

private:
    int max_leaf_size_ = 64;
    std::vector<Building> buildings_;
    std::vector<int> indices_;
    std::vector<TreeNode> nodes_;

    bool can_split_node(int start, int end, double split_x, double split_y) const {
        std::array<bool, 4> has_points{{false, false, false, false}};
        int non_empty = 0;
        for (int i = start; i < end; i++) {
            Pt center = buildings_[indices_[i]].center;
            int quadrant = 0;
            if (center.x > split_x) {
                quadrant += 1;
            }
            if (center.y > split_y) {
                quadrant += 2;
            }
            if (!has_points[quadrant]) {
                has_points[quadrant] = true;
                non_empty++;
            }
        }
        return non_empty > 1;
    }

    static BBox child_cell_bbox(BBox parent, double split_x, double split_y, int quadrant) {
        BBox child = parent;
        if (quadrant & 1) {
            child.minx = split_x;
        } else {
            child.maxx = split_x;
        }
        if (quadrant & 2) {
            child.miny = split_y;
        } else {
            child.maxy = split_y;
        }
        return child;
    }

    int build_node(int start, int end, int depth, BBox cell_bbox) {
        int node_id = (int)nodes_.size();
        nodes_.push_back(TreeNode{});

        TreeNode node;
        node.id = node_id;
        node.start = start;
        node.end = end;
        node.depth = depth;

        BBox building_bbox = BBox::empty();
        for (int i = start; i < end; i++) {
            const Building& building = buildings_[indices_[i]];
            node.center_bbox.expand(building.center);
            building_bbox.expand(building.bbox);
        }

        node.cell_bbox = cell_bbox.valid() ? cell_bbox : node.center_bbox;
        node.bbox = node.cell_bbox;
        node.bbox.expand(building_bbox);

        int count = end - start;
        node.split_x = (node.center_bbox.minx + node.center_bbox.maxx) * 0.5;
        node.split_y = (node.center_bbox.miny + node.center_bbox.maxy) * 0.5;

        if (count > max_leaf_size_ && can_split_node(start, end, node.split_x, node.split_y)) {
            std::array<std::vector<int>, 4> quadrants;
            for (int i = start; i < end; i++) {
                int building_id = indices_[i];
                Pt center = buildings_[building_id].center;
                int quadrant = 0;
                if (center.x > node.split_x) {
                    quadrant += 1;
                }
                if (center.y > node.split_y) {
                    quadrant += 2;
                }
                quadrants[quadrant].push_back(building_id);
            }

            int write = start;
            for (int quadrant = 0; quadrant < 4; quadrant++) {
                if (quadrants[quadrant].empty()) {
                    continue;
                }
                int child_start = write;
                for (int building_id : quadrants[quadrant]) {
                    indices_[write++] = building_id;
                }
                int child_end = write;
                node.children[node.child_count] = build_node(
                    child_start,
                    child_end,
                    depth + 1,
                    child_cell_bbox(node.cell_bbox, node.split_x, node.split_y, quadrant)
                );
                node.child_count++;
            }
        }

        nodes_[node_id] = node;
        return node_id;
    }

    void query_leaf_nodes_intersecting_bbox_recursive(
        int node_id,
        BBox query,
        std::vector<int>& out
    ) const {
        const TreeNode& n = nodes_[node_id];
        if (!bbox_intersects(n.bbox, query)) {
            return;
        }
        if (n.is_leaf()) {
            out.push_back(node_id);
            return;
        }
        for (int i = 0; i < n.child_count; i++) {
            query_leaf_nodes_intersecting_bbox_recursive(n.children[i], query, out);
        }
    }
};

static std::vector<std::vector<Pt>> polygons_from_numpy(
    py::array_t<double> points_arr,
    py::array_t<int> offsets_arr
) {
    auto points = points_arr.unchecked<2>();
    auto offsets = offsets_arr.unchecked<1>();

    if (points.shape(1) != 2) {
        throw std::invalid_argument("points must have shape (n, 2)");
    }
    if (offsets.shape(0) < 1) {
        throw std::invalid_argument("offsets must contain at least one value");
    }

    int num_buildings = (int)offsets.shape(0) - 1;
    std::vector<std::vector<Pt>> polygons;
    polygons.reserve(num_buildings);

    for (int b = 0; b < num_buildings; b++) {
        int begin = offsets(b);
        int end = offsets(b + 1);
        if (begin < 0 || end < begin || end > points.shape(0)) {
            throw std::invalid_argument("offsets are not valid for points");
        }

        std::vector<Pt> polygon;
        polygon.reserve(end - begin);
        for (int i = begin; i < end; i++) {
            polygon.push_back({points(i, 0), points(i, 1)});
        }
        polygons.push_back(std::move(polygon));
    }

    return polygons;
}

class VisibilityEngine {
public:
    VisibilityEngine(
        py::array_t<double> points_arr,
        py::array_t<int> offsets_arr,
        int max_leaf_size = 64,
        double ring_step = 19.0
    ) : polygons_(polygons_from_numpy(points_arr, offsets_arr)),
        tree_(polygons_, max_leaf_size),
        ring_step_(std::max(0.001, ring_step)) {
        build_geometry_metrics();
    }

    int num_buildings() const {
        return tree_.num_buildings();
    }

    int num_nodes() const {
        return tree_.num_nodes();
    }

    int num_leaves() const {
        return tree_.num_leaves();
    }

    py::dict visible_from_point(double x, double y, int max_rings = 10000) const {
        PointVisibilityResult result = visible_from_point_core({x, y}, max_rings);
        return point_result_to_dict(result);
    }

    py::dict visible_from_points(py::array_t<double> points_arr, int max_rings = 10000) const {
        auto points = points_arr.unchecked<2>();
        if (points.shape(1) != 2) {
            throw std::invalid_argument("points must have shape (n, 2)");
        }

        py::ssize_t n = points.shape(0);
        std::vector<long long> offsets((size_t)n + 1, 0);
        std::vector<int> building_ids;
        std::vector<int> edge_indices;
        std::vector<double> segment_data;
        std::vector<int> visited_node_counts((size_t)n, 0);
        std::vector<int> processed_node_counts((size_t)n, 0);
        std::vector<int> ring_counts((size_t)n, 0);

        for (py::ssize_t i = 0; i < n; i++) {
            PointVisibilityResult result = visible_from_point_core(
                {points(i, 0), points(i, 1)},
                max_rings
            );
            offsets[(size_t)i] = (long long)building_ids.size();
            visited_node_counts[(size_t)i] = result.visited_node_count;
            processed_node_counts[(size_t)i] = result.processed_node_count;
            ring_counts[(size_t)i] = result.ring_count;

            for (const VisibleSegment& seg : result.segments) {
                building_ids.push_back(seg.building_index);
                edge_indices.push_back(seg.edge_index);
                segment_data.insert(
                    segment_data.end(),
                    {
                        seg.s1,
                        seg.s2,
                        seg.angle1,
                        seg.angle2,
                        seg.from.x,
                        seg.from.y,
                        seg.to.x,
                        seg.to.y
                    }
                );
            }
        }
        offsets[(size_t)n] = (long long)building_ids.size();

        py::dict out;
        out["point_offsets"] = vector_to_numpy<long long>(offsets);
        out["building_ids"] = vector_to_numpy<int>(building_ids);
        out["edge_indices"] = vector_to_numpy<int>(edge_indices);
        out["segment_data"] = flat_vector_to_numpy(segment_data, (py::ssize_t)building_ids.size(), 8);
        add_geometry_metrics(out);
        out["visited_node_counts"] = vector_to_numpy<int>(visited_node_counts);
        out["processed_node_counts"] = vector_to_numpy<int>(processed_node_counts);
        out["ring_counts"] = vector_to_numpy<int>(ring_counts);
        return out;
    }

    py::array_t<double> leaf_boxes(bool modified_bbox = true) const {
        return boxes_to_numpy(tree_.boxes(true, modified_bbox));
    }

private:
    std::vector<std::vector<Pt>> polygons_;
    SpatialTree tree_;
    std::vector<double> building_perimeters_;
    std::vector<int> edge_offsets_;
    std::vector<double> edge_lengths_;
    double ring_step_ = 19.0;

    void build_geometry_metrics() {
        building_perimeters_.clear();
        edge_offsets_.clear();
        edge_lengths_.clear();

        building_perimeters_.reserve(polygons_.size());
        edge_offsets_.reserve(polygons_.size() + 1);
        edge_offsets_.push_back(0);

        for (const auto& polygon : polygons_) {
            double perimeter = 0.0;
            for (size_t edge_index = 0; edge_index < polygon.size(); edge_index++) {
                const Pt& a = polygon[edge_index];
                const Pt& b = polygon[(edge_index + 1) % polygon.size()];
                double edge_length = std::hypot(b.x - a.x, b.y - a.y);
                edge_lengths_.push_back(edge_length);
                perimeter += edge_length;
            }
            building_perimeters_.push_back(perimeter);
            edge_offsets_.push_back(static_cast<int>(edge_lengths_.size()));
        }
    }

    void add_geometry_metrics(py::dict& out) const {
        out["building_perimeters"] = vector_to_numpy<double>(building_perimeters_);
        out["edge_offsets"] = vector_to_numpy<int>(edge_offsets_);
        out["edge_lengths"] = vector_to_numpy<double>(edge_lengths_);
    }

    static py::array_t<double> boxes_to_numpy(const std::vector<std::vector<double>>& rows) {
        py::array_t<double> result({(py::ssize_t)rows.size(), (py::ssize_t)14});
        auto out = result.mutable_unchecked<2>();
        for (py::ssize_t i = 0; i < (py::ssize_t)rows.size(); i++) {
            for (py::ssize_t j = 0; j < 14; j++) {
                out(i, j) = rows[(size_t)i][(size_t)j];
            }
        }
        return result;
    }

    template <typename T>
    static py::array_t<T> vector_to_numpy(const std::vector<T>& values) {
        py::array_t<T> result((py::ssize_t)values.size());
        auto out = result.template mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < (py::ssize_t)values.size(); i++) {
            out(i) = values[(size_t)i];
        }
        return result;
    }

    static py::array_t<double> flat_vector_to_numpy(
        const std::vector<double>& values,
        py::ssize_t rows,
        py::ssize_t cols
    ) {
        py::array_t<double> result({rows, cols});
        auto out = result.mutable_unchecked<2>();
        for (py::ssize_t i = 0; i < rows; i++) {
            for (py::ssize_t j = 0; j < cols; j++) {
                out(i, j) = values[(size_t)(i * cols + j)];
            }
        }
        return result;
    }

    py::dict point_result_to_dict(const PointVisibilityResult& result) const {
        std::vector<int> building_ids;
        std::vector<int> edge_indices;
        std::vector<double> segment_data;
        building_ids.reserve(result.segments.size());
        edge_indices.reserve(result.segments.size());
        segment_data.reserve(result.segments.size() * 8);

        for (const VisibleSegment& seg : result.segments) {
            building_ids.push_back(seg.building_index);
            edge_indices.push_back(seg.edge_index);
            segment_data.insert(
                segment_data.end(),
                {
                    seg.s1,
                    seg.s2,
                    seg.angle1,
                    seg.angle2,
                    seg.from.x,
                    seg.from.y,
                    seg.to.x,
                    seg.to.y
                }
            );
        }

        std::vector<double> remaining_flat;
        remaining_flat.reserve(result.remaining_angles.size() * 2);
        for (auto interval : result.remaining_angles) {
            remaining_flat.push_back(interval.first);
            remaining_flat.push_back(interval.second);
        }

        py::dict out;
        out["building_ids"] = vector_to_numpy<int>(building_ids);
        out["edge_indices"] = vector_to_numpy<int>(edge_indices);
        out["segment_data"] = flat_vector_to_numpy(segment_data, (py::ssize_t)building_ids.size(), 8);
        add_geometry_metrics(out);
        out["remaining_angles"] = flat_vector_to_numpy(
            remaining_flat,
            (py::ssize_t)result.remaining_angles.size(),
            2
        );
        out["visited_node_count"] = result.visited_node_count;
        out["processed_node_count"] = result.processed_node_count;
        out["ring_count"] = result.ring_count;
        return out;
    }

    SnapResult snap_to_nearest_boundary(Pt input) const {
        SnapResult best;
        if (tree_.num_nodes() == 0) {
            best.point = input;
            return best;
        }

        struct NodeDistance {
            double distance2;
            int node_id;
        };
        auto farther = [](const NodeDistance& lhs, const NodeDistance& rhs) {
            return lhs.distance2 > rhs.distance2;
        };
        std::priority_queue<
            NodeDistance,
            std::vector<NodeDistance>,
            decltype(farther)
        > queue(farther);
        queue.push({point_bbox_distance2(input, tree_.root_bbox()), 0});

        while (!queue.empty()) {
            NodeDistance current = queue.top();
            queue.pop();
            if (current.distance2 > best.distance2 + VIS_EPS * VIS_EPS) {
                break;
            }

            const TreeNode& node = tree_.node(current.node_id);
            if (!node.is_leaf()) {
                for (int i = 0; i < node.child_count; i++) {
                    int child_id = node.children[i];
                    double lower_bound = point_bbox_distance2(
                        input,
                        tree_.node_bbox(child_id)
                    );
                    if (lower_bound <= best.distance2 + VIS_EPS * VIS_EPS) {
                        queue.push({lower_bound, child_id});
                    }
                }
                continue;
            }

            for (int building_index : tree_.node_buildings(current.node_id)) {
                const auto& polygon = polygons_[(size_t)building_index];
                for (int edge_index = 0; edge_index < (int)polygon.size(); edge_index++) {
                    Pt a = polygon[(size_t)edge_index];
                    Pt b = polygon[(size_t)((edge_index + 1) % polygon.size())];
                    Pt projected;
                    double s = project_point_to_segment(input, a, b, projected);
                    Pt delta = sub(input, projected);
                    double distance2 = dot(delta, delta);
                    if (distance2 < best.distance2 - VIS_EPS * VIS_EPS ||
                        (std::abs(distance2 - best.distance2) <= VIS_EPS * VIS_EPS &&
                         (best.building_index < 0 || building_index < best.building_index ||
                          (building_index == best.building_index && edge_index < best.edge_index)))) {
                        best.point = projected;
                        best.distance2 = distance2;
                        best.building_index = building_index;
                        best.edge_index = edge_index;
                        best.edge_s = s;
                    }
                }
            }
        }

        if (best.building_index < 0) {
            best.point = input;
            return best;
        }

        const auto& polygon = polygons_[(size_t)best.building_index];
        int n = (int)polygon.size();
        Pt a = polygon[(size_t)best.edge_index];
        Pt b = polygon[(size_t)((best.edge_index + 1) % n)];
        double distance_to_a = std::hypot(best.point.x - a.x, best.point.y - a.y);
        double distance_to_b = std::hypot(best.point.x - b.x, best.point.y - b.y);

        if (distance_to_a <= VIS_POINT_EPS) {
            best.point = a;
            best.owner_edges[0] = {best.building_index, best.edge_index};
            best.owner_edges[1] = {best.building_index, (best.edge_index - 1 + n) % n};
            best.owner_edge_count = 2;
        } else if (distance_to_b <= VIS_POINT_EPS) {
            best.point = b;
            best.owner_edges[0] = {best.building_index, best.edge_index};
            best.owner_edges[1] = {best.building_index, (best.edge_index + 1) % n};
            best.owner_edge_count = 2;
        } else {
            best.owner_edges[0] = {best.building_index, best.edge_index};
            best.owner_edge_count = 1;
        }
        return best;
    }

    PointVisibilityResult visible_from_point_core(Pt input, int max_rings) const {
        PointVisibilityResult result;
        result.remaining_angles.push_back({0.0, VIS_TWO_PI});

        int node_count = tree_.num_nodes();
        int leaf_count = tree_.num_leaves();
        if (node_count == 0 || leaf_count == 0) {
            return result;
        }

        SnapResult snap = snap_to_nearest_boundary(input);
        Pt p = snap.point;

        // Modified leaf bboxes overlap.  Every leaf bbox covering the antenna
        // is therefore a seed, rather than choosing one arbitrary leaf.
        std::vector<int> seed_nodes;
        tree_.query_leaf_nodes_intersecting_bbox(point_bbox(p), seed_nodes);
        std::sort(seed_nodes.begin(), seed_nodes.end());
        seed_nodes.erase(std::unique(seed_nodes.begin(), seed_nodes.end()), seed_nodes.end());

        BBox safe_bbox = BBox::empty();
        for (int node_id : seed_nodes) {
            safe_bbox.expand(tree_.node_bbox(node_id));
        }
        if (!safe_bbox.valid()) {
            // General API fallback for a query point outside every modified
            // leaf bbox.  Candidate antenna points normally have seed leaves.
            safe_bbox = point_bbox(p);
        }
        std::vector<unsigned char> visited((size_t)node_count, 0);
        std::vector<int> loaded_building_ids;
        std::unordered_set<int> containing_building_ids;
        if (snap.building_index >= 0) {
            containing_building_ids.insert(snap.building_index);
        }
        BBox root_bbox = tree_.root_bbox();

        for (int ring = 0; ring < max_rings && !result.remaining_angles.empty(); ring++) {
            // Grow from the current safe-box boundary, then let every modified
            // leaf bbox touched by that buffer determine the next spatial box.
            // A second query below loads all support leaves intersecting the
            // resulting box without recursively enlarging it in the same layer.
            BBox discovery_bbox = bbox_expanded(safe_bbox, ring_step_);
            std::vector<int> frontier_nodes;
            tree_.query_leaf_nodes_intersecting_bbox(discovery_bbox, frontier_nodes);
            std::sort(frontier_nodes.begin(), frontier_nodes.end());
            frontier_nodes.erase(
                std::unique(frontier_nodes.begin(), frontier_nodes.end()),
                frontier_nodes.end()
            );

            BBox next_safe_bbox = discovery_bbox;
            for (int node_id : frontier_nodes) {
                next_safe_bbox.expand(tree_.node_bbox(node_id));
            }
            safe_bbox = next_safe_bbox;

            std::vector<int> candidates;
            tree_.query_leaf_nodes_intersecting_bbox(safe_bbox, candidates);

            std::sort(candidates.begin(), candidates.end());
            candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

            // All leaf angle tests in this spatial layer use the same snapshot.
            const std::vector<std::pair<double, double>> layer_angles =
                result.remaining_angles;
            bool added_buildings = false;
            for (int node_id : candidates) {
                if (visited[(size_t)node_id]) {
                    continue;
                }
                visited[(size_t)node_id] = 1;
                result.visited_node_count++;

                std::vector<std::pair<double, double>> node_angles =
                    bbox_angle_intervals(p, tree_.node_bbox(node_id));
                if (!intervals_intersect(layer_angles, node_angles)) {
                    continue;
                }

                result.processed_node_count++;
                std::vector<int> ids = tree_.node_buildings(node_id);
                loaded_building_ids.insert(loaded_building_ids.end(), ids.begin(), ids.end());
                added_buildings = true;
            }

            if (added_buildings) {
                std::sort(loaded_building_ids.begin(), loaded_building_ids.end());
                loaded_building_ids.erase(
                    std::unique(loaded_building_ids.begin(), loaded_building_ids.end()),
                    loaded_building_ids.end()
                );
            }

            // Revisit all loaded buildings because an edge loaded in an earlier
            // layer may only now enter the larger safe bbox.  The edge builder
            // clips every edge to safe_bbox, so merely touching a leaf bbox can
            // never commit geometry beyond the verified spatial frontier.
            if (!loaded_building_ids.empty()) {
                visible_edges_for_buildings(
                    p,
                    loaded_building_ids,
                    containing_building_ids,
                    safe_bbox,
                    result.remaining_angles,
                    result.segments
                );
            }

            result.ring_count = ring + 1;
            if (result.remaining_angles.empty() || bbox_contains_bbox(safe_bbox, root_bbox)) {
                break;
            }
        }

        add_visible_owner_edges(p, snap, result.segments);
        merge_visible_segments(result.segments);
        return result;
    }

    std::unordered_set<int> containing_buildings(Pt p) const {
        std::unordered_set<int> result;
        std::vector<int> candidates;
        tree_.query_leaf_nodes_intersecting_bbox(point_bbox(p), candidates);

        for (int node_id : candidates) {
            std::vector<int> ids = tree_.node_buildings(node_id);
            for (int building_id : ids) {
                if (result.find(building_id) != result.end()) {
                    continue;
                }
                if (building_id < 0 || building_id >= (int)polygons_.size()) {
                    continue;
                }
                if (point_in_polygon_inclusive(p, polygons_[(size_t)building_id])) {
                    result.insert(building_id);
                }
            }
        }
        return result;
    }

    void add_visible_owner_edges(
        Pt p,
        const SnapResult& snap,
        std::vector<VisibleSegment>& output
    ) const {
        for (int owner_index = 0; owner_index < snap.owner_edge_count; owner_index++) {
            int building_id = snap.owner_edges[(size_t)owner_index].building_index;
            int edge_index = snap.owner_edges[(size_t)owner_index].edge_index;
            if (building_id < 0 || building_id >= (int)polygons_.size()) {
                continue;
            }

            const auto& polygon = polygons_[(size_t)building_id];
            int n = (int)polygon.size();
            if (edge_index < 0 || edge_index >= n) {
                continue;
            }
            Pt a = polygon[(size_t)edge_index];
            Pt b = polygon[(size_t)((edge_index + 1) % n)];
            if (std::hypot(b.x - a.x, b.y - a.y) <= VIS_POINT_EPS) {
                continue;
            }

            // GIS CUP: touching or travelling along a boundary is not blocked.
            // Only the one or two edges that actually own the snapped point need
            // this t=0 compensation; every other edge stays in the global sweep.
            double angle1 = angle_of(p, a);
            double angle2 = angle_of(p, b);
            output.push_back({
                building_id,
                edge_index,
                0.0,
                1.0,
                angle1,
                angle2,
                a,
                b
            });
        }
    }

    std::vector<Edge> build_edges_for_buildings(
        const std::vector<int>& building_ids,
        BBox safe_bbox
    ) const {
        std::vector<Edge> edges;
        for (int building_id : building_ids) {
            if (building_id < 0 || building_id >= (int)polygons_.size()) {
                continue;
            }
            const auto& polygon = polygons_[(size_t)building_id];
            int n = (int)polygon.size();
            for (int edge_index = 0; edge_index < n; edge_index++) {
                Pt source_a = polygon[(size_t)edge_index];
                Pt source_b = polygon[(size_t)((edge_index + 1) % n)];
                double source_s1 = 0.0;
                double source_s2 = 1.0;
                if (!clip_segment_to_bbox(
                        source_a,
                        source_b,
                        safe_bbox,
                        source_s1,
                        source_s2
                    )) {
                    continue;
                }

                Pt edge_vector = sub(source_b, source_a);
                edges.push_back({
                    building_id,
                    edge_index,
                    add(source_a, mul(edge_vector, source_s1)),
                    add(source_a, mul(edge_vector, source_s2)),
                    source_s1,
                    source_s2
                });
            }
        }
        return edges;
    }

    static std::vector<Event> build_events_from_edges(Pt p, const std::vector<Edge>& edges) {
        std::vector<Event> events;
        for (int edge_index = 0; edge_index < (int)edges.size(); edge_index++) {
            const Edge& edge = edges[(size_t)edge_index];
            auto intervals = edge_angle_intervals(p, edge.a, edge.b);
            for (auto interval : intervals) {
                events.push_back({interval.first, 1, edge_index});
                events.push_back({interval.second, -1, edge_index});
            }
        }

        std::sort(events.begin(), events.end(), [](const Event& lhs, const Event& rhs) {
            if (std::abs(lhs.angle - rhs.angle) > VIS_EPS) {
                return lhs.angle < rhs.angle;
            }
            return lhs.type < rhs.type;
        });
        return events;
    }

    static int nearest_edge_from_edges(
        Pt p,
        const std::vector<Edge>& edges,
        double theta,
        const std::set<int>& active_edges
    ) {
        int best_edge = -1;
        double best_t = std::numeric_limits<double>::infinity();
        for (int edge_index : active_edges) {
            const Edge& edge = edges[(size_t)edge_index];
            RayHit hit = ray_segment_intersection(p, theta, edge.a, edge.b);
            if (!hit.hit || hit.t <= VIS_EPS) {
                continue;
            }
            if (hit.t < best_t - VIS_EPS ||
                (std::abs(hit.t - best_t) <= VIS_EPS && (best_edge < 0 || edge_index < best_edge))) {
                best_t = hit.t;
                best_edge = edge_index;
            }
        }
        return best_edge;
    }

    static bool is_frontmost_edge_at_angle(
        Pt p,
        const std::vector<Edge>& edges,
        int edge_index,
        double theta,
        const std::set<int>& active_edges
    ) {
        int best_edge = nearest_edge_from_edges(p, edges, theta, active_edges);
        return best_edge == edge_index;
    }

    static void record_visible_segment(
        Pt p,
        const std::vector<Edge>& edges,
        int edge_index,
        double theta1,
        double theta2,
        std::vector<VisibleSegment>& output,
        const std::set<int>& active_edges,
        const std::unordered_set<int>& containing_building_ids,
        const std::vector<std::vector<Pt>>& polygons,
        const std::vector<std::pair<double, double>>& available_angles,
        std::vector<std::pair<double, double>>& claimed_angles
    ) {
        if (edge_index < 0 || theta2 <= theta1 + VIS_EPS || theta2 - theta1 < VIS_ANGLE_EPS) {
            return;
        }

        auto overlaps = intersect_intervals(available_angles, {theta1, theta2});
        if (overlaps.empty()) {
            return;
        }

        const Edge& edge = edges[(size_t)edge_index];
        for (auto overlap : overlaps) {
            double mid_theta = 0.5 * (overlap.first + overlap.second);
            if (!is_frontmost_edge_at_angle(p, edges, edge_index, mid_theta, active_edges)) {
                continue;
            }

            double left_theta = overlap.first + VIS_ANGLE_EPS;
            double right_theta = overlap.second - VIS_ANGLE_EPS;
            if (right_theta <= left_theta) {
                continue;
            }

            RayHit hit1 = ray_segment_intersection(p, left_theta, edge.a, edge.b);
            RayHit hit2 = ray_segment_intersection(p, right_theta, edge.a, edge.b);
            if (!hit1.hit || !hit2.hit) {
                continue;
            }

            double clipped_s1 = std::max(0.0, std::min(1.0, hit1.s));
            double clipped_s2 = std::max(0.0, std::min(1.0, hit2.s));
            if (std::abs(clipped_s1 - clipped_s2) <= VIS_EPS) {
                continue;
            }

            double clipped_s_min = std::min(clipped_s1, clipped_s2);
            double clipped_s_max = std::max(clipped_s1, clipped_s2);
            Pt edge_vec = sub(edge.b, edge.a);
            Pt from = add(edge.a, mul(edge_vec, clipped_s_min));
            Pt to = add(edge.a, mul(edge_vec, clipped_s_max));
            double source_span = edge.source_s2 - edge.source_s1;
            double s_min = edge.source_s1 + source_span * clipped_s_min;
            double s_max = edge.source_s1 + source_span * clipped_s_max;

            // Claim only after a real, frontmost segment inside safe_bbox has
            // been found.  Claims are applied after the complete batch sweep.
            claimed_angles.push_back(overlap);
            if (containing_building_ids.find(edge.building_index) != containing_building_ids.end()) {
                Pt mid = mul(add(from, to), 0.5);
                const auto& polygon = polygons[(size_t)edge.building_index];
                if (segment_crosses_polygon_interior(p, mid, polygon)) {
                    continue;
                }
            }
            output.push_back({
                edge.building_index,
                edge.edge_index,
                s_min,
                s_max,
                overlap.first,
                overlap.second,
                from,
                to
            });
        }
    }

    void visible_edges_for_buildings(
        Pt p,
        const std::vector<int>& building_ids,
        const std::unordered_set<int>& containing_building_ids,
        BBox safe_bbox,
        std::vector<std::pair<double, double>>& available_angles,
        std::vector<VisibleSegment>& output
    ) const {
        if (building_ids.empty() || available_angles.empty()) {
            return;
        }

        std::vector<Edge> edges = build_edges_for_buildings(building_ids, safe_bbox);
        if (edges.empty()) {
            return;
        }
        std::vector<Event> events = build_events_from_edges(p, edges);
        const std::vector<std::pair<double, double>> batch_angles = available_angles;
        std::vector<std::pair<double, double>> claimed_angles;
        std::set<int> active_edges;
        double prev_angle = 0.0;
        int prev_edge = -1;

        int i = 0;
        int event_count = (int)events.size();
        while (i < event_count && !batch_angles.empty()) {
            double theta = events[(size_t)i].angle;
            record_visible_segment(
                p,
                edges,
                prev_edge,
                prev_angle,
                theta,
                output,
                active_edges,
                containing_building_ids,
                polygons_,
                batch_angles,
                claimed_angles
            );

            std::vector<Event> group;
            while (i < event_count && std::abs(events[(size_t)i].angle - theta) < VIS_ANGLE_EPS) {
                group.push_back(events[(size_t)i]);
                i++;
            }

            for (const Event& event : group) {
                if (event.type == -1) {
                    active_edges.erase(event.edge);
                }
            }
            for (const Event& event : group) {
                if (event.type == 1) {
                    active_edges.insert(event.edge);
                }
            }

            double query_theta = theta + VIS_ANGLE_EPS;
            if (query_theta >= VIS_TWO_PI) {
                query_theta -= VIS_TWO_PI;
            }
            prev_edge = nearest_edge_from_edges(p, edges, query_theta, active_edges);
            prev_angle = theta;
        }

        record_visible_segment(
            p,
            edges,
            prev_edge,
            prev_angle,
            VIS_TWO_PI,
            output,
            active_edges,
            containing_building_ids,
            polygons_,
            batch_angles,
            claimed_angles
        );

        for (auto claim : claimed_angles) {
            subtract_interval(available_angles, claim);
        }
    }

    static void merge_visible_segments(std::vector<VisibleSegment>& segments) {
        std::sort(segments.begin(), segments.end(), [](const VisibleSegment& lhs, const VisibleSegment& rhs) {
            if (lhs.building_index != rhs.building_index) {
                return lhs.building_index < rhs.building_index;
            }
            if (lhs.edge_index != rhs.edge_index) {
                return lhs.edge_index < rhs.edge_index;
            }
            return lhs.s1 < rhs.s1;
        });

        std::vector<VisibleSegment> merged;
        for (const VisibleSegment& seg : segments) {
            if (std::abs(seg.s2 - seg.s1) <= VIS_EPS) {
                continue;
            }
            if (!merged.empty()) {
                VisibleSegment& cur = merged.back();
                if (cur.building_index == seg.building_index &&
                    cur.edge_index == seg.edge_index &&
                    seg.s1 <= cur.s2 + 1e-6) {
                    if (seg.s2 > cur.s2) {
                        cur.s2 = seg.s2;
                        cur.to = seg.to;
                        cur.angle2 = std::max(cur.angle2, seg.angle2);
                    }
                    continue;
                }
            }
            merged.push_back(seg);
        }
        segments.swap(merged);
    }
};

PYBIND11_MODULE(visibility_tree_cpp, m) {
    m.doc() = "Quadtree-guided radial visibility engine for building polygons";

    py::class_<VisibilityEngine>(m, "VisibilityEngine")
        .def(
            py::init<py::array_t<double>, py::array_t<int>, int, double>(),
            py::arg("points"),
            py::arg("offsets"),
            py::arg("max_leaf_size") = 64,
            py::arg("ring_step") = 19.0
        )
        .def("visible_from_point", &VisibilityEngine::visible_from_point, py::arg("x"), py::arg("y"), py::arg("max_rings") = 10000)
        .def("visible_from_points", &VisibilityEngine::visible_from_points, py::arg("points"), py::arg("max_rings") = 10000)
        .def("leaf_boxes", &VisibilityEngine::leaf_boxes, py::arg("modified_bbox") = true)
        .def_property_readonly("num_buildings", &VisibilityEngine::num_buildings)
        .def_property_readonly("num_nodes", &VisibilityEngine::num_nodes)
        .def_property_readonly("num_leaves", &VisibilityEngine::num_leaves);
}
