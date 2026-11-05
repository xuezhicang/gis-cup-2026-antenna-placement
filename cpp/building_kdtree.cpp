#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "geometry.hpp"

// Default build: Python module callable from Python.
// Build with -DDEBUG_MAIN to compile a standalone C++ executable for quick tests.
#ifdef DEBUG_MAIN
#include <iomanip>
#include <iostream>
#else
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#endif

struct Node {
    int id = -1;
    std::array<int, 4> children{{-1, -1, -1, -1}};

    int child_count = 0;
    int start = 0;
    int end = 0;
    int depth = 0;
    double split_x = 0.0;
    double split_y = 0.0;
    BBox center_bbox = BBox::empty(); // the box of center points
    BBox cell_bbox = BBox::empty(); // the original node boundary from tree splits
    BBox bbox = BBox::empty(); // cell_bbox expanded outward by building bboxes

    bool is_leaf() const {
        return child_count == 0;
    }
};

class BuildingKDTree {
public:
    BuildingKDTree(
        const std::vector<std::vector<Pt>>& polygons,
        int max_leaf_size = 64
    ) : max_leaf_size_(std::max(1, max_leaf_size)) {
        buildings_.reserve(polygons.size());

        for (const auto& polygon : polygons) {
            BBox bbox = polygon_bbox(polygon);
            buildings_.push_back({polygon_center(polygon, bbox), bbox});
        }

        indices_.resize(buildings_.size());
        std::iota(indices_.begin(), indices_.end(), 0);
        
        // if there are buildings, build the tree recursively starting from the root node
        //if there are no buildings, the tree will remain empty
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
        for (const Node& node : nodes_) {
            if (node.is_leaf()) {
                count++;
            }
        }
        return count;
    }

    std::vector<int> node_buildings(int node_id) const {
        if (node_id < 0 || node_id >= (int)nodes_.size()) {
            throw std::out_of_range("node_id is out of range");
        }

        const Node& node = nodes_[node_id];
        return std::vector<int>(
            indices_.begin() + node.start,
            indices_.begin() + node.end
        );
    }

    std::vector<int> query_covering_nodes(Pt point, bool leaves_only = true) const {
        std::vector<int> node_ids;
        if (!nodes_.empty()) {
            query_nodes_recursive(0, point, leaves_only, node_ids);
        }
        return node_ids;
    }

    std::vector<int> query_covering_buildings(Pt point) const {
        std::vector<int> node_ids = query_covering_nodes(point, true);
        std::vector<int> building_ids;

        for (int node_id : node_ids) {
            const Node& node = nodes_[node_id];
            building_ids.insert(
                building_ids.end(),
                indices_.begin() + node.start,
                indices_.begin() + node.end
            );
        }

        std::sort(building_ids.begin(), building_ids.end());
        building_ids.erase(
            std::unique(building_ids.begin(), building_ids.end()),
            building_ids.end()
        );
        return building_ids;
    }

    std::vector<std::vector<double>> boxes(bool leaves_only, bool modified_bbox) const {
        std::vector<std::vector<double>> result;

        for (const Node& node : nodes_) {
            if (leaves_only && !node.is_leaf()) {
                continue;
            }

            BBox bbox = modified_bbox ? node.bbox : node.cell_bbox;
            result.push_back({
                (double)node.id,
                (double)node.children[0],
                (double)node.children[1],
                (double)node.children[2],
                (double)node.children[3],
                (double)node.start,
                (double)node.end,
                (double)node.depth,
                node.split_x,
                node.split_y,
                bbox.minx,
                bbox.miny,
                bbox.maxx,
                bbox.maxy
            });
        }

        return result;
    }

private:
    int max_leaf_size_;
    std::vector<Building> buildings_;
    std::vector<int> indices_;
    std::vector<Node> nodes_;

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
        nodes_.push_back(Node{});

        Node node;
        node.id = node_id;
        node.start = start;
        node.end = end;
        node.depth = depth;

        BBox building_bbox = BBox::empty();
        
        // Compute the bounding box of the building centers and the bounding box of the buildings themselves
        for (int i = start; i < end; i++) {
            const Building& building = buildings_[indices_[i]];
            node.center_bbox.expand(building.center);
            building_bbox.expand(building.bbox);
        }
        
        // If the cell_bbox is valid (not empty), use it; otherwise, use the center_bbox as the cell_bbox
        node.cell_bbox = cell_bbox.valid() ? cell_bbox : node.center_bbox;
        node.bbox = node.cell_bbox;
        node.bbox.expand(building_bbox);

        // If the number of buildings in this node exceeds the maximum leaf size and the node can be split, we will split it into quadrants
        int count = end - start;
        node.split_x = (node.center_bbox.minx + node.center_bbox.maxx) * 0.5;
        node.split_y = (node.center_bbox.miny + node.center_bbox.maxy) * 0.5;
        
        // Check if we can split the node into quadrants based on the center points of the buildings
        if (count > max_leaf_size_ && can_split_node(start, end, node.split_x, node.split_y)) {
            // Create 4 quadrants and distribute the building indices into them
            // Quadrant order: 0 = bottom-left, 1 = bottom-right, 2 = top-left, 3 = top-right
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

           // Write the building indices back into the indices_ array in quadrant order
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

    void query_nodes_recursive(
        int node_id,
        Pt point,
        bool leaves_only,
        std::vector<int>& node_ids
    ) const {
        const Node& node = nodes_[node_id];
        if (!node.bbox.contains(point)) {
            return;
        }

        if (!leaves_only || node.is_leaf()) {
            node_ids.push_back(node_id);
        }

        if (!node.is_leaf()) {
            for (int i = 0; i < node.child_count; i++) {
                query_nodes_recursive(node.children[i], point, leaves_only, node_ids);
            }
        }
    }
};

#ifndef DEBUG_MAIN
// Python module mode: accepts NumPy arrays from Python and exposes BuildingKDTree.
std::vector<std::vector<Pt>> polygons_from_numpy(
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

    int num_buildings = offsets.shape(0) - 1;
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
        polygons.push_back(polygon);
    }

    return polygons;
}

py::array_t<double> boxes_to_numpy(const std::vector<std::vector<double>>& rows) {
    py::array_t<double> result({(py::ssize_t)rows.size(), (py::ssize_t)14});
    auto out = result.mutable_unchecked<2>();

    for (py::ssize_t i = 0; i < (py::ssize_t)rows.size(); i++) {
        for (py::ssize_t j = 0; j < 14; j++) {
            out(i, j) = rows[i][j];
        }
    }

    return result;
}

class PyBuildingKDTree {
public:
    PyBuildingKDTree(
        py::array_t<double> points_arr,
        py::array_t<int> offsets_arr,
        int max_leaf_size = 64
    ) : tree_(polygons_from_numpy(points_arr, offsets_arr), max_leaf_size) {}

    // Query a single point and return a dictionary with node IDs and building indices
    py::dict query_point(double x, double y) const {
        Pt point{x, y};
        py::dict result;
        result["node_ids"] = tree_.query_covering_nodes(point, true);
        result["building_indices"] = tree_.query_covering_buildings(point);
        return result;
    }

    // Query multiple points and return a list of dictionaries for each point
    py::list query_points(py::array_t<double> points_arr) const {
        auto points = points_arr.unchecked<2>();
        if (points.shape(1) != 2) {
            throw std::invalid_argument("points must have shape (n, 2)");
        }

        py::list result;
        for (py::ssize_t i = 0; i < points.shape(0); i++) {
            result.append(query_point(points(i, 0), points(i, 1)));
        }
        return result;
    }
    
    // Query covering nodes for a single point, with an option to return only leaf nodes
    std::vector<int> query_covering_nodes(
        double x,
        double y,
        bool leaves_only = true
    ) const {
        return tree_.query_covering_nodes({x, y}, leaves_only);
    }

    std::vector<int> query_covering_buildings(double x, double y) const {
        return tree_.query_covering_buildings({x, y});
    }
    
    // Return the building indices contained in a specific node
    std::vector<int> node_buildings(int node_id) const {
        return tree_.node_buildings(node_id);
    }

    // Return the bounding boxes of nodes or leaves, optionally using modified bounding boxes
    py::array_t<double> node_boxes(bool modified_bbox = true) const {
        return boxes_to_numpy(tree_.boxes(false, modified_bbox));
    }
    
    // Return the bounding boxes of leaves, optionally using modified bounding boxes
    py::array_t<double> leaf_boxes(bool modified_bbox = true) const {
        return boxes_to_numpy(tree_.boxes(true, modified_bbox));
    }

    int num_buildings() const {
        return tree_.num_buildings();
    }
    
    // Return the number of nodes in the tree
    int num_nodes() const {
        return tree_.num_nodes();
    }
    // Return the number of leaves in the tree
    int num_leaves() const {
        return tree_.num_leaves();
    }

private:
    BuildingKDTree tree_;
};

PYBIND11_MODULE(building_kdtree_cpp, m) {
    m.doc() = "4-way spatial tree over building centroids with node boxes expanded by building bboxes";

    py::class_<PyBuildingKDTree>(m, "BuildingKDTree")
        .def(
            py::init<py::array_t<double>, py::array_t<int>, int>(),
            py::arg("points"),
            py::arg("offsets"),
            py::arg("max_leaf_size") = 64
        )
        .def("query_point", &PyBuildingKDTree::query_point)
        .def("query_points", &PyBuildingKDTree::query_points)
        .def(
            "query_covering_nodes",
            &PyBuildingKDTree::query_covering_nodes,
            py::arg("x"),
            py::arg("y"),
            py::arg("leaves_only") = true
        )
        .def("query_covering_buildings", &PyBuildingKDTree::query_covering_buildings)
        .def("node_buildings", &PyBuildingKDTree::node_buildings)
        .def(
            "node_boxes",
            &PyBuildingKDTree::node_boxes,
            py::arg("modified_bbox") = true
        )
        .def(
            "leaf_boxes",
            &PyBuildingKDTree::leaf_boxes,
            py::arg("modified_bbox") = true
        )
        .def_property_readonly("num_buildings", &PyBuildingKDTree::num_buildings)
        .def_property_readonly("num_nodes", &PyBuildingKDTree::num_nodes)
        .def_property_readonly("num_leaves", &PyBuildingKDTree::num_leaves);
}
#else


// Standalone debug mode: compile with -DDEBUG_MAIN to run this file directly.
// the given polygons are only frothe example. 
int main() {
    std::vector<std::vector<Pt>> buildings = {
        {{0.0, 0.0}, {2.0, 0.0}, {2.0, 2.0}, {0.0, 2.0}},
        {{1.0, 1.0}, {3.0, 1.0}, {3.0, 3.0}, {1.0, 3.0}},
        {{10.0, 10.0}, {12.0, 10.0}, {12.0, 12.0}, {10.0, 12.0}}
    };

    BuildingKDTree tree(buildings, 1);
    Pt query{1.5, 1.5};
    auto nodes = tree.query_covering_nodes(query, true);
    auto building_ids = tree.query_covering_buildings(query);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "debug-run example\n";
    std::cout << "buildings: " << tree.num_buildings() << '\n';
    std::cout << "nodes: " << tree.num_nodes() << '\n';
    std::cout << "leaves: " << tree.num_leaves() << '\n';

    std::cout << "\nleaf boxes after expanding by building bbox\n";
    std::cout << "node_id start end minx miny maxx maxy buildings\n";
    for (const auto& row : tree.boxes(true, true)) {
        int node_id = (int)row[0];
        std::cout
            << (int)row[0] << ' '
            << (int)row[5] << ' '
            << (int)row[6] << ' '
            << row[10] << ' '
            << row[11] << ' '
            << row[12] << ' '
            << row[13] << " [";

        auto node_building_ids = tree.node_buildings(node_id);
        for (int i = 0; i < (int)node_building_ids.size(); i++) {
            if (i > 0) {
                std::cout << ", ";
            }
            std::cout << node_building_ids[i];
        }
        std::cout << "]\n";
    }

    std::cout << "\nquery point: (" << query.x << ", " << query.y << ")\n";
    std::cout << "covering leaf nodes:";
    for (int node_id : nodes) {
        std::cout << ' ' << node_id;
    }
    std::cout << "\nreturned buildings:";
    for (int building_id : building_ids) {
        std::cout << ' ' << building_id;
    }
    std::cout << '\n';
    return 0;
}
#endif
