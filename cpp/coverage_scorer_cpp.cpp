#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

constexpr double COVERAGE_ENDPOINT_SNAP_EPS = 1e-7;
constexpr double COVERAGE_INTERVAL_MERGE_EPS = 1e-9;

using DoubleArray = py::array_t<
    double,
    py::array::c_style | py::array::forcecast
>;
using IntArray = py::array_t<
    int,
    py::array::c_style | py::array::forcecast
>;
using LongArray = py::array_t<
    long long,
    py::array::c_style | py::array::forcecast
>;

template <typename T>
py::array_t<T> vector_to_numpy(const std::vector<T>& values) {
    py::array_t<T> result(static_cast<py::ssize_t>(values.size()));
    auto out = result.template mutable_unchecked<1>();
    for (py::ssize_t index = 0; index < static_cast<py::ssize_t>(values.size()); index++) {
        out(index) = values[static_cast<size_t>(index)];
    }
    return result;
}

int score_union_coverage(
    DoubleArray polygon_points_arr,
    IntArray polygon_offsets_arr,
    IntArray segment_building_ids_arr,
    IntArray segment_edge_indices_arr,
    DoubleArray segment_ranges_arr,
    double tau
) {
    if (!std::isfinite(tau) || tau <= 0.0 || tau > 1.0) {
        throw std::invalid_argument("tau must be finite and in the interval (0, 1]");
    }
    if (polygon_points_arr.ndim() != 2 || polygon_points_arr.shape(1) != 2) {
        throw std::invalid_argument("polygon_points must have shape (n, 2)");
    }
    if (polygon_offsets_arr.ndim() != 1 || polygon_offsets_arr.shape(0) < 1) {
        throw std::invalid_argument("polygon_offsets must be a non-empty 1D array");
    }
    if (segment_building_ids_arr.ndim() != 1 || segment_edge_indices_arr.ndim() != 1) {
        throw std::invalid_argument("segment building and edge IDs must be 1D arrays");
    }
    if (segment_ranges_arr.ndim() != 2 || segment_ranges_arr.shape(1) != 2) {
        throw std::invalid_argument("segment_ranges must have shape (n, 2)");
    }

    py::ssize_t num_segments = segment_building_ids_arr.shape(0);
    if (segment_edge_indices_arr.shape(0) != num_segments ||
        segment_ranges_arr.shape(0) != num_segments) {
        throw std::invalid_argument("all segment arrays must have the same row count");
    }

    auto polygon_points = polygon_points_arr.unchecked<2>();
    auto polygon_offsets = polygon_offsets_arr.unchecked<1>();
    auto segment_building_ids = segment_building_ids_arr.unchecked<1>();
    auto segment_edge_indices = segment_edge_indices_arr.unchecked<1>();
    auto segment_ranges = segment_ranges_arr.unchecked<2>();

    int num_buildings = static_cast<int>(polygon_offsets.shape(0) - 1);
    if (polygon_offsets(0) != 0) {
        throw std::invalid_argument("polygon_offsets must start at zero");
    }
    if (polygon_offsets(num_buildings) != polygon_points.shape(0)) {
        throw std::invalid_argument("last polygon offset must equal polygon point count");
    }

    std::vector<int> edge_offsets(static_cast<size_t>(num_buildings) + 1, 0);
    std::vector<std::vector<double>> edge_lengths(static_cast<size_t>(num_buildings));
    std::vector<double> perimeters(static_cast<size_t>(num_buildings), 0.0);

    for (int building_index = 0; building_index < num_buildings; building_index++) {
        int start = polygon_offsets(building_index);
        int end = polygon_offsets(building_index + 1);
        if (start < 0 || end < start || end > polygon_points.shape(0)) {
            throw std::invalid_argument("polygon_offsets must be sorted and in bounds");
        }

        int vertex_count = end - start;
        if (vertex_count < 3) {
            throw std::invalid_argument("each building polygon must have at least 3 vertices");
        }

        edge_offsets[static_cast<size_t>(building_index + 1)] =
            edge_offsets[static_cast<size_t>(building_index)] + vertex_count;
        auto& lengths = edge_lengths[static_cast<size_t>(building_index)];
        lengths.resize(static_cast<size_t>(vertex_count));

        for (int edge_index = 0; edge_index < vertex_count; edge_index++) {
            int a_index = start + edge_index;
            int b_index = start + ((edge_index + 1) % vertex_count);
            double dx = polygon_points(b_index, 0) - polygon_points(a_index, 0);
            double dy = polygon_points(b_index, 1) - polygon_points(a_index, 1);
            double length = std::hypot(dx, dy);
            if (!std::isfinite(length)) {
                throw std::invalid_argument("polygon contains a non-finite coordinate");
            }
            lengths[static_cast<size_t>(edge_index)] = length;
            perimeters[static_cast<size_t>(building_index)] += length;
        }
    }

    int total_edges = edge_offsets.back();
    std::vector<std::vector<std::pair<double, double>>> intervals_by_edge(
        static_cast<size_t>(total_edges)
    );

    for (py::ssize_t row = 0; row < num_segments; row++) {
        int building_index = segment_building_ids(row);
        int edge_index = segment_edge_indices(row);
        if (building_index < 0 || building_index >= num_buildings) {
            throw std::invalid_argument("segment contains an invalid building_id");
        }
        if (edge_index < 0 ||
            edge_index >= static_cast<int>(edge_lengths[static_cast<size_t>(building_index)].size())) {
            throw std::invalid_argument("segment contains an invalid edge_index");
        }

        double s1 = segment_ranges(row, 0);
        double s2 = segment_ranges(row, 1);
        if (!std::isfinite(s1) || !std::isfinite(s2)) {
            throw std::invalid_argument("segment contains a non-finite s1 or s2");
        }

        double interval_start = std::clamp(std::min(s1, s2), 0.0, 1.0);
        double interval_end = std::clamp(std::max(s1, s2), 0.0, 1.0);
        if (interval_start <= COVERAGE_ENDPOINT_SNAP_EPS) {
            interval_start = 0.0;
        }
        if (interval_end >= 1.0 - COVERAGE_ENDPOINT_SNAP_EPS) {
            interval_end = 1.0;
        }
        if (interval_end <= interval_start + COVERAGE_INTERVAL_MERGE_EPS) {
            continue;
        }

        int global_edge_index =
            edge_offsets[static_cast<size_t>(building_index)] + edge_index;
        intervals_by_edge[static_cast<size_t>(global_edge_index)].push_back(
            {interval_start, interval_end}
        );
    }

    std::vector<double> visible_lengths(static_cast<size_t>(num_buildings), 0.0);

    for (int building_index = 0; building_index < num_buildings; building_index++) {
        const auto& lengths = edge_lengths[static_cast<size_t>(building_index)];
        for (int edge_index = 0; edge_index < static_cast<int>(lengths.size()); edge_index++) {
            int global_edge_index =
                edge_offsets[static_cast<size_t>(building_index)] + edge_index;
            auto& intervals = intervals_by_edge[static_cast<size_t>(global_edge_index)];
            if (intervals.empty()) {
                continue;
            }

            std::sort(intervals.begin(), intervals.end());
            double current_start = intervals[0].first;
            double current_end = intervals[0].second;
            double visible_fraction = 0.0;

            for (size_t interval_index = 1;
                 interval_index < intervals.size();
                 interval_index++) {
                double next_start = intervals[interval_index].first;
                double next_end = intervals[interval_index].second;
                if (next_start <= current_end + COVERAGE_INTERVAL_MERGE_EPS) {
                    current_end = std::max(current_end, next_end);
                    continue;
                }

                visible_fraction += current_end - current_start;
                current_start = next_start;
                current_end = next_end;
            }
            visible_fraction += current_end - current_start;
            visible_fraction = std::clamp(visible_fraction, 0.0, 1.0);
            visible_lengths[static_cast<size_t>(building_index)] +=
                lengths[static_cast<size_t>(edge_index)] * visible_fraction;
        }
    }

    int score = 0;
    for (int building_index = 0; building_index < num_buildings; building_index++) {
        double perimeter = perimeters[static_cast<size_t>(building_index)];
        if (perimeter <= 0.0) {
            continue;
        }
        double coverage = std::clamp(
            visible_lengths[static_cast<size_t>(building_index)] / perimeter,
            0.0,
            1.0
        );
        if (coverage >= tau) {
            score++;
        }
    }

    return score;
}

struct IncrementalPointScore {
    int hard_score = 0;
    double soft_score = 0.0;
    double unique_length = 0.0;
};

struct EdgeInterval {
    std::uint64_t edge_key = 0;
    double start = 0.0;
    double end = 0.0;
};

struct BuildingLengthDelta {
    int building_id = -1;
    double added_length = 0.0;
};

struct PointCoverageDelta {
    std::vector<BuildingLengthDelta> buildings;
    double unique_length = 0.0;
};

struct CoverageScorerData {
    std::vector<long long> point_ids;
    std::vector<long long> point_interval_offsets;
    std::vector<EdgeInterval> point_intervals;
    std::unordered_map<long long, size_t> point_positions;
    std::unordered_map<std::uint64_t, double> edge_lengths;
    std::vector<double> building_perimeters;
};

class IncrementalCoverageScorer {
public:
    IncrementalCoverageScorer(
        LongArray point_ids_arr,
        LongArray point_offsets_arr,
        IntArray segment_building_ids_arr,
        IntArray segment_edge_indices_arr,
        DoubleArray segment_ranges_arr,
        DoubleArray segment_edge_lengths_arr,
        DoubleArray building_perimeters_arr,
        double tau
    ) : data_(std::make_shared<CoverageScorerData>()), tau_(tau) {
        validate_tau(tau_);
        validate_input_shapes(
            point_ids_arr,
            point_offsets_arr,
            segment_building_ids_arr,
            segment_edge_indices_arr,
            segment_ranges_arr,
            segment_edge_lengths_arr,
            building_perimeters_arr
        );

        auto point_ids = point_ids_arr.unchecked<1>();
        auto point_offsets = point_offsets_arr.unchecked<1>();
        auto segment_building_ids = segment_building_ids_arr.unchecked<1>();
        auto segment_edge_indices = segment_edge_indices_arr.unchecked<1>();
        auto segment_ranges = segment_ranges_arr.unchecked<2>();
        auto segment_edge_lengths = segment_edge_lengths_arr.unchecked<1>();
        auto building_perimeters = building_perimeters_arr.unchecked<1>();

        input_segment_count_ = static_cast<long long>(segment_building_ids.shape(0));
        data_->building_perimeters.resize(
            static_cast<size_t>(building_perimeters.shape(0))
        );
        building_visible_lengths_.assign(data_->building_perimeters.size(), 0.0);
        for (py::ssize_t building_id = 0; building_id < building_perimeters.shape(0); building_id++) {
            data_->building_perimeters[static_cast<size_t>(building_id)] =
                building_perimeters(building_id);
        }

        data_->point_ids.reserve(static_cast<size_t>(point_ids.shape(0)));
        data_->point_interval_offsets.reserve(
            static_cast<size_t>(point_ids.shape(0)) + 1
        );
        data_->point_interval_offsets.push_back(0);

        long long previous_point_id = std::numeric_limits<long long>::min();
        long long expected_segment_start = 0;
        for (py::ssize_t point_position = 0; point_position < point_ids.shape(0); point_position++) {
            long long point_id = point_ids(point_position);
            if (point_position > 0 && point_id <= previous_point_id) {
                throw std::invalid_argument("point_ids must be strictly increasing");
            }
            previous_point_id = point_id;
            data_->point_ids.push_back(point_id);
            data_->point_positions.emplace(
                point_id,
                static_cast<size_t>(point_position)
            );

            long long start = point_offsets(point_position);
            long long end = point_offsets(point_position + 1);
            if (start != expected_segment_start || end < start || end > input_segment_count_) {
                throw std::invalid_argument(
                    "point_offsets must be contiguous, sorted, and in segment bounds"
                );
            }
            expected_segment_start = end;

            std::vector<EdgeInterval> point_intervals;
            point_intervals.reserve(static_cast<size_t>(end - start));
            for (long long row = start; row < end; row++) {
                int building_id = segment_building_ids(row);
                int edge_index = segment_edge_indices(row);
                validate_segment_metadata(
                    building_id,
                    edge_index,
                    segment_edge_lengths(row)
                );

                std::uint64_t edge_key = make_edge_key(building_id, edge_index);
                store_edge_length(edge_key, segment_edge_lengths(row));

                double interval_start = 0.0;
                double interval_end = 0.0;
                if (!normalize_interval(
                        segment_ranges(row, 0),
                        segment_ranges(row, 1),
                        interval_start,
                        interval_end
                    )) {
                    continue;
                }
                point_intervals.push_back({edge_key, interval_start, interval_end});
            }

            std::sort(
                point_intervals.begin(),
                point_intervals.end(),
                [](const EdgeInterval& left, const EdgeInterval& right) {
                    if (left.edge_key != right.edge_key) {
                        return left.edge_key < right.edge_key;
                    }
                    if (left.start != right.start) {
                        return left.start < right.start;
                    }
                    return left.end < right.end;
                }
            );

            for (const EdgeInterval& interval : point_intervals) {
                if (!data_->point_intervals.empty()) {
                    EdgeInterval& previous = data_->point_intervals.back();
                    bool same_point =
                        static_cast<long long>(data_->point_intervals.size()) >
                        data_->point_interval_offsets.back();
                    if (same_point &&
                        previous.edge_key == interval.edge_key &&
                        interval.start <= previous.end + COVERAGE_INTERVAL_MERGE_EPS) {
                        previous.end = std::max(previous.end, interval.end);
                        continue;
                    }
                }
                data_->point_intervals.push_back(interval);
            }
            data_->point_interval_offsets.push_back(
                static_cast<long long>(data_->point_intervals.size())
            );
        }

        if (point_offsets(0) != 0 || point_offsets(point_ids.shape(0)) != input_segment_count_) {
            throw std::invalid_argument(
                "point_offsets must start at zero and end at the segment count"
            );
        }
    }

    int hard_score(long long point_id) const {
        return score_from_delta(calculate_point_delta(point_id)).hard_score;
    }

    double soft_score(long long point_id) const {
        return score_from_delta(calculate_point_delta(point_id)).soft_score;
    }

    py::dict evaluate_point(long long point_id) const {
        return score_to_dict(score_from_delta(calculate_point_delta(point_id)));
    }

    py::array_t<int> evaluate_all_hard(LongArray candidate_point_ids_arr) const {
        auto candidate_point_ids = candidate_point_ids_arr.unchecked<1>();
        std::vector<int> scores;
        scores.reserve(static_cast<size_t>(candidate_point_ids.shape(0)));
        for (py::ssize_t index = 0; index < candidate_point_ids.shape(0); index++) {
            scores.push_back(hard_score(candidate_point_ids(index)));
        }
        return vector_to_numpy<int>(scores);
    }

    py::array_t<double> evaluate_all_soft(LongArray candidate_point_ids_arr) const {
        auto candidate_point_ids = candidate_point_ids_arr.unchecked<1>();
        std::vector<double> scores;
        scores.reserve(static_cast<size_t>(candidate_point_ids.shape(0)));
        for (py::ssize_t index = 0; index < candidate_point_ids.shape(0); index++) {
            scores.push_back(soft_score(candidate_point_ids(index)));
        }
        return vector_to_numpy<double>(scores);
    }

    py::dict evaluate_all(LongArray candidate_point_ids_arr) const {
        auto candidate_point_ids = candidate_point_ids_arr.unchecked<1>();
        std::vector<long long> point_ids;
        std::vector<int> hard_scores;
        std::vector<double> soft_scores;
        std::vector<double> unique_lengths;
        size_t count = static_cast<size_t>(candidate_point_ids.shape(0));
        point_ids.reserve(count);
        hard_scores.reserve(count);
        soft_scores.reserve(count);
        unique_lengths.reserve(count);

        for (py::ssize_t index = 0; index < candidate_point_ids.shape(0); index++) {
            long long point_id = candidate_point_ids(index);
            IncrementalPointScore score = score_from_delta(calculate_point_delta(point_id));
            point_ids.push_back(point_id);
            hard_scores.push_back(score.hard_score);
            soft_scores.push_back(score.soft_score);
            unique_lengths.push_back(score.unique_length);
        }

        py::dict out;
        out["point_indices"] = vector_to_numpy<long long>(point_ids);
        out["hard_scores"] = vector_to_numpy<int>(hard_scores);
        out["soft_scores"] = vector_to_numpy<double>(soft_scores);
        out["unique_lengths"] = vector_to_numpy<double>(unique_lengths);
        return out;
    }

    void add_point(long long point_id) {
        if (selected_point_ids_.find(point_id) != selected_point_ids_.end()) {
            throw std::invalid_argument("point_id has already been selected");
        }

        PointCoverageDelta delta = calculate_point_delta(point_id);
        auto [start, end] = point_interval_range(point_id);
        long long interval_index = start;
        while (interval_index < end) {
            std::uint64_t edge_key =
                data_->point_intervals[static_cast<size_t>(interval_index)].edge_key;
            long long edge_end = interval_index + 1;
            while (edge_end < end &&
                   data_->point_intervals[static_cast<size_t>(edge_end)].edge_key ==
                       edge_key) {
                edge_end++;
            }

            std::vector<std::pair<double, double>> candidate_intervals;
            candidate_intervals.reserve(static_cast<size_t>(edge_end - interval_index));
            for (long long index = interval_index; index < edge_end; index++) {
                const EdgeInterval& interval =
                    data_->point_intervals[static_cast<size_t>(index)];
                candidate_intervals.push_back({interval.start, interval.end});
            }

            auto& current_intervals = current_intervals_by_edge_[edge_key];
            current_intervals = merge_interval_sets(current_intervals, candidate_intervals);
            interval_index = edge_end;
        }

        for (const BuildingLengthDelta& building_delta : delta.buildings) {
            size_t building_id = static_cast<size_t>(building_delta.building_id);
            double perimeter = data_->building_perimeters[building_id];
            double old_coverage = coverage_for_building(building_delta.building_id);
            building_visible_lengths_[building_id] = std::min(
                perimeter,
                building_visible_lengths_[building_id] + building_delta.added_length
            );
            double new_coverage = coverage_for_building(building_delta.building_id);
            if (old_coverage < tau_ && new_coverage >= tau_) {
                current_score_++;
            }
        }

        selected_point_ids_.insert(point_id);
        selected_point_order_.push_back(point_id);
    }

    void add_points(LongArray point_ids_arr) {
        auto point_ids = point_ids_arr.unchecked<1>();
        for (py::ssize_t index = 0; index < point_ids.shape(0); index++) {
            add_point(point_ids(index));
        }
    }

    void remove_point(long long point_id) {
        auto selected_iterator = selected_point_ids_.find(point_id);
        if (selected_iterator == selected_point_ids_.end()) {
            throw std::invalid_argument("point_id is not currently selected");
        }

        auto [removed_start, removed_end] = point_interval_range(point_id);
        std::unordered_set<std::uint64_t> affected_edges;
        std::unordered_set<int> affected_buildings;
        for (long long index = removed_start; index < removed_end; index++) {
            std::uint64_t edge_key =
                data_->point_intervals[static_cast<size_t>(index)].edge_key;
            affected_edges.insert(edge_key);
            affected_buildings.insert(building_id_from_edge_key(edge_key));
        }

        std::unordered_map<
            std::uint64_t,
            std::vector<std::pair<double, double>>
        > remaining_intervals_by_edge;
        for (long long selected_point_id : selected_point_order_) {
            if (selected_point_id == point_id) {
                continue;
            }
            auto [start, end] = point_interval_range(selected_point_id);
            for (long long index = start; index < end; index++) {
                const EdgeInterval& interval =
                    data_->point_intervals[static_cast<size_t>(index)];
                if (affected_edges.find(interval.edge_key) == affected_edges.end()) {
                    continue;
                }
                remaining_intervals_by_edge[interval.edge_key].push_back(
                    {interval.start, interval.end}
                );
            }
        }

        std::unordered_map<int, bool> building_was_scored;
        for (int building_id : affected_buildings) {
            building_was_scored.emplace(
                building_id,
                coverage_for_building(building_id) >= tau_
            );
        }

        std::unordered_map<int, double> building_length_deltas;
        static const std::vector<std::pair<double, double>> empty_intervals;
        for (std::uint64_t edge_key : affected_edges) {
            auto current_iterator = current_intervals_by_edge_.find(edge_key);
            const auto& current_intervals =
                current_iterator == current_intervals_by_edge_.end()
                ? empty_intervals
                : current_iterator->second;
            double old_fraction = interval_fraction(current_intervals);

            auto remaining_iterator = remaining_intervals_by_edge.find(edge_key);
            std::vector<std::pair<double, double>> rebuilt_intervals;
            if (remaining_iterator != remaining_intervals_by_edge.end()) {
                auto& intervals = remaining_iterator->second;
                std::sort(intervals.begin(), intervals.end());
                rebuilt_intervals = merge_interval_sets(empty_intervals, intervals);
            }
            double new_fraction = interval_fraction(rebuilt_intervals);

            int building_id = building_id_from_edge_key(edge_key);
            building_length_deltas[building_id] +=
                data_->edge_lengths.at(edge_key) * (new_fraction - old_fraction);

            if (rebuilt_intervals.empty()) {
                current_intervals_by_edge_.erase(edge_key);
            } else {
                current_intervals_by_edge_[edge_key] = std::move(rebuilt_intervals);
            }
        }

        for (const auto& entry : building_length_deltas) {
            int building_id = entry.first;
            size_t building_index = static_cast<size_t>(building_id);
            building_visible_lengths_[building_index] = std::clamp(
                building_visible_lengths_[building_index] + entry.second,
                0.0,
                data_->building_perimeters[building_index]
            );

            bool is_scored = coverage_for_building(building_id) >= tau_;
            bool was_scored = building_was_scored.at(building_id);
            if (was_scored && !is_scored) {
                current_score_--;
            } else if (!was_scored && is_scored) {
                current_score_++;
            }
        }

        selected_point_ids_.erase(selected_iterator);
        selected_point_order_.erase(
            std::remove(
                selected_point_order_.begin(),
                selected_point_order_.end(),
                point_id
            ),
            selected_point_order_.end()
        );
    }

    IncrementalCoverageScorer clone() const {
        return *this;
    }

    void reset() {
        current_intervals_by_edge_.clear();
        std::fill(
            building_visible_lengths_.begin(),
            building_visible_lengths_.end(),
            0.0
        );
        selected_point_ids_.clear();
        selected_point_order_.clear();
        current_score_ = 0;
    }

    double building_coverage(int building_id) const {
        return coverage_for_building(building_id);
    }

    py::array_t<long long> point_ids() const {
        return vector_to_numpy<long long>(data_->point_ids);
    }

    py::array_t<long long> selected_point_ids() const {
        return vector_to_numpy<long long>(selected_point_order_);
    }

    py::dict current_visible_segments() const {
        std::vector<std::uint64_t> edge_keys;
        edge_keys.reserve(current_intervals_by_edge_.size());
        for (const auto& entry : current_intervals_by_edge_) {
            edge_keys.push_back(entry.first);
        }
        std::sort(edge_keys.begin(), edge_keys.end());

        size_t segment_count = 0;
        for (std::uint64_t edge_key : edge_keys) {
            segment_count += current_intervals_by_edge_.at(edge_key).size();
        }

        std::vector<int> building_ids;
        std::vector<int> edge_indices;
        std::vector<double> edge_lengths;
        std::vector<double> segment_ranges;
        building_ids.reserve(segment_count);
        edge_indices.reserve(segment_count);
        edge_lengths.reserve(segment_count);
        segment_ranges.reserve(segment_count * 2);

        for (std::uint64_t edge_key : edge_keys) {
            int building_id = building_id_from_edge_key(edge_key);
            int edge_index = static_cast<int>(static_cast<std::uint32_t>(edge_key));
            double edge_length = data_->edge_lengths.at(edge_key);
            for (const auto& interval : current_intervals_by_edge_.at(edge_key)) {
                building_ids.push_back(building_id);
                edge_indices.push_back(edge_index);
                edge_lengths.push_back(edge_length);
                segment_ranges.push_back(interval.first);
                segment_ranges.push_back(interval.second);
            }
        }

        py::array_t<double> ranges_array({
            static_cast<py::ssize_t>(segment_count),
            static_cast<py::ssize_t>(2)
        });
        auto ranges = ranges_array.mutable_unchecked<2>();
        for (py::ssize_t row = 0; row < static_cast<py::ssize_t>(segment_count); row++) {
            ranges(row, 0) = segment_ranges[static_cast<size_t>(row) * 2];
            ranges(row, 1) = segment_ranges[static_cast<size_t>(row) * 2 + 1];
        }

        py::dict out;
        out["building_ids"] = vector_to_numpy<int>(building_ids);
        out["edge_indices"] = vector_to_numpy<int>(edge_indices);
        out["segment_ranges"] = std::move(ranges_array);
        out["edge_lengths"] = vector_to_numpy<double>(edge_lengths);
        return out;
    }

    int current_score() const {
        return current_score_;
    }

    long long selected_count() const {
        return static_cast<long long>(selected_point_ids_.size());
    }

    long long num_points() const {
        return static_cast<long long>(data_->point_ids.size());
    }

    long long input_segment_count() const {
        return input_segment_count_;
    }

    long long stored_interval_count() const {
        return static_cast<long long>(data_->point_intervals.size());
    }

    double tau() const {
        return tau_;
    }

private:
    // Candidate visibility data is immutable after construction and shared by clones.
    std::shared_ptr<CoverageScorerData> data_;
    double tau_ = 0.0;
    std::vector<double> building_visible_lengths_;
    std::unordered_map<
        std::uint64_t,
        std::vector<std::pair<double, double>>
    > current_intervals_by_edge_;
    std::unordered_set<long long> selected_point_ids_;
    std::vector<long long> selected_point_order_;
    int current_score_ = 0;
    long long input_segment_count_ = 0;

    static void validate_tau(double tau) {
        if (!std::isfinite(tau) || tau <= 0.0 || tau > 1.0) {
            throw std::invalid_argument("tau must be finite and in the interval (0, 1]");
        }
    }

    static void validate_input_shapes(
        const LongArray& point_ids_arr,
        const LongArray& point_offsets_arr,
        const IntArray& segment_building_ids_arr,
        const IntArray& segment_edge_indices_arr,
        const DoubleArray& segment_ranges_arr,
        const DoubleArray& segment_edge_lengths_arr,
        const DoubleArray& building_perimeters_arr
    ) {
        if (point_ids_arr.ndim() != 1) {
            throw std::invalid_argument("point_ids must be a 1D array");
        }
        if (point_offsets_arr.ndim() != 1 ||
            point_offsets_arr.shape(0) != point_ids_arr.shape(0) + 1) {
            throw std::invalid_argument("point_offsets must have length len(point_ids) + 1");
        }
        if (segment_building_ids_arr.ndim() != 1 ||
            segment_edge_indices_arr.ndim() != 1 ||
            segment_edge_lengths_arr.ndim() != 1) {
            throw std::invalid_argument("segment metadata arrays must be 1D");
        }
        if (segment_ranges_arr.ndim() != 2 || segment_ranges_arr.shape(1) != 2) {
            throw std::invalid_argument("segment_ranges must have shape (n, 2)");
        }
        py::ssize_t segment_count = segment_building_ids_arr.shape(0);
        if (segment_edge_indices_arr.shape(0) != segment_count ||
            segment_edge_lengths_arr.shape(0) != segment_count ||
            segment_ranges_arr.shape(0) != segment_count) {
            throw std::invalid_argument("all segment arrays must have the same row count");
        }
        if (building_perimeters_arr.ndim() != 1) {
            throw std::invalid_argument("building_perimeters must be a 1D array");
        }
    }

    void validate_segment_metadata(
        int building_id,
        int edge_index,
        double edge_length
    ) const {
        if (building_id < 0 ||
            building_id >= static_cast<int>(data_->building_perimeters.size())) {
            throw std::invalid_argument("segment contains an invalid building_id");
        }
        double perimeter =
            data_->building_perimeters[static_cast<size_t>(building_id)];
        if (!std::isfinite(perimeter) || perimeter <= 0.0) {
            throw std::invalid_argument("segment building has an invalid perimeter");
        }
        if (edge_index < 0) {
            throw std::invalid_argument("segment contains a negative edge_index");
        }
        if (!std::isfinite(edge_length) || edge_length < 0.0) {
            throw std::invalid_argument("segment contains an invalid edge_length");
        }
    }

    static std::uint64_t make_edge_key(int building_id, int edge_index) {
        return
            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(building_id)) << 32) |
            static_cast<std::uint32_t>(edge_index);
    }

    static int building_id_from_edge_key(std::uint64_t edge_key) {
        return static_cast<int>(edge_key >> 32);
    }

    void store_edge_length(std::uint64_t edge_key, double edge_length) {
        auto [iterator, inserted] =
            data_->edge_lengths.emplace(edge_key, edge_length);
        if (!inserted) {
            double existing = iterator->second;
            double tolerance = 1e-10 * std::max({1.0, std::abs(existing), std::abs(edge_length)});
            if (std::abs(existing - edge_length) > tolerance) {
                throw std::invalid_argument(
                    "inconsistent edge_length values for the same building edge"
                );
            }
        }
    }

    static bool normalize_interval(
        double s1,
        double s2,
        double& interval_start,
        double& interval_end
    ) {
        if (!std::isfinite(s1) || !std::isfinite(s2)) {
            throw std::invalid_argument("segment contains a non-finite s1 or s2");
        }
        interval_start = std::clamp(std::min(s1, s2), 0.0, 1.0);
        interval_end = std::clamp(std::max(s1, s2), 0.0, 1.0);
        if (interval_start <= COVERAGE_ENDPOINT_SNAP_EPS) {
            interval_start = 0.0;
        }
        if (interval_end >= 1.0 - COVERAGE_ENDPOINT_SNAP_EPS) {
            interval_end = 1.0;
        }
        return interval_end > interval_start + COVERAGE_INTERVAL_MERGE_EPS;
    }

    std::pair<long long, long long> point_interval_range(long long point_id) const {
        auto iterator = data_->point_positions.find(point_id);
        if (iterator == data_->point_positions.end()) {
            throw std::invalid_argument("point_id is not present in the segment data");
        }
        size_t point_position = iterator->second;
        return {
            data_->point_interval_offsets[point_position],
            data_->point_interval_offsets[point_position + 1]
        };
    }

    static double additional_fraction(
        const std::vector<EdgeInterval>& candidate,
        long long candidate_start,
        long long candidate_end,
        const std::vector<std::pair<double, double>>& current
    ) {
        double candidate_length = 0.0;
        double intersection_length = 0.0;
        size_t current_index = 0;

        for (long long index = candidate_start; index < candidate_end; index++) {
            const EdgeInterval& candidate_interval = candidate[static_cast<size_t>(index)];
            candidate_length += candidate_interval.end - candidate_interval.start;
            while (current_index < current.size() &&
                   current[current_index].second <= candidate_interval.start) {
                current_index++;
            }

            size_t overlap_index = current_index;
            while (overlap_index < current.size() &&
                   current[overlap_index].first < candidate_interval.end) {
                double overlap_start = std::max(
                    candidate_interval.start,
                    current[overlap_index].first
                );
                double overlap_end = std::min(
                    candidate_interval.end,
                    current[overlap_index].second
                );
                if (overlap_end > overlap_start) {
                    intersection_length += overlap_end - overlap_start;
                }
                if (current[overlap_index].second >= candidate_interval.end) {
                    break;
                }
                overlap_index++;
            }
        }

        return std::clamp(candidate_length - intersection_length, 0.0, 1.0);
    }

    static std::vector<std::pair<double, double>> merge_interval_sets(
        const std::vector<std::pair<double, double>>& left,
        const std::vector<std::pair<double, double>>& right
    ) {
        std::vector<std::pair<double, double>> merged;
        merged.reserve(left.size() + right.size());
        size_t left_index = 0;
        size_t right_index = 0;

        auto append = [&merged](const std::pair<double, double>& interval) {
            if (!merged.empty() &&
                interval.first <= merged.back().second + COVERAGE_INTERVAL_MERGE_EPS) {
                merged.back().second = std::max(merged.back().second, interval.second);
                return;
            }
            merged.push_back(interval);
        };

        while (left_index < left.size() || right_index < right.size()) {
            if (right_index >= right.size() ||
                (left_index < left.size() && left[left_index] < right[right_index])) {
                append(left[left_index]);
                left_index++;
            } else {
                append(right[right_index]);
                right_index++;
            }
        }
        return merged;
    }

    static double interval_fraction(
        const std::vector<std::pair<double, double>>& intervals
    ) {
        double fraction = 0.0;
        for (const auto& interval : intervals) {
            fraction += interval.second - interval.first;
        }
        return std::clamp(fraction, 0.0, 1.0);
    }

    PointCoverageDelta calculate_point_delta(long long point_id) const {
        auto [start, end] = point_interval_range(point_id);
        PointCoverageDelta delta;
        int active_building_id = -1;
        double active_building_length = 0.0;
        long long interval_index = start;

        while (interval_index < end) {
            std::uint64_t edge_key =
                data_->point_intervals[static_cast<size_t>(interval_index)].edge_key;
            int building_id = building_id_from_edge_key(edge_key);
            long long edge_end = interval_index + 1;
            while (edge_end < end &&
                   data_->point_intervals[static_cast<size_t>(edge_end)].edge_key ==
                       edge_key) {
                edge_end++;
            }

            static const std::vector<std::pair<double, double>> empty_intervals;
            auto current_iterator = current_intervals_by_edge_.find(edge_key);
            const auto& current_intervals =
                current_iterator == current_intervals_by_edge_.end()
                ? empty_intervals
                : current_iterator->second;

            double added_fraction = additional_fraction(
                data_->point_intervals,
                interval_index,
                edge_end,
                current_intervals
            );
            double added_length = data_->edge_lengths.at(edge_key) * added_fraction;

            if (building_id != active_building_id) {
                if (active_building_id >= 0 && active_building_length > 0.0) {
                    delta.buildings.push_back({active_building_id, active_building_length});
                }
                active_building_id = building_id;
                active_building_length = 0.0;
            }
            active_building_length += added_length;
            delta.unique_length += added_length;
            interval_index = edge_end;
        }

        if (active_building_id >= 0 && active_building_length > 0.0) {
            delta.buildings.push_back({active_building_id, active_building_length});
        }
        return delta;
    }

    IncrementalPointScore score_from_delta(const PointCoverageDelta& delta) const {
        IncrementalPointScore score;
        score.unique_length = delta.unique_length;

        for (const BuildingLengthDelta& building_delta : delta.buildings) {
            size_t building_id = static_cast<size_t>(building_delta.building_id);
            double perimeter = data_->building_perimeters[building_id];
            double old_coverage = coverage_for_building(building_delta.building_id);
            double new_coverage = std::clamp(
                (building_visible_lengths_[building_id] + building_delta.added_length) /
                    perimeter,
                0.0,
                1.0
            );

            if (old_coverage < tau_ && new_coverage >= tau_) {
                score.hard_score++;
            }
            score.soft_score +=
                std::min(new_coverage, tau_) -
                std::min(old_coverage, tau_);
        }

        score.soft_score = std::max(0.0, score.soft_score);
        return score;
    }

    double coverage_for_building(int building_id) const {
        if (building_id < 0 ||
            building_id >= static_cast<int>(data_->building_perimeters.size())) {
            throw std::invalid_argument("building_id is out of range");
        }
        double perimeter =
            data_->building_perimeters[static_cast<size_t>(building_id)];
        if (!std::isfinite(perimeter) || perimeter <= 0.0) {
            throw std::invalid_argument("building_id has no valid perimeter");
        }
        return std::clamp(
            building_visible_lengths_[static_cast<size_t>(building_id)] / perimeter,
            0.0,
            1.0
        );
    }

    static py::dict score_to_dict(const IncrementalPointScore& score) {
        py::dict out;
        out["hard_score"] = score.hard_score;
        out["soft_score"] = score.soft_score;
        out["unique_length"] = score.unique_length;
        return out;
    }
};

PYBIND11_MODULE(coverage_scorer_cpp, m) {
    m.doc() = "Union building-boundary coverage scorer for selected visibility segments";
    m.def(
        "score_union_coverage",
        &score_union_coverage,
        py::arg("polygon_points"),
        py::arg("polygon_offsets"),
        py::arg("segment_building_ids"),
        py::arg("segment_edge_indices"),
        py::arg("segment_ranges"),
        py::arg("tau"),
        "Return the number of buildings whose union boundary coverage meets tau."
    );

    py::class_<IncrementalCoverageScorer>(m, "IncrementalCoverageScorer")
        .def(
            py::init<
                LongArray,
                LongArray,
                IntArray,
                IntArray,
                DoubleArray,
                DoubleArray,
                DoubleArray,
                double
            >(),
            py::arg("point_ids"),
            py::arg("point_offsets"),
            py::arg("segment_building_ids"),
            py::arg("segment_edge_indices"),
            py::arg("segment_ranges"),
            py::arg("segment_edge_lengths"),
            py::arg("building_perimeters"),
            py::arg("tau")
        )
        .def("hard_score", &IncrementalCoverageScorer::hard_score, py::arg("point_id"))
        .def("soft_score", &IncrementalCoverageScorer::soft_score, py::arg("point_id"))
        .def(
            "evaluate_point",
            &IncrementalCoverageScorer::evaluate_point,
            py::arg("point_id")
        )
        .def(
            "evaluate_all_hard",
            &IncrementalCoverageScorer::evaluate_all_hard,
            py::arg("candidate_point_ids")
        )
        .def(
            "evaluate_all_soft",
            &IncrementalCoverageScorer::evaluate_all_soft,
            py::arg("candidate_point_ids")
        )
        .def(
            "evaluate_all",
            &IncrementalCoverageScorer::evaluate_all,
            py::arg("candidate_point_ids")
        )
        .def("add_point", &IncrementalCoverageScorer::add_point, py::arg("point_id"))
        .def("add_points", &IncrementalCoverageScorer::add_points, py::arg("point_ids"))
        .def(
            "remove_point",
            &IncrementalCoverageScorer::remove_point,
            py::arg("point_id")
        )
        .def("clone", &IncrementalCoverageScorer::clone)
        .def("reset", &IncrementalCoverageScorer::reset)
        .def(
            "building_coverage",
            &IncrementalCoverageScorer::building_coverage,
            py::arg("building_id")
        )
        .def(
            "current_visible_segments",
            &IncrementalCoverageScorer::current_visible_segments
        )
        .def_property_readonly("point_ids", &IncrementalCoverageScorer::point_ids)
        .def_property_readonly(
            "selected_point_ids",
            &IncrementalCoverageScorer::selected_point_ids
        )
        .def_property_readonly("current_score", &IncrementalCoverageScorer::current_score)
        .def_property_readonly("selected_count", &IncrementalCoverageScorer::selected_count)
        .def_property_readonly("num_points", &IncrementalCoverageScorer::num_points)
        .def_property_readonly(
            "input_segment_count",
            &IncrementalCoverageScorer::input_segment_count
        )
        .def_property_readonly(
            "stored_interval_count",
            &IncrementalCoverageScorer::stored_interval_count
        )
        .def_property_readonly("tau", &IncrementalCoverageScorer::tau);
}
