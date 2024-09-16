#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <optional>
#include <queue>
#include <tuple>
#include <vector>

class HeapQueue {
public:
    const static int max_size = std::numeric_limits<int>::max();

private:
    std::priority_queue<std::pair<float, int>> queue;

public:
    HeapQueue(int size) {
        for (int i = 0; i < size; i++) {
            queue.push({std::numeric_limits<float>::infinity(), 0});
        }
    }

    float pop_push(float dist, int id) {
        queue.pop();
        queue.push({dist, id});
        return queue.top().first;
    }

    float top() {
        return queue.top().first;
    }

    std::vector<std::pair<float, int>> elems() {
        std::priority_queue<std::pair<float, int>> copy = queue;

        std::vector<std::pair<float, int>> result;
        while (!copy.empty()) {
            auto el = copy.top();
            copy.pop();
            if (el.first < std::numeric_limits<float>::infinity()) result.push_back(el);
        }
        return result;
    }
};

class ArrQueue {
public:
    const static int max_size = 16;

private:
    int size;

    float top_value = std::numeric_limits<float>::infinity();
    int top_idx = 0;

public:
    std::array<std::pair<float, int>, max_size> values;

    ArrQueue(int _size) : size(_size) {
        values.fill({std::numeric_limits<float>::infinity(), 0});
    }

    float pop_push(float dist, int id) {
        values[top_idx] = {dist, id};
        top_value = dist;

        for (int i = 0; i < size; i++) {
            if (values[i].first > top_value) {
                top_value = values[i].first;
                top_idx = i;
            }
        }

        return top_value;
    }

    float top() {
        return top_value;
    }

    std::vector<std::pair<float, int>> elems() {
        std::vector<std::pair<float, int>> result;
        for (auto el : values) {
            if (el.first < std::numeric_limits<float>::infinity()) result.push_back(el);
        }
        return result;
    }
};

template<uint8_t DIMS>
using Point = std::array<float, DIMS>;

template <
    uint8_t DIMS,
    int BUCKET_SIZE = 64,
    uint8_t SUBTREE_SIZE = 3,
    uint8_t SUBTREE_BYTES = 32
>
class KDTree {
public:
    using Point = Point<DIMS>;
private:
    struct Bucket {
        float point_data[DIMS][BUCKET_SIZE];
        int point_ids[BUCKET_SIZE];
        int num_points;

        Bucket(int _num_points) : num_points(_num_points) {
            for (int i = 0; i < DIMS; i++) {
                for (int j = 0; j < BUCKET_SIZE; j++) {
                    point_data[i][j] = std::numeric_limits<float>::infinity();
                }
            }
        }
    };

    std::vector<Bucket> buckets;

    struct NodeRef {
        int node_idx;
        int inner_idx;
    };

    const static uint8_t direction_bits = 8 * (SUBTREE_BYTES - 8 * SUBTREE_SIZE - 1) / SUBTREE_SIZE;

    struct alignas(SUBTREE_BYTES) Node {
        enum Type : uint8_t {
            LEAF,
            INNER
        } __attribute__((packed));

        union {
            int bucket_idx;
            struct {
                float split[SUBTREE_SIZE];
                uint32_t child_nodes[SUBTREE_SIZE];
                uint64_t direction: SUBTREE_SIZE * direction_bits;
            } __attribute__((packed)) inner;
        } __attribute__((packed)) data;

        Type type;
    } __attribute__((packed));

    static_assert(sizeof(Node) <= SUBTREE_BYTES, "Nodes should fit into a single cache block.");
    static_assert(DIMS <= (1 << direction_bits), "Not enough bits to encode direction.");

    uint8_t get_direction(const Node &node, NodeRef ref) const {
        uint64_t mask = (1 << direction_bits) - 1;
        return (node.data.inner.direction >> (direction_bits * ref.inner_idx)) & mask;
    }

    void set_direction(Node &node, NodeRef ref, uint64_t value) {
        uint64_t mask = (1 << direction_bits) - 1;
        uint64_t offset = direction_bits * ref.inner_idx;
        node.data.inner.direction &= ~(mask << offset);
        node.data.inner.direction |= (value & mask) << offset;
    }

    std::vector<Node> nodes;   

    NodeRef get_child(NodeRef ref, bool right) const {
        int child_inner = 2 * ref.inner_idx + 1 + right;
        if (child_inner < SUBTREE_SIZE) {
            return NodeRef {ref.node_idx, child_inner};
        }

        int child_node;

        // Special case for left most child, the node was allocated right next to this one
        if (child_inner == SUBTREE_SIZE) {
            child_node = ref.node_idx + 1;
        } else {
            child_node = nodes[ref.node_idx].data.inner.child_nodes[child_inner - SUBTREE_SIZE - 1];
        }

        return NodeRef {child_node, 0};
    } 

    NodeRef create_at(NodeRef ref, bool right) {
        int child_inner = 2 * ref.inner_idx + 1 + right;
        if (child_inner < SUBTREE_SIZE) {
            return NodeRef {ref.node_idx, child_inner};
        }

        int child_node = nodes.size();
        nodes.emplace_back();

        if (child_inner == SUBTREE_SIZE) {
            assert(child_node == ref.node_idx + 1);
        } else {
            nodes[ref.node_idx].data.inner.child_nodes[child_inner - SUBTREE_SIZE - 1] = child_node;
        }

        return NodeRef {child_node, 0};
    }

    void construct_subtree(
        NodeRef at,
        typename std::vector<std::pair<Point, int>>::iterator lit,
        typename std::vector<std::pair<Point, int>>::iterator rit,
        Point &bb_min, Point &bb_max
    ) {
        int num_points = std::distance(lit, rit);

        if (num_points <= BUCKET_SIZE) {
            // Add sentinel child nodes until we point to the root of a subtree
            while (at.inner_idx != 0) {
                nodes[at.node_idx].data.inner.split[at.inner_idx] = std::numeric_limits<float>::infinity();
                set_direction(nodes[at.node_idx], at, 0);
                at = create_at(at, false);
            }

            nodes[at.node_idx].data.bucket_idx = buckets.size();
            nodes[at.node_idx].type = Node::Type::LEAF;

            buckets.emplace_back(num_points);
            Bucket &bucket = buckets.back();

            for (int i = 0; i < num_points; i++) {
                bucket.point_ids[i] = (lit + i)->second;
                for (int j = 0; j < DIMS; j++) {
                    bucket.point_data[j][i] = (lit + i)->first[j];
                }
            }

            return;
        }

        uint8_t direction;
        float max_width = -std::numeric_limits<float>::infinity();

        for (int i = 0; i < DIMS; i++) {
            float width = bb_max[i] - bb_min[i];
            assert(width >= 0);
            if (width > max_width) {
                max_width = width;
                direction = i;
            }
        }

        float split = (bb_min[direction] + bb_max[direction]) / 2;
        auto mit = std::partition(lit, rit, [&](const std::pair<Point, int> &p) {
            return p.first[direction] < split;
        });

        if (mit == lit) {
            split = std::numeric_limits<float>::infinity();
            for (auto it = mit; it != rit; ++it) split = std::min(split, it->first[direction]);
        } else if (mit == rit) {
            split = -std::numeric_limits<float>::infinity();
            for (auto it = lit; it != mit; ++it) split = std::max(split, it->first[direction]);
        }

        nodes[at.node_idx].type = Node::Type::INNER;
        nodes[at.node_idx].data.inner.split[at.inner_idx] = split;

        for (auto it = lit; it != mit; ++it) {
            assert(it->first[direction] <= split);
        }

        for (auto it = mit; it != rit; ++it) {
            assert(it->first[direction] >= split);
        }

        set_direction(nodes[at.node_idx], at, direction);

        float old_bb_max = bb_max[direction];
        bb_max[direction] = split;
        construct_subtree(create_at(at, false), lit, mit, bb_min, bb_max);
        bb_max[direction] = old_bb_max;

        float old_bb_min = bb_min[direction];
        bb_min[direction] = split;
        construct_subtree(create_at(at, true), mit, rit, bb_min, bb_max);
        bb_min[direction] = old_bb_min;
    }

    void construct_from(std::vector<std::pair<Point, int>> &points) {
        nodes.emplace_back();
        NodeRef root {0, 0};

        bb_min = Point {std::numeric_limits<float>::infinity()};
        bb_max  = Point {-std::numeric_limits<float>::infinity()};

        for (auto p : points) {
            for (int i = 0; i < DIMS; i++) {
                bb_min[i] = std::min(bb_min[i], p.first[i]);
                bb_max[i] = std::max(bb_max[i], p.first[i]);
            }
        }

        construct_subtree(root, points.begin(), points.end(), bb_min, bb_max);
    }

    template<typename QUEUE, int BLOCK_SIZE = 16>
    void find_in_bucket(
        const Bucket &bucket,
        const Point &p,
        QUEUE &queue
    ) const {
        float req_dist = queue.top();

        static_assert(BUCKET_SIZE % BLOCK_SIZE == 0, "BLOCK_SIZE must divide BUCKET_SIZE");

        int num_points = bucket.num_points;

        // This might go slightly out of bounds, but we filled point data with
        // infty so it is still correct
        for (int i = 0; i < num_points; i += BLOCK_SIZE) {
            float dist[BLOCK_SIZE] = {0};

            for (int j = 0; j < DIMS; j++) {
                for (int k = 0; k < BLOCK_SIZE; k++) {
                    float d = p[j] - bucket.point_data[j][i + k];
                    dist[k] += d * d;
                }
            }

            for (int k = 0; k < BLOCK_SIZE; k++) {
                if (dist[k] <= req_dist) {
                    req_dist = queue.pop_push(dist[k], bucket.point_ids[i + k]);
                }
            }
        }
    }

    template<typename QUEUE>
    void find_in_node(
        NodeRef ref,
        const Point &p,
        QUEUE &queue,
        float s,
        Point h
    ) const {
        const Node &node = nodes[ref.node_idx];

        if (node.type == Node::Type::LEAF) {
            find_in_bucket(buckets[node.data.bucket_idx], p, queue);
            return;
        }

        float split = node.data.inner.split[ref.inner_idx];
        uint8_t direction = get_direction(node, ref);

        bool right = p[direction] > split;
        find_in_node(get_child(ref, right), p, queue, s, h);

        float d = p[direction] - split;
        s += d * d - h[direction];
        h[direction] = d * d;

        if (queue.top() > s) {
            find_in_node(get_child(ref, !right), p, queue, s, h);
        }
    }

    std::vector<std::pair<uint8_t, float>> wrapped_dims;
public:
    Point bb_min;
    Point bb_max;

    KDTree(std::vector<std::pair<Point, int>> &points_with_ids) {
        construct_from(points_with_ids);
    }

    KDTree(const std::vector<Point> &points) {
        std::vector<std::pair<Point, int>> points_with_ids;
        for (int i = 0; i < points.size(); i++) points_with_ids.emplace_back(points[i], i);
        construct_from(points_with_ids);
    }

    void make_wrapped(uint8_t dim, float dim_width) {
        wrapped_dims.push_back({dim, dim_width});
    }

    template<typename QUEUE = HeapQueue>
    QUEUE knn(
        const Point &point,
        int k = 1,
        std::optional<std::reference_wrapper<QUEUE>> maybe_queue = std::nullopt
    ) const {
        assert(k <= QUEUE::max_size);

        QUEUE local_queue(k);
        if (!maybe_queue) {
            maybe_queue = local_queue;
        }

        QUEUE &queue = maybe_queue.value();

        NodeRef root {0, 0};

        Point h;
        h.fill(0);

        int m = wrapped_dims.size();
        for (int i = 0; i < (1 << m); i++) {
            Point p = point;
            for (int j = 0; j < m; j++) {
                if (i & (1 << j)) {
                    int dim_idx;
                    float dim_width;
                    std::tie(dim_idx, dim_width) = wrapped_dims[j];
                    p[dim_idx] -= dim_width;
                }
            }
            find_in_node(root, p, queue, 0, h);
        }

        return queue;
    }
};
