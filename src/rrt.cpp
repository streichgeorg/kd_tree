#include <random>
#include <iostream>

#include "kd_tree.hpp"

template<typename KDTree>
class PointSet {
    using PointWithId = std::pair<typename KDTree::Point, int>;

    const static int buffer_size = 1000;

    std::vector<PointWithId> buffer;
    std::vector<std::optional<std::pair<KDTree, std::vector<PointWithId>>>> trees;

    int counter;
public: 
    PointSet() : trees(22) {}

    void insert(int point_id, const typename KDTree::Point &point) {
        buffer.push_back({point, point_id});

        if (buffer.size() >= buffer_size) {
            int i = 0;
            for (; i < trees.size() && trees[i].has_value(); i++) {
                // Extend the buffer associated with the tree since it contains more points
                auto new_buffer = std::move(trees[i].value().second);
                new_buffer.insert(new_buffer.end(), buffer.begin(), buffer.end());
                buffer = std::move(new_buffer);

                trees[i] = std::nullopt;
            }

            KDTree tree(buffer);
            trees[i] = {tree, std::move(buffer)};
            buffer = std::vector<PointWithId>();
        }
    }

    template<typename QUEUE = ArrQueue>
    QUEUE knn(
        const typename KDTree::Point &point,
        int k = 1
    ) const {
        QUEUE queue(k);

        for (auto &tree : trees) {
            if (tree) std::ignore = tree.value().first.template knn<QUEUE>(point, k, queue);
        }

        for (auto p : buffer) {
            float dist = 0;
            for (int i = 0; i < point.size(); i++) {
                float d = point[i] - p.first[i];
                dist += d * d;
            }
            if (dist < queue.top()) {
                queue.pop_push(dist, p.second);
            }
        }

        return queue;
    }
};

int main(int argc, char **argv) {
    if (argc != 2) throw std::runtime_error("Please specify number of points");

    int n = std::stoi(argv[1]);

    using Point = Point<2>;

    std::vector<Point> points;
    std::vector<std::pair<int, int>> edges;
        
    PointSet<KDTree<2>> tree;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> uniform(0.0, 1.0);

    for (int i = 0; i < n; i++) {
        Point point;
        for (int j = 0; j < point.size(); j++) {
            point[j] = uniform(gen);
        }

        int point_id = points.size();
        points.push_back(point);

        ArrQueue q = tree.knn<ArrQueue>(point, 1);
        if (q.top() < std::numeric_limits<float>::max()) {
            int nearest_id = q.elems()[0].second;
            edges.push_back({point_id, nearest_id});
        }

        tree.insert(point_id, point);
    }

    std::cout << points.size() << std::endl;

    for (auto p : points) {
        std::cout << p[0] << " " << p[1] << std::endl;
    }

    for (auto e : edges) {
        std::cout << e.first << " " << e.second << std::endl;
    }
}
