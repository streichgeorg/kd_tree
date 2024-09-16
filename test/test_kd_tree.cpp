#include <vector>

#include "util.hpp"
#include "kd_tree.hpp"

#undef NDEBUG
#include <cassert>

template<uint8_t DIMS, typename QUEUE = HeapQueue>
static void test_points(
    const std::vector<Point<DIMS>> &points,
    int k,
    int num_queries = 1000
) {
    KDTree<DIMS> tree(points);

    auto queries = generate_uniform_points<DIMS>(num_queries, tree.bb_min, tree.bb_max);

    for (auto &query : queries) {
        HeapQueue queue(k);

        for (int i = 0; i < points.size(); i++) {
            float dist = 0;
            for (int j = 0; j < DIMS; j++) {
                float d = points[i][j] - query[j];
                dist += d * d;
            }
            if (dist < queue.top()) {
                std::ignore = queue.pop_push(dist, i);
            }
        }

        std::vector<std::pair<float, int>> correct = queue.elems();
        std::vector<std::pair<float, int>> mine = tree.template knn<QUEUE>(query, k).elems();

        assert(correct.size() == mine.size());

        std::sort(correct.begin(), correct.end());
        std::sort(mine.begin(), mine.end());

        for (int i = 0; i < correct.size(); i++) {
            float err = abs(correct[i].first - mine[i].first);
            assert(err < 1e-3);
            if (points.size() < 1000) assert(correct[i].second == mine[i].second);
        }
    }
}

template<uint8_t DIMS, typename QUEUE = HeapQueue>
static void test_uniform_points(int num_points, int k = 1, int num_queries = 1000) {
    Point<DIMS> zeros;
    zeros.fill(0.0);
    Point<DIMS> ones;
    ones.fill(1.0);

    auto points = generate_uniform_points<DIMS>(num_points, zeros, ones);
    test_points<DIMS, QUEUE>(points, k, num_queries);
}

template<typename QUEUE = HeapQueue>
static void test_eth_pointcloud(int k, int num_parts = 37, int num_queries = 1000) {
    auto points = load_eth_pointcloud(num_parts);
    test_points<3, QUEUE>(points, k, num_queries);
}


int main(int argc, char **argv) {
    test_uniform_points<2>(10);
    test_uniform_points<2>(512);
    test_uniform_points<3>(100000);
    test_uniform_points<4>(100000, 5);
    test_uniform_points<3, ArrQueue>(100000, 5);
    test_uniform_points<3, HeapQueue>(100000, 20);

    test_eth_pointcloud(1, 1);
    test_eth_pointcloud(5, 36);
    test_eth_pointcloud<ArrQueue>(5, 36);
}
