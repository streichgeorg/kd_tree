#include <benchmark/benchmark.h>

#include "Eigen/Core"
#include "nabo/nabo.h"

#include "kd_tree.hpp"
#include "util.hpp"

template<uint8_t DIMS, typename QUEUE = HeapQueue>
static void BM_query(
    benchmark::State& state,
    const KDTree<DIMS> &tree
) {
    int k = state.range(0);

    for (auto _ : state) {
        auto query = generate_uniform_points<DIMS>(1, tree.bb_min, tree.bb_max)[0];
        auto result = tree.template knn<QUEUE>(query, k);
    }
}

template<uint8_t DIMS>
Eigen::Matrix<float, DIMS, -1> to_eigen(const std::vector<Point<DIMS>> &points) {
    Eigen::Matrix<float, DIMS, -1> M = Eigen::Matrix<float, DIMS, -1>::Random(DIMS, points.size());
    for (int i = 0; i < points.size(); i++) {
        for (int j = 0; j < DIMS; j++) {
            M(j, i) = points[i][j];
        }
    }
    return M;
}

template<uint8_t DIMS>
static void BM_libnabo_query(
    benchmark::State& state,
    Nabo::NNSearchF *tree,
    const Point<DIMS> &bb_min,
    const Point<DIMS> &bb_max
) {
    int k = state.range(0);

    for (auto _ : state) {
        Eigen::MatrixXi indices;
        indices.resize(k, 1);
        Eigen::MatrixXf dists;
        dists.resize(k, 1);

        auto query = generate_uniform_points<DIMS>(1, bb_min, bb_max);
        tree->knn(to_eigen<DIMS>(query), indices, dists, k, 0, Nabo::NNSearchF::ALLOW_SELF_MATCH);
    }
}

int main(int argc, char** argv) {
    auto eth_points = load_eth_pointcloud();
    KDTree<3> eth_tree(eth_points);

    Eigen::MatrixXf M = to_eigen<3>(eth_points);
    Nabo::NNSearchF *eth_libnabo = Nabo::NNSearchF::createKDTreeLinearHeap(M);

    // benchmark::RegisterBenchmark(
    //     "eth_point_cloud_query",
    //     BM_query<3>,
    //     eth_tree
    // )->RangeMultiplier(2)->Range(1, 128);

    benchmark::RegisterBenchmark(
        "eth_point_cloud_query_arr_queue",
        BM_query<3, ArrQueue>,
        eth_tree
    )->RangeMultiplier(2)->Range(1, 16);

    benchmark::RegisterBenchmark(
        "eth_point_cloud_libnabo_query",
        BM_libnabo_query<3>,
        eth_libnabo,
        eth_tree.bb_min,
        eth_tree.bb_max
    )->RangeMultiplier(2)->Range(1, 16);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();

    delete eth_libnabo;

}
