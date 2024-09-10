#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "kd_tree.hpp"

static std::random_device rd;
static std::mt19937 gen(rd());

template<uint8_t DIMS>
std::vector<Point<DIMS>> generate_uniform_points(uint32_t num_points, Point<DIMS> bb_min, Point<DIMS> bb_max) {
    std::uniform_real_distribution<float> uniform(0.0, 1.0);

    std::vector<Point<DIMS>> points(num_points);

    for (int i = 0; i < num_points; i++) {
        for (int j = 0; j < DIMS; j++) {
            float alpha = uniform(gen);
            points[i][j] = alpha * bb_max[j] + (1 - alpha) * bb_min[j];
        }
    }

    return points;
}

template<typename QUEUE = HeapQueue>
std::vector<Point<3>> load_eth_pointcloud(int num_parts = 36) {
    std::vector<Point<3>> points;

    for (int i = 0; i < num_parts; i++) {
        std::string filename = "../eth_pointcloud/PointCloud" + std::to_string(i) + ".csv";
        std::ifstream file(filename);

        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            exit(1);
        }

        std::string line;
        std::getline(file, line);

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string token;
            std::vector<float> row;

            std::getline(ss, token, ',');

            points.emplace_back();

            for (int i = 0; i < 3; ++i) {
                std::getline(ss, token, ',');
                points.back()[i] = std::stof(token);
            }
        }

        file.close();
    }

    return points;
}
