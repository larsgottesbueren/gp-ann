#include <iostream>
#include "defs.h"
#include "points_io.h"

void Test() {

    std::vector<double> local_work = {
            0.635611, 0.729202, 0.641303, 4.86097, 0.610147, 1.11201, 0.668587 ,0.672009,
            0.525562, 0.582003, 0.417016 ,1.00467 ,1.42631, 0.370525, 0.432954, 0.595387 ,0.536881,
            0.55986 ,0.696593 ,0.321054,
            0.607208 ,0.587357 ,0.529585 ,0.63173, 0.629399 ,1.14386, 1.78718, 0.871503 ,1.09877 ,2.82946 ,
            0.864496 ,0.793463, 0.785419, 0.523559, 0.737946, 0.764495 ,0.576986 ,0.649949 ,0.520211, 0.594644
    };

    auto lwr = local_work;
    std::vector<size_t> assigned_hosts(local_work.size(), 1);

    const size_t num_queries = 100000;

    size_t num_hosts = local_work.size();

    double prev_qps_per_res = 1000000;

    while (true) {
        double mini = *std::ranges::min_element(lwr);
        auto maxiter = std::ranges::max_element(lwr);
        double maxi = *maxiter;
        double QPS = num_queries / maxi;
        double QPS_per_res = QPS / num_hosts;
        size_t max_shard = std::distance(lwr.begin(), maxiter);
        double ratio = maxi / mini;

        int snd_max = -1;
        for (int i = 0; i < lwr.size(); ++i) {
            if (i != max_shard && (snd_max == - 1 || lwr[snd_max] < lwr[i])) {
                snd_max = i;
            }
        }

        double ratio2 = maxi / lwr[snd_max];

        std::cout << "nh = " << num_hosts << " QPS " << QPS << " QPS/res " << QPS_per_res << " ratio " << ratio << " ratio2 " << ratio2 << " max shard " << max_shard << std::endl;

        if (QPS_per_res > 2800) break;
        if (false && std::abs(QPS_per_res - prev_qps_per_res) < 1e-3 ) { break; }

        prev_qps_per_res = QPS_per_res;

        num_hosts++;
        assigned_hosts[max_shard]++;
        lwr[max_shard] = local_work[max_shard] / assigned_hosts[max_shard];
    }


}


int main(int argc, const char* argv[]) {
    Test();
    std::exit(0);
    if (argc != 4) {
        std::cerr << "Usage ./Convert input-points output-points size" << std::endl;
        std::abort();
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];
    std::string k_string = argv[3];
    size_t n = std::stoi(k_string);

    PointSet points = ReadPoints(input_file, n);
    WritePoints(points, output_file);
}
