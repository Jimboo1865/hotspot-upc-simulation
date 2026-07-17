#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <string>

const double HBARC = 0.1973269804; // Gev fm

struct Position {
    double x;
    double y;
};

struct SimulationConfig {
    int A = 208;
    double R_WS = 6.62; // fm
    double a_WS = 0.546; // fm
    double r_max = 15.0; // fm

    double B_p_GeV2 = 4.75;
    double B_hs_GeV2 = 0.8;

    double p0 = 0.015;
    double p1 = -0.58;
    double p2 = 300.0;

    double M_V = 3.09690; // J/psi
    double Q2 = 0.05;

    double sigma_hotspot_position_fm() const { return std::sqrt(B_p_GeV2 * HBARC * HBARC); }
    double sigma_hotspot_shape_fm() const { return std::sqrt(B_hs_GeV2 * HBARC * HBARC); }

    double calculate_bjorken_x(double W) const {
        return (Q2 + M_V * M_V) / (Q2 + W * W);
    }

    double calculate_mean_hotspots(double W) const {
        double x = calculate_bjorken_x(W);
        return p0 * std::pow(x, p1) * (1.0 + (p2 * std::sqrt(x)));
    }
};

class DiscreteHotSpotSimulator {
    private:
        SimulationConfig config;
        std::mt19937_64 rng;

        // Grid parameters
        double grid_max_fm;
        double dx;
        int grid_size;

        // Cached grid coordinates and smooth Woods-Saxon baseline
        std::vector<double> x_coords;
        std::vector<double> T_WS_hat;

        double woods_saxon_density(double r) {
            return 1.0 / (1.0 + std::exp((r - config.R_WS) / config.a_WS));
        }

        int sample_zero_truncated_poisson(double mean) {
            std::poisson_distribution<int> poisson(mean);
            int val = poisson(rng);
            while (val == 0) {
                val = poisson(rng);
            }
            return val;
        }

        void precompute_grid_and_baseline() {
            grid_size = static_cast<int>((2.0 * grid_max_fm) / dx) + 1;
            x_coords.resize(grid_size);
            for (int i = 0; i < grid_size; ++i) {
                x_coords[i] = -grid_max_fm + i * dx;
            }

            T_WS_hat.assign(grid_size * grid_size, 0.0);
            double sum_T_WS = 0.0;

            for (int r = 0; r < grid_size; ++r) {
                double y = x_coords[r];
                for (int c =0; c < grid_size; ++c) {
                    double x = x_coords[c];
                    double radius = std::sqrt(x*x + y*y);
                    double val = woods_saxon_density(radius);
                    T_WS_hat[r * grid_size + c] = val;
                    sum_T_WS += val;
                }
            }

            double integral_T_WS = sum_T_WS * (dx * dx);
            for (auto& val : T_WS_hat) {
                val /= (integral_T_WS + 1e-12);
            }
        }

    public:
        DiscreteHotSpotSimulator(const SimulationConfig& cfg, double spatial_step_fm = 0.1, double max_boundry_fm = 15.0, unsigned long long seed = 12345)
            : config(cfg), rng(seed), grid_max_fm(max_boundry_fm), dx(spatial_step_fm) {
                precompute_grid_and_baseline();
            }

            std::vector<Position> generate_nucleons() {
                std::vector<Position> nucleons;
                nucleons.reserve(config.A);
                std::uniform_real_distribution<double> dist_u(0.0, 1.0);
                std::uniform_real_distribution<double> dist_cos(-1.0, 1.0);
                std::uniform_real_distribution<double> dist_phi(0.0, 2.0 * M_PI);

                while (nucleons.size() < static_cast<size_t>(config.A)) {
                    double u = dist_u(rng);
                    double r = config.r_max * std::cbrt(u);

                    if (dist_u(rng) < woods_saxon_density(r)) {
                        double cos_theta = dist_cos(rng);
                        double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
                        double phi = dist_phi(rng);

                        Position p;
                        p.x = r * sin_theta * std::cos(phi);
                        p.y = r * sin_theta * std::sin(phi);
                        nucleons.push_back(p);
                    }
                }
                return nucleons;
            }

            std::vector<Position> generate_hotspots(const std::vector<Position>& nucleons, double W, int& actual_hotspots) {
                double mean_hs = config.calculate_mean_hotspots(W);
                double sigma_p = config.sigma_hotspot_position_fm();
                std::normal_distribution<double> gauss(0.0, sigma_p);

                std::vector<Position> hotspots;
                actual_hotspots = 0;

                for (const auto& n_center : nucleons) {
                    int num_hs = sample_zero_truncated_poisson(mean_hs);
                    actual_hotspots += num_hs;
                    for (int i = 0; i < num_hs; ++i) {
                        Position hs;
                        hs.x = n_center.x + gauss(rng);
                        hs.y = n_center.y + gauss(rng);
                        hotspots.push_back(hs);
                    }
                }
                return hotspots;
            }

            std::vector<double> compute_event_profile(const std::vector<Position>& hotspots, double epsilon = 1e-5) {
                std::vector<double> y_C(grid_size * grid_size, 0.0);
                std::vector<double> T_g_C(grid_size * grid_size, 0.0);

                double sigma_hs = config.sigma_hotspot_shape_fm();
                double norm = 1.0 / (2.0 * M_PI * sigma_hs * sigma_hs);
                double sum_T_g = 0.0;

                for (int r =0; r < grid_size; ++r) {
                    double y = x_coords[r];
                    for (int c = 0; c < grid_size; ++c) {
                        double x = x_coords[c];
                        double sum_gaussians = 0.0;

                        for (const auto& hs : hotspots) {
                            double dx_val = x - hs.x;
                            double dy_val = y - hs.y;
                            double r2 = dx_val*dx_val + dy_val*dy_val;
                            sum_gaussians += std::exp(-r2 / (2.0 * sigma_hs * sigma_hs));
                        }
                        double cell_val = sum_gaussians * norm;
                        T_g_C[r * grid_size + c] = cell_val;
                        sum_T_g += cell_val;
                    }
                }

                double integral_T_g = sum_T_g * (dx * dx);
                for (int i = 0; i < grid_size * grid_size; ++i) {
                    double T_g_C_hat = T_g_C[i] / (integral_T_g + 1e-12);
                    double R_g_C = T_g_C_hat / (T_WS_hat[i] + epsilon);
                    y_C[i] = std::log(R_g_C + epsilon);
                }

                return y_C;
            }

            void export_to_csv(const std::string& filename, const std::vector<double>& profile) {
                std::ofstream file(filename);
                file << "x_fm,y_fm,y_C\n";
                for (int r = 0; r < grid_size; ++r) {
                    for (int c = 0; c < grid_size; ++c) {
                        file << x_coords[c] << "," << x_coords[r] << "," << profile[r * grid_size + c] << "\n";
                    }
                }
                file.close();
            }

            int get_grid_size() const { return grid_size; }
};

int main() {
    SimulationConfig config;
    double grid_resolution = 0.1;
    DiscreteHotSpotSimulator simulator(config, grid_resolution, 15.0, 42);

    double W_test = 100.0;
    int test_runs = 5;
    bool write_output = false;

    std::cout << "Initializing Discrete Hot-Spot Profile Generator Matrix..." << std::endl;
    std::cout << "Resolution configuration: " << grid_resolution << " fm (" 
              << simulator.get_grid_size() << "x" << simulator.get_grid_size() << " cells)." << std::endl;

    // Guaranteed baseline ballpark check math
    double mean_hs_per_nucleon = config.calculate_mean_hotspots(W_test);
    double expected_ballpark_hotspots = mean_hs_per_nucleon * config.A;
    std::cout << "Guaranteed Ballpark Target (Mean Hotspots x A): " << expected_ballpark_hotspots << "\n\n";

    auto start_bench = std::chrono::high_resolution_clock::now();
    double total_hotspots_generated = 0;

    for (int i = 0; i < test_runs; ++i) {
        auto start_event = std::chrono::high_resolution_clock::now();
        
        auto nucleons = simulator.generate_nucleons();
        int actual_hs = 0;
        auto hotspots = simulator.generate_hotspots(nucleons, W_test, actual_hs);
        total_hotspots_generated += actual_hs;

        auto y_C_matrix = simulator.compute_event_profile(hotspots);

        auto end_event = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> event_ms = end_event - start_event;

        std::cout << " -> Event " << (i + 1) << " computed. Hotspots generated: " << actual_hs 
                  << " | Runtime: " << std::fixed << std::setprecision(2) << event_ms.count() << " ms" << std::endl;

        if (write_output && i == 0) { // Exports the first event to a CSV file as a proxy demonstrator [cite: 5, 9, 517]
            std::string out_name = "event_profile_res_" + std::to_string(grid_resolution) + ".csv";
            simulator.export_to_csv(out_name, y_C_matrix);
            std::cout << "    [Data Logger] Exported grid matrix state to: " << out_name << std::endl;
        }
    }

    auto end_bench = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_ms = end_bench - start_bench;

    std::cout << "\n=============================================" << std::endl;
    std::cout << "         PERFORMANCE PROFILE SUMMARY         " << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << "Total Iterations Processed : " << test_runs << std::endl;
    std::cout << "Guaranteed Ballpark Extrap : " << expected_ballpark_hotspots << std::endl;
    std::cout << "Avg Hot Spots Per Event    : " << total_hotspots_generated / test_runs << std::endl;
    std::cout << "Mean Execution Time        : " << total_ms.count() / test_runs << " ms" << std::endl;
    std::cout << "Estimated Output Processing: " << (1000.0 / (total_ms.count() / test_runs)) << " events/sec" << std::endl;
    std::cout << "=============================================\n" << std::endl;

    return 0;
}