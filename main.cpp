#include "DiscreteHotSpotSimulator.h"
#include <iostream>
#include <fstream>
#include <set>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <random>
#include <numeric>

int main() {
    SimulationConfig config;
    double spatial_step = 0.2;
    double max_boundary = 15.0;
    DiscreteHotSpotSimulator simulator(config, spatial_step, max_boundary, 12345);

    const int total_events_target = 100000;
    const int rows_per_file = 10000;
    
    const double W_min = 20.0;     
    const double W_max = 500.0;    
    const int W_steps = 1000; 
    const int runs_per_energy = 100; 
    
    // Cap at 7,000 hotspots (14,000 r, phi coordinate columns)
    const int max_hs_features = 7000; 

    std::random_device rd;
    std::set<unsigned long long> spent_seeds;
    
    std::cout << "==========================================================" << std::endl;
    std::cout << "  PARITY SIMULATION ENGINE (POLAR HOTSPOTS -> yc_hat DATASET) " << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << "Target Dataset Size : " << total_events_target << " entries." << std::endl;
    std::cout << "Energy Window       : [" << W_min << " to " << W_max << "] GeV" << std::endl;

    int generated_events = 0;
    int file_counter = 0;
    std::ofstream csv_file;

    auto execution_start = std::chrono::high_resolution_clock::now();
    double delta_W = (W_max - W_min) / static_cast<double>(W_steps - 1);

    for (int s = 0; s < W_steps && generated_events < total_events_target; ++s) {
        double current_W = W_min + s * delta_W;
        double bx = config.calculate_bjorken_x(current_W);
        double mean_hs = config.calculate_mean_hotspots(current_W);

        for (int r = 0; r < runs_per_energy && generated_events < total_events_target; ++r) {
            
            if (!csv_file.is_open()) {
                std::stringstream ss;
                ss << "hotspot_training_part_" << file_counter << ".csv";
                std::string current_filename = ss.str();
                
                csv_file.open(current_filename);
                if (!csv_file.is_open()) {
                    std::cerr << "Error: Failed to open: " << current_filename << std::endl;
                    return 1;
                }
                
                std::cout << " [Data Logger] Opening chunk: " << current_filename << std::endl;
                
                // --- 1. Write Clean Header Row ---
                csv_file << "R_value,event_id,seed,W_GeV,bjorken_x,mean_hs_expected,actual_hs_count,yc_hat";
                
                // Append Polar Hotspot Headers (r, phi)
                for (int i = 0; i < max_hs_features; ++i) {
                    csv_file << ",hs_" << i << "_r,hs_" << i << "_phi";
                }
                csv_file << "\n";
                file_counter++;
            }

            // Generate unique seed
            unsigned long long current_seed = ((unsigned long long)rd() << 32) | rd();
            while (spent_seeds.find(current_seed) != spent_seeds.end()) {
                current_seed = ((unsigned long long)rd() << 32) | rd();
            }
            spent_seeds.insert(current_seed);
            simulator.set_seed(current_seed);

            // --- Monte Carlo Generation ---
            auto nucleons = simulator.generate_nucleons();
            int actual_hs = 0;
            auto hotspots = simulator.generate_hotspots(nucleons, current_W, actual_hs);

            // --- Compute Nuclear Radius (R_value in fm) ---
            double sum_r_sq = 0.0;
            for (const auto& n : nucleons) {
                sum_r_sq += (n.x * n.x + n.y * n.y);
            }
            double R_value = (nucleons.empty()) ? 0.0 : std::sqrt(sum_r_sq / nucleons.size());

            // --- Compute y_C Density Profile and calculate yc_hat ---
            auto y_C_matrix = simulator.compute_event_profile(hotspots);
            double yc_sum = std::accumulate(y_C_matrix.begin(), y_C_matrix.end(), 0.0);
            double yc_hat = y_C_matrix.empty() ? 0.0 : (yc_sum / y_C_matrix.size());

            // --- 2. Write Metadata & Identifiers ---
            csv_file << std::fixed << std::setprecision(4) << R_value << ","
                     << generated_events << ","
                     << current_seed << ","
                     << std::fixed << std::setprecision(4) << current_W << ","
                     << std::scientific << std::setprecision(6) << bx << ","
                     << std::fixed << std::setprecision(3) << mean_hs << ","
                     << actual_hs << ","
                     << std::fixed << std::setprecision(6) << yc_hat;

            // --- 3. Write Hotspot Locations in Polar Coordinates (r, phi) ---
            int logged_hs = 0;
            for (const auto& hs : hotspots) {
                if (logged_hs >= max_hs_features) break;

                // Cartesian -> Polar Conversion
                double r_coord = std::sqrt(hs.x * hs.x + hs.y * hs.y);
                double phi_coord = std::atan2(hs.y, hs.x);

                csv_file << "," << std::fixed << std::setprecision(3) << r_coord
                         << "," << std::fixed << std::setprecision(3) << phi_coord;
                logged_hs++;
            }

            // Zero-pad remaining coordinate slots
            for (int i = logged_hs; i < max_hs_features; ++i) {
                csv_file << ",0,0";
            }
            csv_file << "\n";

            generated_events++;

            if (generated_events % rows_per_file == 0) {
                csv_file.close();
                auto check_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = check_time - execution_start;
                std::cout << " -> Completed part " << (file_counter - 1) 
                          << " (" << generated_events << " total events. Elapsed: " 
                          << std::fixed << std::setprecision(1) << elapsed.count() << "s)" << std::endl;
            }
        }
    }

    if (csv_file.is_open()) {
        csv_file.close();
    }
    
    auto execution_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_runtime = execution_end - execution_start;

    std::cout << "\n[Success] All data segments written completely." << std::endl;
    std::cout << "Total runtime: " << total_runtime.count() << " seconds." << std::endl;
    
    return 0;
}