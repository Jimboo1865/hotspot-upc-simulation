#include "DiscreteHotSpotSimulator.h"
#include <set>
#include <sstream>

int main() {
    SimulationConfig config;
    DiscreteHotSpotSimulator simulator(config, 0.1, 15.0, 12345);

    // High-throughput configuration variables
    const int total_events_target = 100000;
    const int rows_per_file = 10000; // Automatically splits into 10 manageable files
    
    const double W_min = 20.0;     
    const double W_max = 500.0;    
    const int W_steps = 1000; 
    const int runs_per_energy = 100; 
    const int max_hs_features = 12000; 

    // Setup non-repeating random seed validation
    std::random_device rd;
    std::set<unsigned long long> spent_seeds;
    
    std::cout << "==========================================================" << std::endl;
    std::cout << "       PARITY SIMULATION DATA GENERATION ENGINE           " << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << "Target Dataset Size : " << total_events_target << " entries." << std::endl;
    std::cout << "Chunk Profile Size  : " << rows_per_file << " rows per CSV." << std::endl;
    std::cout << "Energy Sampling Window: [" << W_min << " to " << W_max << "] GeV" << std::endl;

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
            
            // Check if we need to open a brand-new file chunk
            if (!csv_file.is_open()) {
                std::stringstream ss;
                ss << "hotspot_training_part_" << file_counter << ".csv";
                std::string current_filename = ss.str();
                
                csv_file.open(current_filename);
                if (!csv_file.is_open()) {
                    std::cerr << "Error: Failed to open: " << current_filename << std::endl;
                    return 1;
                }
                
                std::cout << " [Data Logger] Opening new chunk: " << current_filename << std::endl;
                
                // Write fresh headers for the new file
                csv_file << "event_id,seed,W_GeV,bjorken_x,mean_hs_expected,actual_hs_count";
                for (int i = 0; i < max_hs_features; ++i) {
                    csv_file << ",hs_" << i << "_x,hs_" << i << "_y";
                }
                csv_file << "\n";
                file_counter++;
            }

            // Unique seed generation
            unsigned long long current_seed = ((unsigned long long)rd() << 32) | rd();
            while (spent_seeds.find(current_seed) != spent_seeds.end()) {
                current_seed = ((unsigned long long)rd() << 32) | rd();
            }
            spent_seeds.insert(current_seed);
            simulator.set_seed(current_seed);

            // Execute Monte Carlo generation
            auto nucleons = simulator.generate_nucleons();
            int actual_hs = 0;
            auto hotspots = simulator.generate_hotspots(nucleons, current_W, actual_hs);

            // Row writing
            csv_file << generated_events << ","
                     << current_seed << ","
                     << std::fixed << std::setprecision(4) << current_W << ","
                     << std::scientific << std::setprecision(6) << bx << ","
                     << std::fixed << std::setprecision(3) << mean_hs << ","
                     << actual_hs;

            int logged_hs = 0;
            for (const auto& hs : hotspots) {
                if (logged_hs >= max_hs_features) break;
                csv_file << "," << std::fixed << std::setprecision(5) << hs.x 
                         << "," << std::fixed << std::setprecision(5) << hs.y;
                logged_hs++;
            }

            // Pad remaining columns out to 12,000
            for (int i = logged_hs; i < max_hs_features; ++i) {
                csv_file << ",0.0,0.0";
            }
            csv_file << "\n";

            generated_events++;

            // Close file if chunk target is hit
            if (generated_events % rows_per_file == 0) {
                csv_file.close();
                auto check_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = check_time - execution_start;
                std::cout << " -> Completed part " << (file_counter - 1) 
                          << " (" << generated_events << " total events processed. Elapsed: " 
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
    std::cout << "Total runtime metrics: " << total_runtime.count() << " seconds." << std::endl;
    
    return 0;
}