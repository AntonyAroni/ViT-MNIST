//
// Created by JAYAN on 01/07/2025.
//

#ifndef FILE_IO_H
#define FILE_IO_H

#include <string>
#include <vector>
#include "../matrix/matrix.h"

namespace FileIO {
    
    // Load matrix from CSV file
    Matrix load_matrix_from_csv(const std::string& filename, bool has_header = false);
    
    // Load vector from CSV file (single column or row)
    std::vector<double> load_vector_from_csv(const std::string& filename, bool has_header = false);
    
    // Load vector from CSV file as Matrix (single row vector)
    Matrix load_vector_as_matrix(const std::string& filename, bool has_header = false);
    
    // Save matrix to CSV file
    void save_matrix_to_csv(const Matrix& matrix, const std::string& filename);
    
    // MNIST IDX format loaders
    Matrix load_mnist_images(const std::string& filename);
    std::vector<int> load_mnist_labels(const std::string& filename);
    
    // Utility functions
    std::vector<std::string> split_string(const std::string& str, char delimiter);
    bool file_exists(const std::string& filename);
    uint32_t read_big_endian_uint32(std::ifstream& file);
}

#endif //FILE_IO_H