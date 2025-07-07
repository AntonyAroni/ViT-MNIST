#include "../include/transformer/mlp.h"
#include "../include/matrix/matrix.h"
#include <iostream>
/*
g++ -std=c++17 -I. test_code/03_test_mlp.cpp src/matrix/matrix.cpp src/matrix/matrix_ops.cpp src/matrix/activation_functions.h.cpp src/transformer/mlp.cpp -o test_mlp && ./test_mlp

 */
int main() {
    try {
        std::cout << "Testing MLP Block..." << std::endl;
        
        // MNIST ViT config: 256 input, 1024 hidden (4x expansion)
        size_t input_dim = 256;
        size_t hidden_dim = 1024;
        size_t seq_len = 50;
        
        MLP mlp(input_dim, hidden_dim);
        
        // Test input
        Matrix input = Matrix::random(seq_len, input_dim);
        std::cout << "Input shape: " << input.getRows() << " x " << input.getCols() << std::endl;
        
        // Forward pass
        Matrix output = mlp.forward(input);
        std::cout << "Output shape: " << output.getRows() << " x " << output.getCols() << std::endl;
        std::cout << "✅ MLP Block working!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}