#include "../include/transformer/multi_head_attention.h"
#include "../include/matrix/matrix.h"
#include <iostream>

/*
 g++ -std=c++17 -I. test_code/02_test_attention.cpp src/matrix/matrix.cpp src/matrix/matrix_ops.cpp src/matrix/activation_functions.h.cpp src/transformer/multi_head_attention.cpp -o test_attention && ./test_attention
*/
int main() {
    try {
        std::cout << "Testing Multi-Head Self-Attention..." << std::endl;
        
        // MNIST ViT typical config: 256 embed_dim, 8 heads
        size_t embed_dim = 256;
        size_t num_heads = 8;
        size_t seq_len = 50;  // 49 patches + 1 class token
        
        MultiHeadAttention mha(embed_dim, num_heads);
        
        // Create test input (sequence of embeddings)
        Matrix input = Matrix::random(seq_len, embed_dim);
        
        std::cout << "Input shape: " << input.getRows() << " x " << input.getCols() << std::endl;
        
        // Forward pass
        Matrix output = mha.forward(input);
        
        std::cout << "Output shape: " << output.getRows() << " x " << output.getCols() << std::endl;
        std::cout << "✅ Multi-Head Attention working!" << std::endl;
        
        // Check output range
        double min_val = output(0, 0), max_val = output(0, 0);
        for (size_t i = 0; i < output.getRows(); ++i) {
            for (size_t j = 0; j < output.getCols(); ++j) {
                if (output(i, j) < min_val) min_val = output(i, j);
                if (output(i, j) > max_val) max_val = output(i, j);
            }
        }
        std::cout << "Output range: [" << min_val << ", " << max_val << "]" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}