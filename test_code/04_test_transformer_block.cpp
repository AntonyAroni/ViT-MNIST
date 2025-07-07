#include "../include/transformer/transformer_block.h"
#include "../include/matrix/matrix.h"
#include <iostream>
/*
g++ -std=c++17 -I. test_code/04_test_transformer_block.cpp src/matrix/matrix.cpp src/matrix/matrix_ops.cpp src/matrix/activation_functions.h.cpp src/transformer/multi_head_attention.cpp src/transformer/mlp.cpp src/transformer/layer_norm.cpp src/transformer/transformer_block.cpp src/utils/file_io.cpp -o test_transformer_block && ./test_transformer_block
*/


int main() {
    try {
        std::cout << "Testing Transformer Block..." << std::endl;
        
        // MNIST ViT config
        size_t embed_dim = 256;
        size_t num_heads = 8;
        size_t mlp_hidden_dim = 1024;
        size_t seq_len = 50;
        
        TransformerBlock block(embed_dim, num_heads, mlp_hidden_dim);
        
        // Test input
        Matrix input = Matrix::random(seq_len, embed_dim);
        std::cout << "Input shape: " << input.getRows() << " x " << input.getCols() << std::endl;
        
        // Forward pass
        Matrix output = block.forward(input);
        std::cout << "Output shape: " << output.getRows() << " x " << output.getCols() << std::endl;
        std::cout << "✅ Transformer Block working!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}