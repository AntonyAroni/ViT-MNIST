#include "../include/transformer/vision_transformer.h"
#include "../include/utils/file_io.h"
#include "../include/matrix/activation_functions.h"
#include <iostream>

/*
g++ -std=c++17 -I. test_code/05_test_vit.cpp src/matrix/matrix.cpp src/matrix/matrix_ops.cpp src/matrix/activation_functions.h.cpp src/transformer/multi_head_attention.cpp src/transformer/mlp.cpp src/transformer/layer_norm.cpp src/transformer/transformer_block.cpp src/transformer/embedding.cpp src/transformer/vision_transformer.cpp src/utils/file_io.cpp -o test_vit && ./test_vit

 */
int main() {
    try {
        std::cout << "Testing Complete Vision Transformer..." << std::endl;
        
        // MNIST ViT config
        size_t image_size = 28;
        size_t patch_size = 4;  // 7x7 patches
        size_t embed_dim = 256;
        size_t num_heads = 8;
        size_t num_layers = 6;
        size_t num_classes = 10;
        
        VisionTransformer vit(image_size, patch_size, embed_dim, num_heads, num_layers, num_classes);
        
        std::cout << "Loading MNIST data..." << std::endl;
        Matrix test_images = FileIO::load_mnist_images("data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte");
        std::vector<int> test_labels = FileIO::load_mnist_labels("data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte");
        
        // Test with first 5 images
        Matrix batch(5, test_images.getCols());
        for (size_t i = 0; i < 5; ++i) {
            for (size_t j = 0; j < test_images.getCols(); ++j) {
                batch(i, j) = test_images(i, j);
            }
        }
        
        std::cout << "Running inference..." << std::endl;
        Matrix logits = vit.forward(batch);
        
        std::cout << "Predictions for first 5 images:" << std::endl;
        for (size_t i = 0; i < 5; ++i) {
            // Apply softmax to get probabilities
            Matrix single_logit(1, num_classes);
            for (size_t j = 0; j < num_classes; ++j) {
                single_logit(0, j) = logits(i, j);
            }
            Matrix probs = ActivationFunctions::softmax(single_logit);
            
            // Find predicted class
            size_t pred_class = 0;
            double max_prob = probs(0, 0);
            for (size_t j = 1; j < num_classes; ++j) {
                if (probs(0, j) > max_prob) {
                    max_prob = probs(0, j);
                    pred_class = j;
                }
            }
            
            std::cout << "Image " << i << ": True=" << test_labels[i] 
                      << ", Pred=" << pred_class << ", Conf=" << max_prob << std::endl;
        }
        
        std::cout << "✅ Vision Transformer working!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}