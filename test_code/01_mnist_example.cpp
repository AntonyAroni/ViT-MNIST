#include "../include/utils/file_io.h"
#include "../include/matrix/matrix.h"
#include <iostream>



// g++ -std=c++17 -I. test_code/01_mnist_example.cpp src/matrix/matrix.cpp src/utils/file_io.cpp -o mnist_test && ./mnist_test

int main() {
    try {
        // Load training data
        std::cout << "Loading MNIST training images..." << std::endl;
        Matrix train_images = FileIO::load_mnist_images("data/train-images-idx3-ubyte/train-images-idx3-ubyte");
        std::vector<int> train_labels = FileIO::load_mnist_labels("data/train-labels-idx1-ubyte/train-labels-idx1-ubyte");
        
        std::cout << "Training images: " << train_images.getRows() << " x " << train_images.getCols() << std::endl;
        std::cout << "Training labels: " << train_labels.size() << std::endl;
        
        // Load test data
        std::cout << "Loading MNIST test images..." << std::endl;
        Matrix test_images = FileIO::load_mnist_images("data/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte");
        std::vector<int> test_labels = FileIO::load_mnist_labels("data/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte");
        
        std::cout << "Test images: " << test_images.getRows() << " x " << test_images.getCols() << std::endl;
        std::cout << "Test labels: " << test_labels.size() << std::endl;
        
        // Show first few samples
        std::cout << "\nFirst 5 training labels: ";
        for (int i = 0; i < 5; ++i) {
            std::cout << train_labels[i] << " ";
        }
        std::cout << std::endl;
        
        // Show pixel range for first image
        std::cout << "First image pixel range: " << train_images(0, 0) << " to ";
        double max_val = 0;
        for (size_t j = 0; j < train_images.getCols(); ++j) {
            if (train_images(0, j) > max_val) max_val = train_images(0, j);
        }
        std::cout << max_val << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}