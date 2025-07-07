cmake_minimum_required(VERSION 3.16)
project(VIT_MNIST_PROJECT)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(include)

# Source files
set(SOURCES
    main.cpp
    src/matrix/matrix.cpp
    src/matrix/matrix_ops.cpp
    src/matrix/activation_functions.h.cpp
    src/utils/file_io.cpp
    src/transformer/layer_norm.cpp
    src/transformer/embedding.cpp
)

# Create executable
add_executable(vit_mnist ${SOURCES})

# Compiler flags
target_compile_options(vit_mnist PRIVATE -O2)