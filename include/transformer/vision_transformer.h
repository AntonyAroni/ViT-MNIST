#ifndef VISION_TRANSFORMER_H
#define VISION_TRANSFORMER_H

#include "../matrix/matrix.h"
#include "transformer_block.h"
#include "embedding.h"
#include <vector>

class VisionTransformer {
private:
    size_t image_size;
    size_t patch_size;
    size_t num_patches;
    size_t embed_dim;
    size_t num_heads;
    size_t num_layers;
    size_t num_classes;
    
    PatchEmbedding patch_embed;
    Matrix pos_embedding;
    Matrix cls_token;
    std::vector<TransformerBlock> blocks;
    Matrix classifier_head;
    
public:
    VisionTransformer(size_t image_size, size_t patch_size, size_t embed_dim, 
                     size_t num_heads, size_t num_layers, size_t num_classes);
    
    Matrix forward(const Matrix& images);
    Matrix image_to_patches(const Matrix& image);
    void initialize_weights();
};

#endif