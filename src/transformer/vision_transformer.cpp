#include "../../include/transformer/vision_transformer.h"
#include "../../include/matrix/matrix_ops.h"
#include "../../include/matrix/activation_functions.h"
#include <cmath>

VisionTransformer::VisionTransformer(size_t image_size, size_t patch_size, size_t embed_dim,
                                   size_t num_heads, size_t num_layers, size_t num_classes)
    : image_size(image_size), patch_size(patch_size), embed_dim(embed_dim),
      num_heads(num_heads), num_layers(num_layers), num_classes(num_classes),
      patch_embed(patch_size * patch_size, embed_dim) {
    
    num_patches = (image_size / patch_size) * (image_size / patch_size);
    
    // Initialize transformer blocks
    for (size_t i = 0; i < num_layers; ++i) {
        blocks.emplace_back(embed_dim, num_heads, embed_dim * 4);
    }
    
    initialize_weights();
}

void VisionTransformer::initialize_weights() {
    // Position embeddings (num_patches + 1 for cls token)
    pos_embedding = Matrix::random(num_patches + 1, embed_dim) * 0.02;
    
    // Class token
    cls_token = Matrix::random(1, embed_dim) * 0.02;
    
    // Classification head
    double scale = sqrt(2.0 / embed_dim);
    classifier_head = Matrix::random(embed_dim, num_classes) * scale;
}

Matrix VisionTransformer::image_to_patches(const Matrix& image) {
    // image: [28*28] flattened
    // Convert to patches: [num_patches, patch_size*patch_size]
    size_t patches_per_side = image_size / patch_size;
    Matrix patches(num_patches, patch_size * patch_size);
    
    for (size_t p = 0; p < num_patches; ++p) {
        size_t patch_row = p / patches_per_side;
        size_t patch_col = p % patches_per_side;
        
        for (size_t i = 0; i < patch_size; ++i) {
            for (size_t j = 0; j < patch_size; ++j) {
                size_t img_row = patch_row * patch_size + i;
                size_t img_col = patch_col * patch_size + j;
                size_t img_idx = img_row * image_size + img_col;
                size_t patch_idx = i * patch_size + j;
                
                patches(p, patch_idx) = image(0, img_idx);
            }
        }
    }
    
    return patches;
}

Matrix VisionTransformer::forward(const Matrix& images) {
    size_t batch_size = images.getRows();
    Matrix batch_output(batch_size, num_classes);
    
    // Process each image in batch
    for (size_t b = 0; b < batch_size; ++b) {
        // Extract single image
        Matrix single_image(1, images.getCols());
        for (size_t j = 0; j < images.getCols(); ++j) {
            single_image(0, j) = images(b, j);
        }
        
        // Convert to patches
        Matrix patches = image_to_patches(single_image);
        
        // Patch embedding
        Matrix patch_embeddings = patch_embed.forward(patches);
        
        // Add class token
        Matrix sequence(num_patches + 1, embed_dim);
        
        // Class token at position 0
        for (size_t j = 0; j < embed_dim; ++j) {
            sequence(0, j) = cls_token(0, j);
        }
        
        // Patch embeddings at positions 1 to num_patches
        for (size_t i = 0; i < num_patches; ++i) {
            for (size_t j = 0; j < embed_dim; ++j) {
                sequence(i + 1, j) = patch_embeddings(i, j);
            }
        }
        
        // Add position embeddings
        sequence = MatrixOps::add(sequence, pos_embedding);
        
        // Pass through transformer blocks
        Matrix x = sequence;
        for (size_t i = 0; i < num_layers; ++i) {
            x = blocks[i].forward(x);
        }
        
        // Extract class token (first token)
        Matrix cls_output(1, embed_dim);
        for (size_t j = 0; j < embed_dim; ++j) {
            cls_output(0, j) = x(0, j);
        }
        
        // Classification head
        Matrix logits = MatrixOps::matmul(cls_output, classifier_head);
        
        // Store in batch output
        for (size_t j = 0; j < num_classes; ++j) {
            batch_output(b, j) = logits(0, j);
        }
    }
    
    return batch_output;
}