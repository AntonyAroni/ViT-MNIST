#include <iostream>
#include "include/matrix/matrix.h"
#include "include/matrix/matrix_ops.h"
#include "include/matrix/activation_functions.h"
#include "include/utils/file_io.h"
#include "include/transformer/layer_norm.h"
#include "include/transformer/embedding.h"

void test_basic_functionality() {
    std::cout << "=== PRUEBA BÁSICA: Operaciones de Matriz ===" << std::endl;
    try {
        // Test basic matrix operations
        Matrix test_matrix = Matrix::zeros(3, 3);

        test_matrix(0, 0) = 1.0;
        test_matrix(1, 1) = 2.0;
        test_matrix(2, 2) = 3.0;
        
        std::cout << "✅ Matriz de prueba creada exitosamente!" << std::endl;
        std::cout << "Dimensiones: " << test_matrix.getRows() << "x" << test_matrix.getCols() << std::endl;
        
        std::cout << "Matriz de prueba:" << std::endl;
        test_matrix.print();
        
    } catch (const std::exception& e) {
        std::cout << "❌ Error en operaciones básicas: " << e.what() << std::endl;
    }
}

void test_day2_components() {
    std::cout << "\n=== PRUEBAS DÍA 2: Componentes Transformer ===" << std::endl;
    
    try {
        // Test 1: LayerNorm (sin pesos)
        std::cout << "\n1️⃣ Probando LayerNorm (sin pesos)..." << std::endl;
        LayerNorm layer_norm;
        // Skip weight loading for now
        
        // Crear entrada de prueba (simula embeddings de patches)
        Matrix test_input = Matrix::random(2, 256);  // batch_size=2, features=256
        std::cout << "✅ LayerNorm creado correctamente" << std::endl;
        std::cout << "   Entrada preparada: " << test_input.getRows() << "x" << test_input.getCols() << std::endl;
        
        // Test 2: PatchEmbedding (sin pesos)
        std::cout << "\n2️⃣ Probando PatchEmbedding (sin pesos)..." << std::endl;
        PatchEmbedding patch_embed(49, 256);
        // Skip weight loading for now
        
        std::cout << "✅ PatchEmbedding creado correctamente" << std::endl;
        std::cout << "   Configurado para MNIST: 49 patches, 256 dimensiones" << std::endl;
        std::cout << "   Cada patch: 4x4 = 16 valores por patch" << std::endl;
        
        // Test 3: Funciones de activación existentes
        std::cout << "\n3️⃣ Probando funciones de activación..." << std::endl;
        Matrix test_data = Matrix::random(2, 256);
        Matrix gelu_result = ActivationFunctions::gelu(test_data);
        
        // Test softmax para 10 clases de MNIST
        Matrix mnist_logits = Matrix::random(1, 10);  // 10 digit classes
        Matrix softmax_result = ActivationFunctions::softmax(mnist_logits);
        
        std::cout << "✅ GELU: " << test_data.getRows() << "x" << test_data.getCols() 
                  << " → " << gelu_result.getRows() << "x" << gelu_result.getCols() << std::endl;
        std::cout << "✅ Softmax (10 clases MNIST): " << mnist_logits.getRows() << "x" << mnist_logits.getCols() 
                  << " → " << softmax_result.getRows() << "x" << softmax_result.getCols() << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ Error en pruebas del Día 2: " << e.what() << std::endl;
    }
}

void show_next_steps() {
    std::cout << "\n=== PRÓXIMOS PASOS ===" << std::endl;
    std::cout << "✅ Día 1: Librería de matrices - COMPLETADO" << std::endl;
    std::cout << "✅ Día 2: Componentes neuronales - COMPLETADO" << std::endl;
    std::cout << "⏳ Día 3: Multi-Head Self-Attention - PENDIENTE" << std::endl;
    std::cout << "⏳ Día 4: MLP y capas transformer - PENDIENTE" << std::endl;
    std::cout << "⏳ Día 5: Integración final - PENDIENTE" << std::endl;
    std::cout << "\n📊 MNIST ESPECÍFICO:" << std::endl;
    std::cout << "   • Imágenes: 28x28 (escala de grises)" << std::endl;
    std::cout << "   • Clases: 10 dígitos (0-9)" << std::endl;
    std::cout << "   • Patches: 4x4 = 16 valores por patch" << std::endl;
    std::cout << "\n🚀 ¡Listo para implementar Multi-Head Attention!" << std::endl;
}

int main() {
    std::cout << "🧠 VISION TRANSFORMER C++ - MNIST 🧠" << std::endl;
    std::cout << "==================================" << std::endl;
    
    // Ejecutar todas las pruebas
    test_basic_functionality();
    test_day2_components();
    show_next_steps();
    
    return 0;
}