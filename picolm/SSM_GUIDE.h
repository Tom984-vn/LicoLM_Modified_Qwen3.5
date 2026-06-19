// ====================================================================
// SSM (State Space Model) Implementation Guide for Qwen 3.5
// ====================================================================
// 
// This file documents how to extend PicoLM with SSM support.
// Qwen 3.5 0.8B uses a hybrid Transformer + Mamba-style SSM architecture.
//
// Key Learning Points:
// 1. Understanding the difference between Transformer Attention vs SSM
// 2. Implementing selective state updates
// 3. Memory-efficient state management
// 4. How to parse Qwen GGUF files correctly
//
// ====================================================================

#ifndef SSM_H
#define SSM_H

#include <stdint.h>
#include <stddef.h>

/* ====================================================================
 * PART 1: SSM THEORY (IMPORTANT!)
 * ====================================================================
 *
 * Transformer Attention (Current PicoLM):
 *   - For each query token, compute attention over ALL past tokens
 *   - Complexity: O(n²) where n = sequence length
 *   - Requires storing Q, K, V for all positions (KV cache = huge memory)
 *
 * State Space Model (Mamba/Qwen 3.5):
 *   - Maintain a HIDDEN STATE that evolves over time
 *   - At each step: state = A * state + B * input
 *   - Output = C * state
 *   - Complexity: O(n) - LINEAR TIME!
 *   - State size << KV cache size
 *
 * Visual comparison:
 *
 *   ATTENTION (current):              SSM (Qwen 3.5):
 *   ┌─────────────────────┐          ┌──────────────┐
 *   │ Token 0 (position 0)│          │ Token 0      │
 *   └──────────┬──────────┘          └──────┬───────┘
 *              │                              │
 *   ┌─────────────────────┐          ┌──────────────┐
 *   │ Token 1 (pos 0,1)   │◄──────────│ Token 1      │◄─────┐
 *   │ Attend to 0,1       │ attention │ Update state │      │
 *   └──────────┬──────────┘          └──────┬───────┘      │
 *              │                             │               │
 *   ┌─────────────────────┐          ┌──────────────┐       │
 *   │ Token 2 (pos 0,1,2) │◄─────────│ Token 2      │       │
 *   │ Attend to 0,1,2     │          │ Update state │───────┘
 *   └─────────────────────┘          └──────────────┘
 *
 *   O(n²) complexity                  O(n) complexity
 *
 * ====================================================================
 */

/* ====================================================================
 * PART 2: SSM LAYER STRUCTURE
 * ====================================================================
 *
 * An SSM layer (also called "Mamba block") has:
 * - Input projection
 * - Selective state update
 * - Output projection
 *
 * In code, we need:
 */

typedef struct {
    // State parameters (A, B, C matrices - the core SSM)
    const void *mat_A;        // State transition matrix [state_dim, state_dim]
    const void *mat_B;        // Input matrix [state_dim, 1]
    const void *mat_C;        // Output matrix [hidden_dim, state_dim]
    const void *mat_D;        // Skip connection [hidden_dim] (optional)
    
    // Projections
    const void *proj_in;      // Input proj [state_dim, hidden_dim]
    const void *proj_out;     // Output proj [hidden_dim, state_dim]
    
    // Gating mechanism (the "selective" part)
    const void *gate_proj;    // Gating [state_dim, hidden_dim]
    
    // Quantization types for each weight
    int qtype_A, qtype_B, qtype_C, qtype_D;
    int qtype_in, qtype_out, qtype_gate;
    
    // Dimension info
    int state_dim;            // Hidden state dimension (usually 256-512)
    int hidden_dim;           // Model width (same as n_embd)
    
} ssm_layer_weights_t;

/* ====================================================================
 * PART 3: RUNTIME STATE FOR SSM
 * ====================================================================
 *
 * Unlike Transformer attention which needs KV cache for ALL past tokens,
 * SSM only needs to store the current hidden state.
 */

typedef struct {
    // Current hidden state (evolves per token)
    float *state;              // [state_dim] - updated each step
    
    // Temporary buffers for computation
    float *x_in;               // Input after projection [hidden_dim]
    float *x_gate;             // Gated input [hidden_dim]
    float *A_state;            // Temp: A * state [state_dim]
    float *B_x;                // Temp: B * input [state_dim]
    float *state_new;          // Temp: updated state [state_dim]
    float *C_state;            // Temp: C * state [hidden_dim]
    
    // Pre-dequantized norm weights (from parent model)
    float *norm_weights;       // Norm coefficients
    
} ssm_runtime_state_t;

/* ====================================================================
 * PART 4: FORWARD PASS - THE CORE SSM COMPUTATION
 * ====================================================================
 *
 * Here's what happens when you call model_forward() on an SSM layer:
 *
 * PSEUDOCODE:
 * 
 *   // 1. Input projection (token embedding -> state_dim)
 *   x_in = linear_layer(x, proj_in)
 *   
 *   // 2. Compute gating (decide how much to update)
 *   gate = sigmoid(linear_layer(x, gate_proj))
 *   x_gated = x_in * gate
 *   
 *   // 3. Selective state update (the SSM magic!)
 *   state_new = A @ state + B @ x_gated
 *   state = state_new
 *   
 *   // 4. Output from state
 *   y = C @ state + D * x_in  (residual)
 *   
 *   // 5. Output projection
 *   output = linear_layer(y, proj_out)
 *
 * Why this is efficient:
 * - State is small (256-512 dims) vs KV cache (millions of elements)
 * - Matrix A is sparse or structured (fast ops)
 * - Can skip gate computation when gate ≈ 0
 *
 * ====================================================================
 */

/* ====================================================================
 * PART 5: C CODE SKELETON
 * ====================================================================
 *
 * Here's how to implement it in C:
 */

// Simplified SSM forward pass
float *ssm_forward(
    float *output,            // Output [hidden_dim]
    const float *x,           // Input embedding [hidden_dim]
    const ssm_layer_weights_t *weights,
    ssm_runtime_state_t *state
) {
    int hidden_dim = weights->hidden_dim;
    int state_dim = weights->state_dim;
    
    // 1. Input projection
    // matmul(state->x_in, x, weights->proj_in, hidden_dim, state_dim, weights->qtype_in);
    
    // 2. Gating
    // matmul(state->x_gate, x, weights->gate_proj, hidden_dim, state_dim, weights->qtype_gate);
    // gate = sigmoid(x_gate)
    // x_in *= gate
    
    // 3. State update: state = A @ state + B @ x_in
    // matmul(state->A_state, state->state, weights->mat_A, state_dim, state_dim, weights->qtype_A);
    // matmul(state->B_x, state->x_in, weights->mat_B, state_dim, 1, weights->qtype_B);
    // state->state_new = A_state + B_x
    // memcpy(state->state, state->state_new, state_dim * sizeof(float));
    
    // 4. Output: y = C @ state
    // matmul(state->C_state, state->state, weights->mat_C, state_dim, hidden_dim, weights->qtype_C);
    
    // 5. Output projection
    // matmul(output, state->C_state, weights->proj_out, state_dim, hidden_dim, weights->qtype_out);
    
    return output;
}

/* ====================================================================
 * PART 6: PARSING QWEN GGUF FOR SSM
 * ====================================================================
 *
 * Qwen 3.5 GGUF files contain:
 *
 * Token embeddings:
 *   - "token_embd.weight"
 *
 * Per-layer tensors (for EACH block):
 *   For Transformer blocks (attention layers):
 *     - "blk.N.attn_norm.weight"
 *     - "blk.N.attn_q.weight"
 *     - "blk.N.attn_k.weight"
 *     - "blk.N.attn_v.weight"
 *     - "blk.N.attn_output.weight"
 *
 *   For SSM blocks (state space layers):
 *     - "blk.N.ssm_norm.weight"           ← NEW!
 *     - "blk.N.ssm_input_proj.weight"     ← NEW!
 *     - "blk.N.ssm_state_matrix.weight"   ← NEW! (the A matrix)
 *     - "blk.N.ssm_input_matrix.weight"   ← NEW! (the B matrix)
 *     - "blk.N.ssm_output_matrix.weight"  ← NEW! (the C matrix)
 *     - "blk.N.ssm_output_proj.weight"    ← NEW!
 *
 * Challenge: You need to:
 * 1. Detect if it's an SSM layer or Attention layer
 * 2. Allocate the right buffers
 * 3. Parse the right weight names
 *
 * ====================================================================
 */

/* ====================================================================
 * PART 7: KEY IMPLEMENTATION STEPS
 * ====================================================================
 *
 * To make Qwen 3.5 work:
 *
 * Step 1: Modify model.h to support SSM layers
 *   - Add ssm_layer_weights_t to model_weights_t
 *   - Add ssm_runtime_state_t to run_state_t
 *   - Increase state allocation
 *
 * Step 2: Modify parse_gguf() in model.c
 *   - Detect layer type (attention vs SSM) from tensor names
 *   - Load SSM weights into new structures
 *
 * Step 3: Modify model_forward()
 *   - After embedding, check if layer is SSM or attention
 *   - Call ssm_forward() or attention_forward() accordingly
 *
 * Step 4: Implement ssm_forward() in tensor.c or new ssm.c
 *   - Implement matrix-vector products for SSM
 *   - Implement gating mechanism
 *   - Implement state update
 *
 * Step 5: Test with Qwen GGUF file
 *   - Check first token output matches expected
 *   - Profile: SSM should be faster than attention
 *
 * ====================================================================
 */

/* ====================================================================
 * PART 8: ADVANCED C CODING PATTERNS USED HERE
 * ====================================================================
 *
 * 1. POINTER ARITHMETIC (like in PicoLM)
 *    const void *ptr = (const uint8_t *)base + offset;
 *    - Cast to uint8_t to do byte-level pointer math
 *    - Then recast to actual type
 *
 * 2. TYPE ERASURE
 *    void *data with int qtype field
 *    - Generic function pointer pattern (function overloading in C)
 *    - Switch on qtype to call Q4, Q8, F32 kernels
 *
 * 3. MEMORY POOLING
 *    Single malloc, carve pointers like:
 *    float *p = (float *)block; ptr1 = p; p += size1; ptr2 = p; p += size2;
 *    - Reduces fragmentation, cache-friendly
 *    - Mimics arena allocators
 *
 * 4. SIMD INTRINSICS (optional but powerful)
 *    #ifdef NEON / #ifdef SSE2 / #else
 *    - Auto-detect CPU capabilities
 *    - Compile-time conditional for ARM/x86/generic
 *
 * 5. PTHREADS for parallelization
 *    - Divide matmul rows across threads
 *    - Work-stealing pattern
 *
 * ====================================================================
 */

#endif /* SSM_H */
