# Complete Qwen 3.5 Integration Guide for PicoLM

## Table of Contents
1. [LLM Architecture Fundamentals](#llm-architecture-fundamentals)
2. [Comparing LLM Types](#comparing-llm-types)
3. [Qwen 3.5 Specific Architecture](#qwen-35-specific-architecture)
4. [Code Structure Changes](#code-structure-changes)
5. [Implementation Walkthrough](#implementation-walkthrough)

---

## LLM Architecture Fundamentals

### What is a Transformer?

All modern LLMs (GPT, Llama, Qwen, Phi, etc.) use the **Transformer architecture** introduced in "Attention is All You Need" (2017).

**Core Components**:
```
Input Tokens
    ↓
[Token Embedding]        ← Convert token IDs to vectors
    ↓
┌─────────────────────────────────────────┐
│  N Transformer Layers (stacked)         │
├─────────────────────────────────────────┤
│  For each layer:                        │
│  1. [Attention Block]                   │
│     - Multi-head self-attention         │
│     - Output projection                 │
│  2. [Feed-Forward Network (FFN)]        │
│     - 2-3 linear layers with activation │
│  3. [Residual Connections & Normalization] │
└─────────────────────────────────────────┘
    ↓
[Output Normalization]
    ↓
[Output Projection]      ← Project to vocab size
    ↓
Output Logits (probabilities for next token)
```

### The Attention Mechanism

**Self-Attention** is the "secret sauce" that makes transformers work:

```
For each position in the sequence:
1. Query (Q) = Linear(x)        ← "What am I looking for?"
2. Key (K) = Linear(x)          ← "What information do I have?"
3. Value (V) = Linear(x)        ← "What information should I pass forward?"

4. Attention scores = Q @ K^T / sqrt(d_head)
   ├─ Projects queries onto keys
   ├─ Normalized by sqrt(head_dim) for stability
   └─ Each token "attends" to all previous tokens (causal mask)

5. Attention weights = softmax(scores)
   └─ Convert scores to probabilities [0,1] summing to 1

6. Output = Attention weights @ V
   └─ Weighted sum of values
```

**Grouped-Query Attention (GQA)**:
- **Multi-Head Attention (MHA)**: Each head has its own Q, K, V
  - TinyLlama: 32 heads (full MHA)
  - Memory intensive: KV cache = seq_len × n_heads × head_dim

- **Grouped-Query Attention (GQA)**: Multiple Q heads share K, V heads
  - TinyLlama: 32 Q heads, 4 KV heads (8:1 sharing)
  - Qwen 3.5-0.8B: Similar GQA for memory efficiency
  - Memory efficient: KV cache = seq_len × n_kv_heads × head_dim (8x smaller)

**Visual Comparison**:
```
┌─ MHA (Full) ─────────────────────┐
│ Q1 Q2 Q3  Q32                     │
│  ↓  ↓  ↓   ↓                      │
│ K1 K2 K3  K32   V1 V2 V3  V32     │
│ One K and V per Q head            │
└───────────────────────────────────┘

┌─ GQA (Grouped) ───────────────────┐
│ Q1 Q2 ... Q8 | Q9 Q10 ... Q16 | Q17...  (32 Q heads) │
│      ↓               ↓                          ↓       │
│      K1              K2                         K4      │ (4 KV heads) │
│      V1              V2                         V4      │
│ Multiple Q heads share one K,V pair           │
└──────────────────────────────────────────────────────────┘
```

---

## Comparing LLM Types

### Architectural Differences

| Feature | TinyLlama | Llama 2 | Qwen 3.5 | Phi-2 | GPT-3.5/4 |
|---------|-----------|---------|----------|-------|-----------|
| **Base Model** | LLaMA | LLaMA | LLaMA-based | LLaMA-inspired | Transformer |
| **Parameters** | 1.1B | 7B-70B | 0.8B-110B | 2.7B | 175B-1.5T |
| **Attention** | GQA (32→4) | MHA | GQA (n_heads→n_kv_heads) | GQA | GQA/MQA |
| **FFN Type** | SwiGLU | SwiGLU | SwiGLU | Gated Linear Unit | Varies |
| **Activation** | SiLU | SiLU | SiLU | SiLU | GELU/others |
| **Norm Type** | RMSNorm | RMSNorm | RMSNorm | LayerNorm | LayerNorm |
| **Position Encoding** | RoPE | RoPE | RoPE + ALiBi | RoPE | ALiBi/RoPE |
| **Vocab Size** | 32,000 | 32,000 | ~152K | 51,200 | 50,257 |
| **Context Length** | 2,048 | 4,096 | 8,192+ | 2,048+ | 4,096-128K |

### Design Philosophy

**TinyLlama**: Minimal, fast, optimized for inference
- Uses standard LLaMA architecture
- Smaller embedding dimension
- Lower head count

**Llama 2**: Production-ready, balanced
- Well-tested, widely supported
- Medium to large parameter counts
- Good quality-to-size ratio

**Qwen 3.5**: Chinese LLM optimized for edge
- Better non-English support
- Optimized for mobile/embedded (like your use case!)
- Efficient architecture (0.8B variant)
- Larger vocabulary (better multilingual)

**Phi-2**: Compact research model
- Teaching/demo purposes
- Unusual: no Grouped-Query Attention
- Uses standard Linear layers in FFN

---

## The FFN (Feed-Forward Network) Problem

### Standard FFN (Transformer Original)
```
x → [Linear(d → 4d)]  ──→ [ReLU]  → [Linear(4d → d)]  → output
          ↓
    "expand"
```

**Formula**: 
```
FFN(x) = ReLU(xW1 + b1)W2 + b2
```

- **Issue**: ReLU can "dead" (output 0), losing information

### Gated Linear Unit (GLU) Family

**Why Gating?** Two pathways through FFN:
```
x ──→ [Linear: gate] ──→ [Sigmoid] ─────┐
                                        [×] → output
x ──→ [Linear: value] ────────────────────┘

Output = Sigmoid(gate) × value
```

**Why better**:
- Sigmoid output is [0,1], acts as "filter"
- Value pathway is preserved
- More expressive than standard FFN

### SwiGLU: Switched Gated Linear Unit (Used by Llama, Qwen, TinyLlama)

```
x ──────────────────→ [Linear(d → 4d): gate] ──→ [SiLU] ─────┐
                                                              [×] → output
x ──────→ [Linear(d → 4d): value_up] ───────────────────────┐
                                                              [×]
         [Linear(4d → d): down] ←──────────────────────────────
```

**Formula**:
```
h = SiLU(xW_gate + b_gate)  ⊙  (xW_up + b_up)    [⊙ means element-wise multiply]
output = hW_down + b_down
```

**In Code** (from PicoLM):
```c
// Line 689-695 in model.c
matmul(s->hb,  s->xb, lw->ffn_gate, dim, n_ffn, lw->type_ffn_gate);    // gate = x @ W_gate
matmul(s->hb2, s->xb, lw->ffn_up,   dim, n_ffn, lw->type_ffn_up);      // up = x @ W_up

silu(s->hb, n_ffn);                          // gate = SiLU(gate)
elemwise_mul(s->hb, s->hb, s->hb2, n_ffn);   // gate *= up (element-wise multiply)

matmul(s->xb, s->hb, lw->ffn_down, n_ffn, dim, lw->type_ffn_down);     // output = (gate*up) @ W_down
vec_add(s->x, s->xb, dim);                   // add residual
```

**Why "Switched"?**
- The gate "switches" the value pathway on/off
- Different than pure gating in some designs

### Why Qwen 3.5 Uses SwiGLU

```
Parameter efficiency:
- Standard FFN: d → 4d → d = 8d² parameters
- SwiGLU: (d → 4d gate) + (d → 4d up) + (4d → d down) 
        = d×4d + d×4d + 4d×d = 12d² parameters
        BUT: research shows SwiGLU uses ~2/3 the parameters for same quality
```

**Key Insight**: SwiGLU achieves better quality with fewer parameters → perfect for edge devices!

---

## Qwen 3.5 Specific Architecture

### Configuration Parameters

```yaml
# Qwen 3.5-0.8B typical config:
embedding_dimension: 1024          # n_embd
num_attention_heads: 16            # n_heads
num_kv_heads: 2                    # n_kv_heads (for GQA)
num_layers: 24                     # n_layers
intermediate_size: 2816           # n_ffn (hidden size in FFN)
vocab_size: 152064                # Much larger than Llama!
max_position_embeddings: 8192     # Context length
rope_theta: 1000000.0             # RoPE base (different from 10,000!)
```

### Why These Numbers?

**1. Large Vocabulary (152K vs 32K in Llama)**
```
- Llama: 32,000 tokens
- Qwen: 152,064 tokens
- Reason: Better coverage for Chinese + English + multilingual
- Impact: Final output layer is larger, embedding matrix is larger
```

**2. Head Count & GQA**
```
- 16 query heads
- 2 KV heads
- 8:1 ratio (each KV head is shared by 8 Q heads)
- Llama 2 7B: 32 heads, 8 kv heads = 4:1
- Qwen is MORE memory efficient
```

**3. RoPE Theta = 1,000,000**
```
- Controls how RoPE positions encode
- Llama uses 10,000
- Qwen 3.5 uses 1,000,000 for better long-range extrapolation
- Higher theta = can extend context beyond training length
```

**4. FFN Expansion (2816)**
```
n_ffn / n_embd = 2816 / 1024 = 2.75x
- Llama: 4x expansion
- Qwen: 2.75x (parameter efficient!)
- Smaller models benefit from this ratio
```

### Attention Pattern

```
Attention head dimension = n_embd / n_heads = 1024 / 16 = 64

For position encoding (RoPE):
head_dim / 2 = 32 pairs to encode

Query shape: [batch, seq_len, n_heads, head_dim] = [batch, seq_len, 16, 64]
Key shape:  [batch, seq_len, n_kv_heads, head_dim] = [batch, seq_len, 2, 64]
Value shape: [batch, seq_len, n_kv_heads, head_dim] = [batch, seq_len, 2, 64]

Each Query head can attend to all Key/Value heads through grouping
```

### Tensor Naming Convention

Qwen 3.5 in GGUF uses this pattern (must verify with your file!):

```
Layer structure:
blk.0.attn_norm.weight              ← Attention RMSNorm
blk.0.attn_q.weight                 ← Q projection (1024 → 1024)
blk.0.attn_k.weight                 ← K projection (1024 → 128, n_kv_heads*head_dim)
blk.0.attn_v.weight                 ← V projection (1024 → 128)
blk.0.attn_output.weight            ← Output (1024 → 1024)

blk.0.ffn_norm.weight               ← FFN RMSNorm
blk.0.ffn_gate.weight               ← Gate projection (1024 → 2816)
blk.0.ffn_up.weight                 ← Up projection (1024 → 2816)
blk.0.ffn_down.weight               ← Down projection (2816 → 1024)

Shared weights:
token_embd.weight                   ← [vocab_size=152064, 1024]
output_norm.weight                  ← Final RMSNorm [1024]
output.weight                       ← LM head [152064, 1024] or tied with token_embd
```

---

## Code Structure Changes

### Change 1: Update Metadata Key Parsing

**File**: `picolm/model.c`, function `parse_gguf()` (around line 235-260)

**Current Code**:
```c
if (str_eq(key, "qwen35.embedding_length") || str_eq(key, "general.embedding_length")) {
    int dummy; cfg->n_embd = (int)skip_meta_value(&r, vtype, &dummy);
}
```

**Problem**: Hard-coded keys, no fallback chain, no debug output

**Improved Version**:
```c
// Try multiple key variations in order of preference
if (str_eq(key, "qwen35.embedding_length")) {
    int dummy; cfg->n_embd = (int)skip_meta_value(&r, vtype, &dummy);
    fprintf(stderr, "Found qwen35.embedding_length = %d\n", cfg->n_embd);
} else if (str_eq(key, "general.embedding_length")) {
    int dummy; cfg->n_embd = (int)skip_meta_value(&r, vtype, &dummy);
    fprintf(stderr, "Found general.embedding_length = %d\n", cfg->n_embd);
} else if (str_eq(key, "llama.embedding_length")) {
    int dummy; cfg->n_embd = (int)skip_meta_value(&r, vtype, &dummy);
    fprintf(stderr, "Found llama.embedding_length = %d\n", cfg->n_embd);
}
```

### Change 2: Add Configuration Validation

**File**: `picolm/model.c`, after line 288

**Add**:
```c
// ---- Configuration Validation ----
fprintf(stderr, "\n=== Configuration Validation ===\n");

int config_ok = 1;
if (cfg->n_embd == 0) {
    fprintf(stderr, "ERROR: n_embd not loaded from metadata\n");
    config_ok = 0;
}
if (cfg->n_ffn == 0) {
    fprintf(stderr, "ERROR: n_ffn not loaded from metadata\n");
    config_ok = 0;
}
if (cfg->n_heads == 0) {
    fprintf(stderr, "ERROR: n_heads not loaded from metadata\n");
    config_ok = 0;
}
if (cfg->n_kv_heads == 0) {
    fprintf(stderr, "ERROR: n_kv_heads not loaded from metadata\n");
    config_ok = 0;
}
if (cfg->n_layers == 0) {
    fprintf(stderr, "ERROR: n_layers not loaded from metadata\n");
    config_ok = 0;
}
if (cfg->vocab_size == 0) {
    fprintf(stderr, "ERROR: vocab_size not loaded from metadata\n");
    config_ok = 0;
}

if (!config_ok) {
    fprintf(stderr, "\nDEBUG: Parsed config:\n");
    fprintf(stderr, "  n_embd=%d, n_ffn=%d, n_heads=%d, n_kv_heads=%d\n",
            cfg->n_embd, cfg->n_ffn, cfg->n_heads, cfg->n_kv_heads);
    fprintf(stderr, "  n_layers=%d, vocab_size=%d\n",
            cfg->n_layers, cfg->vocab_size);
    return -1;
}

fprintf(stderr, "✓ All config parameters validated\n");
fprintf(stderr, "  Head dimension: %d\n", cfg->head_dim);
fprintf(stderr, "  KV multiplier: %d\n", cfg->n_heads / cfg->n_kv_heads);
fprintf(stderr, "  FFN expansion: %.2f\n", (float)cfg->n_ffn / cfg->n_embd);
fprintf(stderr, "  Context: %d tokens\n", cfg->max_seq_len);
fprintf(stderr, "  Total params (approx): %.1fB\n",
        (float)(cfg->n_embd * cfg->n_embd +
                cfg->n_layers * (cfg->n_embd * cfg->n_ffn * 3 +
                                cfg->n_embd * cfg->n_embd * 4)) / 1e9);
```

### Change 3: Add Tensor Validation

**File**: `picolm/model.c`, after tensor loading (after line 378)

**Add**:
```c
// ---- Tensor Presence Validation ----
fprintf(stderr, "\n=== Tensor Validation ===\n");

int tensors_ok = 1;

// Critical tensors
if (!w->token_embd) {
    fprintf(stderr, "ERROR: token_embd.weight not found\n");
    tensors_ok = 0;
}
if (!w->output_norm) {
    fprintf(stderr, "ERROR: output_norm.weight not found\n");
    tensors_ok = 0;
}
if (!w->output) {
    fprintf(stderr, "ERROR: output.weight (LM head) not found\n");
    tensors_ok = 0;
}

// Check per-layer tensors
int layers_ok = 1;
for (int l = 0; l < cfg->n_layers; l++) {
    layer_weights_t *lw = &w->layers[l];
    if (!lw->attn_norm || !lw->attn_q || !lw->attn_k || !lw->attn_v || 
        !lw->attn_output || !lw->ffn_norm || !lw->ffn_gate || 
        !lw->ffn_up || !lw->ffn_down) {
        fprintf(stderr, "ERROR: Layer %d missing tensors\n", l);
        if (!lw->attn_norm) fprintf(stderr, "  - attn_norm\n");
        if (!lw->attn_q) fprintf(stderr, "  - attn_q\n");
        if (!lw->attn_k) fprintf(stderr, "  - attn_k\n");
        if (!lw->attn_v) fprintf(stderr, "  - attn_v\n");
        if (!lw->attn_output) fprintf(stderr, "  - attn_output\n");
        if (!lw->ffn_norm) fprintf(stderr, "  - ffn_norm\n");
        if (!lw->ffn_gate) fprintf(stderr, "  - ffn_gate\n");
        if (!lw->ffn_up) fprintf(stderr, "  - ffn_up\n");
        if (!lw->ffn_down) fprintf(stderr, "  - ffn_down\n");
        layers_ok = 0;
    }
}

if (!tensors_ok || !layers_ok) {
    fprintf(stderr, "\nFailed tensor check. Dumping first few tensor names:\n");
    return -1;
}

fprintf(stderr, "✓ All tensors validated (%d layers)\n", cfg->n_layers);
```

### Change 4: Handle Qwen's Large Vocabulary

**File**: `picolm/model.c`, around line 385-399

**Current**:
```c
if (cfg->vocab_size == 0) {
    for (uint64_t i = 0; i < n_tensors; i++) {
        if (str_eq(tinfos[i].name, "token_embd.weight")) {
            if (tinfos[i].n_dims >= 2) {
                int d0 = (int)tinfos[i].dims[0];
                int d1 = (int)tinfos[i].dims[1];
                cfg->vocab_size = (d0 == cfg->n_embd) ? d1 : d0;
            }
            break;
        }
    }
}
```

**Problem**: Doesn't log what vocab size was detected

**Improved**:
```c
if (cfg->vocab_size == 0) {
    for (uint64_t i = 0; i < n_tensors; i++) {
        if (str_eq(tinfos[i].name, "token_embd.weight")) {
            if (tinfos[i].n_dims >= 2) {
                int d0 = (int)tinfos[i].dims[0];
                int d1 = (int)tinfos[i].dims[1];
                // token_embd is [vocab_size, n_embd]
                cfg->vocab_size = (d0 == cfg->n_embd) ? d1 : d0;
                fprintf(stderr, "Detected vocab_size from token_embd: %d\n", cfg->vocab_size);
                
                // Qwen has very large vocab - validate reasonable range
                if (cfg->vocab_size < 1000 || cfg->vocab_size > 200000) {
                    fprintf(stderr, "WARNING: Unusual vocab_size %d (expected 10k-200k)\n", 
                            cfg->vocab_size);
                }
            }
            break;
        }
    }
}
```

### Change 5: Handle RoPE Theta Variations

**File**: `picolm/model.c`, around line 247-252

**Current**:
```c
} else if (str_eq(key, "qwen35.rope.freq_base")) {
    if (vtype == GGUF_META_FLOAT32) {
        cfg->rope_freq_base = read_f32(&r);
    } else {
        int dummy; skip_meta_value(&r, vtype, &dummy);
    }
```

**Improved**:
```c
} else if (str_eq(key, "qwen35.rope.freq_base")) {
    if (vtype == GGUF_META_FLOAT32) {
        cfg->rope_freq_base = read_f32(&r);
        fprintf(stderr, "Found qwen35.rope.freq_base = %.0f\n", cfg->rope_freq_base);
    } else {
        int dummy; skip_meta_value(&r, vtype, &dummy);
    }
} else if (str_eq(key, "llama.rope.freq_base")) {
    if (vtype == GGUF_META_FLOAT32) {
        cfg->rope_freq_base = read_f32(&r);
        fprintf(stderr, "Found llama.rope.freq_base = %.0f\n", cfg->rope_freq_base);
    } else {
        int dummy; skip_meta_value(&r, vtype, &dummy);
    }
}
```

### Change 6: Update Tokenizer for Qwen

**File**: `picolm/tokenizer.c`, function `tokenizer_load()` (around line TBD)

**Qwen-specific tokens**:
```c
// After loading tokenizer, add special token detection:
if (tokenizer_decode(tok, 0, 0) && strstr(tokenizer_decode(tok, 0, 0), "<|im_start|>")) {
    fprintf(stderr, "Detected Qwen-style chat tokens\n");
    // Qwen uses different chat template markers
}

// Validate special tokens
fprintf(stderr, "Tokenizer info:\n");
fprintf(stderr, "  Vocab size: %ld\n", tok->vocab_size);
fprintf(stderr, "  BOS token: %u\n", tok->bos_id);
fprintf(stderr, "  EOS token: %u\n", tok->eos_id);
```

---

## Implementation Walkthrough

### Step 1: Prepare Your Qwen 3.5 GGUF File

Before modifying code, inspect your file:

```bash
# Option A: Using python-gguf library
pip install gguf
python3 << 'EOF'
import gguf
reader = gguf.GGUFReader("qwen35-0.8b.gguf")

print("=== METADATA ===")
for key, val in reader.fields.items():
    if key.startswith(('qwen', 'general', 'llama')):
        print(f"{key}: {val.parts[-1] if hasattr(val, 'parts') else val}")

print("\n=== FIRST 30 TENSORS ===")
for i, tensor in enumerate(reader.tensors[:30]):
    print(f"{i}: {tensor.name} {tensor.tensor.shape}")
EOF

# Option B: Using strings (quick check)
strings qwen35-0.8b.gguf | grep -E "^(qwen|general|llama)" | head -20
```

**Expected output for Qwen 3.5-0.8B**:
```
qwen35.embedding_length: 1024
qwen35.feed_forward_length: 2816
qwen35.attention.head_count: 16
qwen35.attention.head_count_kv: 2
qwen35.block_count: 24
qwen35.context_length: 8192
qwen35.rope.freq_base: 1000000
qwen35.vocab_size: 152064

Tensors:
0: token_embd.weight [1024, 152064]
1: output_norm.weight [1024]
2: output.weight [1024, 152064] (or may be tied)
3: blk.0.attn_norm.weight [1024]
4: blk.0.attn_q.weight [1024, 1024]
...
```

### Step 2: Create a Test Build

Create a new branch for Qwen 3.5 support:

```bash
cd picolm
git checkout -b qwen35-support
```

### Step 3: Apply Changes Systematically

**3a. Update model.c metadata parsing:**

Replace lines 235-260 with improved version from Change 1.

**3b. Add configuration validation:**

Add code from Change 2 after line 288.

**3c. Add tensor validation:**

Add code from Change 3 after line 378.

**3d. Update vocab handling:**

Update lines 385-399 with improved version from Change 4.

**3e. Add RoPE handling:**

Update lines 247-252 with improved version from Change 5.

### Step 4: Test Step-by-Step

```bash
# Compile with debug info
make clean
make debug

# Test 1: Model loading with full debug
./picolm /path/to/qwen35.gguf -p "Hello" -n 5 2>&1 | head -100

# You should see:
# ✓ Configuration Validation
# ✓ All tensors validated
# Model config printout
```

**If you see errors**, check the debug output:
```
ERROR: n_embd not loaded from metadata
DEBUG: Parsed config: n_embd=0, ...
```

This means the metadata key name doesn't match. Use step 1 to find the actual key.

### Step 5: Handle Key Mismatches

If metadata keys differ, update the mapping:

```c
// In parse_gguf(), update key comparisons:
if (str_eq(key, "qwen35.embedding_length")) {
    // Primary key
} else if (str_eq(key, "qwen.embedding_length")) {
    // Alternative (if actual GGUF uses this)
} else if (str_eq(key, "general.embedding_length")) {
    // Fallback
}
```

### Step 6: Verify Forward Pass

Once model loads, test generation:

```bash
echo "The capital of France is" | ./picolm qwen35.gguf -n 10 -t 0

# Should produce reasonable output like:
# The capital of France is Paris. It is...
```

If output is gibberish or crashes, check:
1. All tensors loaded correctly (Change 3 validation)
2. FFN structure matches (likely still SwiGLU, but verify)
3. Vocab size is correct (impacts output projection)

---

## Common Qwen 3.5 Specific Issues

### Issue 1: Metadata Keys Don't Match

**Symptom**: `ERROR: n_embd not loaded`

**Root Cause**: Qwen GGUF might use different namespace (qwen, qwen2, general)

**Fix**:
```bash
# Find actual keys:
strings qwen35.gguf | grep "embedding_length"

# Update parse_gguf() with found keys
```

### Issue 2: Huge Vocabulary (152K)

**Symptom**: High memory usage, slow inference

**Root Cause**: Output projection is [152064, 1024] matrix

**Mitigation** (optional):
```c
// In allocate_run_state(), can reduce by limiting vocab:
if (cfg->vocab_size > 100000) {
    fprintf(stderr, "WARNING: Vocab size %d is very large\n", cfg->vocab_size);
    fprintf(stderr, "Consider using a smaller model or quantization\n");
}
```

### Issue 3: Different RoPE Theta

**Symptom**: Correct loading but wrong positional encoding beyond training length

**Note**: Qwen's 1M theta is intentional for long context

**No fix needed** - just recognize the difference.

### Issue 4: Tokenizer Special Tokens

**Symptom**: Chat format not recognized

**Root Cause**: Qwen uses `<|im_start|>` and `<|im_end|>` not standard GGML tokens

**Handle**: Already supported if GGUF metadata has tokenizer data

---

## Final Integration Checklist

- [ ] Inspect Qwen 3.5 GGUF file and document metadata keys
- [ ] Apply metadata parsing improvements (Change 1)
- [ ] Add configuration validation (Change 2)
- [ ] Add tensor validation (Change 3)
- [ ] Update vocab handling (Change 4)
- [ ] Update RoPE handling (Change 5)
- [ ] Compile without errors
- [ ] Model loads successfully with debug output
- [ ] Forward pass completes without crashes
- [ ] Output is coherent (not gibberish)
- [ ] Generation speed is acceptable (>0.5 tok/s on target device)
- [ ] Test with actual Qwen 3.5 model
- [ ] Create pull request with changes

---

## References

- **LLaMA Paper**: https://arxiv.org/abs/2302.13971
- **Attention is All You Need**: https://arxiv.org/abs/1706.03762
- **RoPE (Rotary Position Embedding)**: https://arxiv.org/abs/2104.09864
- **Grouped-Query Attention**: https://arxiv.org/abs/2305.13245
- **GLU Variants Improve Transformer**: https://arxiv.org/abs/2002.05202
- **Qwen Documentation**: https://github.com/QwenLM/Qwen
- **GGUF Format Spec**: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
