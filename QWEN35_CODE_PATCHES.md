# Qwen 3.5 Code Implementation - Ready-to-Use Patches

This file contains copy-paste-ready code blocks to integrate Qwen 3.5 support into PicoLM.

## File 1: model.c - Enhanced Metadata Parsing

### Patch 1a: Improve parse_gguf() metadata keys (around line 235-260)

Replace the existing metadata parsing section with:

```c
// ---- Enhanced metadata parsing with Qwen 3.5 support ----
for (uint64_t i = 0; i < n_metadata; i++) {
    gguf_str_t key = read_gguf_string(&r);
    uint32_t vtype = read_u32(&r);
    
    // Print debug info for important keys
    if (vtype < 10) {  // Skip complex types
        fprintf(stderr, "META: %.*s (type=%u)\n", (int)key.len, key.str, vtype);
    }

    // EMBEDDING DIMENSION
    if (str_eq(key, "qwen35.embedding_length")) {
        int dummy; 
        cfg->n_embd = (int)skip_meta_value(&r, vtype, &dummy);
        fprintf(stderr, "✓ Found qwen35.embedding_length = %d\n", cfg->n_embd);
    } else if (str_eq(key, "qwen.embedding_length")) {
        int dummy; 
        cfg->n_embd = (int)skip_meta_value(&r, vtype, &dummy);
        fprintf(stderr, "✓ Found qwen.embedding_length = %d\n", cfg->n_embd);
    } else if (str_eq(key, "llama.embedding_length")) {
        int dummy; 
        cfg->n_embd = (int)skip_meta_value(&r, vtype, &dummy);
        fprintf(stderr, "✓ Found llama.embedding_length = %d\n", cfg->n_embd);
    } else if (str_eq(key, "general.embedding_length")) {
        int dummy; 
        cfg->n_embd = (int)skip_meta_value(&r, vtype, &dummy);
        fprintf(stderr, "✓ Found general.embedding_length = %d\n", cfg->n_embd);
    }
    
    // FEED-FORWARD DIMENSION
    else if (str_eq(key, "qwen35.feed_forward_length")) {
        int dummy; 
        cfg->n_ffn = (int)skip_meta_value(&r, vtype, &dummy);
        fprintf(stderr, "✓ Found qwen35.feed_forward_length = %d\n", cfg->n_ffn);
    } else if (str_eq(key, "qwen.feed_forward_length")) {
        int dummy; 
        cfg->n_ffn = (int)skip_meta_value(&r, vtype, &dummy);
        fprintf(stderr, "✓ Found qwen.feed_forward_length = %d\n", cfg->n_ffn);
    } else if (str_eq(key, "llama.feed_forward_length")) {
        int dummy; 
        cfg->n_ffn = (int)skip_meta_value(&r, vtype, &dummy);
        fprintf(stderr, "✓ Found llama.feed_forward_length = %d\n", cfg->n_ffn);
    }
    
    // ATTENTION HEAD COUNT
    else if (str_eq(key, "qwen35.attention.head_count")) {
        int dummy; 
        cfg->n_heads = (int)skip_meta_value(&r, vtype, &dummy);
        fprintf(stderr, "✓ Found qwen35.attention.head_count = %d\n", cfg->n_heads);
    } else if (str_eq(key, "llama.attention.head_count")) {
        int dummy; 
        cfg->n_heads = (int)skip_meta_value(&r, vtype, &dummy);
        fprintf(stderr, "✓ Found llama.attention.head_count = %d\n", cfg->n_heads);
    }
    
    // KV HEAD COUNT (for GQA)
    else if (str_eq(key, "qwen35.attention.head_count_kv")) {
        int dummy; 
        cfg->n_kv_heads = (int)skip_meta_value(&r, vtype, &dummy);
        fprintf(stderr, "✓ Found qwen35.attention.head_count_kv = %d\n", cfg->n_kv_heads);
    } else if (str_eq(key, "llama.attention.head_count_kv")) {
        int dummy; 
        cfg->n_kv_heads = (int)skip_meta_value(&r, vtype, &dummy);
        fprintf(stderr, "✓ Found llama.attention.head_count_kv = %d\n", cfg->n_kv_heads);
    }
    
    // NUMBER OF LAYERS
    else if (str_eq(key, "qwen35.block_count")) {
        int dummy; 
        cfg->n_layers = (int)skip_meta_value(&r, vtype, &dummy);
        fprintf(stderr, "✓ Found qwen35.block_count = %d\n", cfg->n_layers);
    } else if (str_eq(key, "llama.block_count")) {
        int dummy; 
        cfg->n_layers = (int)skip_meta_value(&r, vtype, &dummy);
        fprintf(stderr, "✓ Found llama.block_count = %d\n", cfg->n_layers);
    }
    
    // CONTEXT LENGTH / MAX SEQUENCE LENGTH
    else if (str_eq(key, "qwen35.context_length")) {
        int dummy; 
        cfg->max_seq_len = (int)skip_meta_value(&r, vtype, &dummy);
        fprintf(stderr, "✓ Found qwen35.context_length = %d\n", cfg->max_seq_len);
    } else if (str_eq(key, "llama.context_length")) {
        int dummy; 
        cfg->max_seq_len = (int)skip_meta_value(&r, vtype, &dummy);
        fprintf(stderr, "✓ Found llama.context_length = %d\n", cfg->max_seq_len);
    }
    
    // RoPE FREQUENCY BASE
    else if (str_eq(key, "qwen35.rope.freq_base")) {
        if (vtype == GGUF_META_FLOAT32) {
            cfg->rope_freq_base = read_f32(&r);
            fprintf(stderr, "✓ Found qwen35.rope.freq_base = %.0f\n", cfg->rope_freq_base);
        } else {
            int dummy; skip_meta_value(&r, vtype, &dummy);
        }
    } else if (str_eq(key, "llama.rope.freq_base")) {
        if (vtype == GGUF_META_FLOAT32) {
            cfg->rope_freq_base = read_f32(&r);
            fprintf(stderr, "✓ Found llama.rope.freq_base = %.0f\n", cfg->rope_freq_base);
        } else {
            int dummy; skip_meta_value(&r, vtype, &dummy);
        }
    }
    
    // VOCAB SIZE
    else if (str_eq(key, "qwen35.vocab_size")) {
        int dummy; 
        cfg->vocab_size = (int)skip_meta_value(&r, vtype, &dummy);
        fprintf(stderr, "✓ Found qwen35.vocab_size = %d\n", cfg->vocab_size);
    } else if (str_eq(key, "general.vocab_size")) {
        int dummy; 
        cfg->vocab_size = (int)skip_meta_value(&r, vtype, &dummy);
        fprintf(stderr, "✓ Found general.vocab_size = %d\n", cfg->vocab_size);
    }
    
    // ALIGNMENT
    else if (str_eq(key, "general.alignment")) {
        int dummy; 
        cfg->alignment = (int)skip_meta_value(&r, vtype, &dummy);
    }
    
    // SPECIAL TOKENS
    else if (str_eq(key, "tokenizer.ggml.bos_token_id")) {
        int dummy; 
        m->tok_bos_id = (uint32_t)skip_meta_value(&r, vtype, &dummy);
    } else if (str_eq(key, "tokenizer.ggml.eos_token_id")) {
        int dummy; 
        m->tok_eos_id = (uint32_t)skip_meta_value(&r, vtype, &dummy);
    }
    
    // TOKENIZER DATA
    else if (str_eq(key, "tokenizer.ggml.tokens")) {
        if (vtype != GGUF_META_ARRAY) {
            int dummy; skip_meta_value(&r, vtype, &dummy);
        } else {
            uint32_t arr_type = read_u32(&r);
            uint64_t arr_len  = read_u64(&r);
            m->tok_tokens_data = r.data + r.pos;
            m->tok_n_tokens = arr_len;
            int dummy;
            for (uint64_t j = 0; j < arr_len; j++) {
                skip_meta_value(&r, arr_type, &dummy);
            }
        }
    } else if (str_eq(key, "tokenizer.ggml.scores")) {
        if (vtype != GGUF_META_ARRAY) {
            int dummy; skip_meta_value(&r, vtype, &dummy);
        } else {
            uint32_t arr_type = read_u32(&r);
            uint64_t arr_len  = read_u64(&r);
            (void)arr_type;
            m->tok_scores_data = r.data + r.pos;
            m->tok_n_scores = arr_len;
            r.pos += arr_len * 4;
        }
    } else {
        int dummy; skip_meta_value(&r, vtype, &dummy);
    }
}
```

### Patch 1b: Add configuration validation (insert after line 288)

```c
// ====================================================================
// CONFIGURATION VALIDATION - CRITICAL FOR QWEN 3.5
// ====================================================================

fprintf(stderr, "\n╔══════════════════════════════════════════════════════╗\n");
fprintf(stderr, "║         Configuration Validation (Qwen 3.5)         ║\n");
fprintf(stderr, "╚══════════════════════════════════════════════════════╝\n\n");

int config_ok = 1;
const char *missing_params[10];
int num_missing = 0;

// Check all required parameters
if (cfg->n_embd == 0) {
    fprintf(stderr, "❌ ERROR: n_embd (embedding dimension) not found\n");
    missing_params[num_missing++] = "n_embd";
    config_ok = 0;
}
if (cfg->n_ffn == 0) {
    fprintf(stderr, "❌ ERROR: n_ffn (FFN hidden size) not found\n");
    missing_params[num_missing++] = "n_ffn";
    config_ok = 0;
}
if (cfg->n_heads == 0) {
    fprintf(stderr, "❌ ERROR: n_heads (attention heads) not found\n");
    missing_params[num_missing++] = "n_heads";
    config_ok = 0;
}
if (cfg->n_kv_heads == 0) {
    fprintf(stderr, "❌ ERROR: n_kv_heads (KV heads for GQA) not found\n");
    missing_params[num_missing++] = "n_kv_heads";
    config_ok = 0;
}
if (cfg->n_layers == 0) {
    fprintf(stderr, "❌ ERROR: n_layers (number of transformer layers) not found\n");
    missing_params[num_missing++] = "n_layers";
    config_ok = 0;
}
if (cfg->vocab_size == 0) {
    fprintf(stderr, "❌ ERROR: vocab_size not found\n");
    missing_params[num_missing++] = "vocab_size";
    config_ok = 0;
}

if (!config_ok) {
    fprintf(stderr, "\n⚠️  Failed parameter check. Current config:\n");
    fprintf(stderr, "  n_embd=%d, n_ffn=%d\n", cfg->n_embd, cfg->n_ffn);
    fprintf(stderr, "  n_heads=%d, n_kv_heads=%d\n", cfg->n_heads, cfg->n_kv_heads);
    fprintf(stderr, "  n_layers=%d, vocab_size=%d\n", cfg->n_layers, cfg->vocab_size);
    fprintf(stderr, "\n💡 TIP: Check GGUF metadata key names using:\n");
    fprintf(stderr, "  strings qwen35.gguf | grep -E '^(qwen|llama|general)' | head -20\n");
    return -1;
}

fprintf(stderr, "✓ All required parameters loaded:\n");
fprintf(stderr, "  Embedding dim: %d\n", cfg->n_embd);
fprintf(stderr, "  FFN hidden: %d (%.2fx expansion)\n", cfg->n_ffn, (float)cfg->n_ffn / cfg->n_embd);
fprintf(stderr, "  Attention: %d heads → %d KV heads (GQA %.0f:1)\n", 
        cfg->n_heads, cfg->n_kv_heads, (float)cfg->n_heads / cfg->n_kv_heads);
fprintf(stderr, "  Layers: %d\n", cfg->n_layers);
fprintf(stderr, "  Vocab: %d tokens\n", cfg->vocab_size);
fprintf(stderr, "  Context: %d positions\n", cfg->max_seq_len);
fprintf(stderr, "  RoPE base: %.0f\n", cfg->rope_freq_base);

// Calculate approximate total parameters
long long total_params = 
    (long long)cfg->n_embd * cfg->vocab_size +  // token embedding
    (long long)cfg->n_layers * (
        cfg->n_embd * cfg->n_embd +              // attention Q, K, V, O
        cfg->n_embd * cfg->n_ffn * 3             // FFN gate, up, down
    ) +
    cfg->n_embd;  // output norm
total_params = total_params / 1000000;  // Convert to millions

fprintf(stderr, "  Approx parameters: %lld M\n\n", total_params);
```

### Patch 1c: Add tensor validation (insert after line 378)

```c
// ====================================================================
// TENSOR VALIDATION - ENSURES ALL WEIGHTS LOADED
// ====================================================================

fprintf(stderr, "╔══════════════════════════════════════════════════════╗\n");
fprintf(stderr, "║            Tensor Loading Validation                ║\n");
fprintf(stderr, "╚══════════════════════════════════════════════════════╝\n\n");

int tensors_ok = 1;
int missing_tensors = 0;

// Check critical global tensors
fprintf(stderr, "Checking global tensors...\n");
if (!w->token_embd) {
    fprintf(stderr, "  ❌ token_embd.weight not found\n");
    missing_tensors++;
    tensors_ok = 0;
} else {
    fprintf(stderr, "  ✓ token_embd.weight loaded\n");
}

if (!w->output_norm) {
    fprintf(stderr, "  ❌ output_norm.weight not found\n");
    missing_tensors++;
    tensors_ok = 0;
} else {
    fprintf(stderr, "  ✓ output_norm.weight loaded\n");
}

if (!w->output) {
    fprintf(stderr, "  ❌ output.weight (LM head) not found\n");
    missing_tensors++;
    tensors_ok = 0;
} else {
    fprintf(stderr, "  ✓ output.weight (LM head) loaded\n");
}

// Check per-layer tensors
fprintf(stderr, "\nChecking %d layers...\n", cfg->n_layers);
int layers_ok = 1;
for (int l = 0; l < cfg->n_layers; l++) {
    layer_weights_t *lw = &w->layers[l];
    
    int layer_ok = 1;
    if (!lw->attn_norm) {
        if (l == 0) fprintf(stderr, "  Layer %d: ❌ attn_norm\n", l);
        layer_ok = 0;
    }
    if (!lw->attn_q) {
        if (l == 0) fprintf(stderr, "  Layer %d: ❌ attn_q\n", l);
        layer_ok = 0;
    }
    if (!lw->attn_k) {
        if (l == 0) fprintf(stderr, "  Layer %d: ❌ attn_k\n", l);
        layer_ok = 0;
    }
    if (!lw->attn_v) {
        if (l == 0) fprintf(stderr, "  Layer %d: ❌ attn_v\n", l);
        layer_ok = 0;
    }
    if (!lw->attn_output) {
        if (l == 0) fprintf(stderr, "  Layer %d: ❌ attn_output\n", l);
        layer_ok = 0;
    }
    if (!lw->ffn_norm) {
        if (l == 0) fprintf(stderr, "  Layer %d: ❌ ffn_norm\n", l);
        layer_ok = 0;
    }
    if (!lw->ffn_gate) {
        if (l == 0) fprintf(stderr, "  Layer %d: ❌ ffn_gate\n", l);
        layer_ok = 0;
    }
    if (!lw->ffn_up) {
        if (l == 0) fprintf(stderr, "  Layer %d: ❌ ffn_up\n", l);
        layer_ok = 0;
    }
    if (!lw->ffn_down) {
        if (l == 0) fprintf(stderr, "  Layer %d: ❌ ffn_down\n", l);
        layer_ok = 0;
    }
    
    if (!layer_ok) {
        layers_ok = 0;
    }
}

if (layers_ok) {
    fprintf(stderr, "  ✓ All %d layers loaded successfully\n", cfg->n_layers);
}

if (!tensors_ok || !layers_ok) {
    fprintf(stderr, "\n❌ Tensor validation failed!\n");
    fprintf(stderr, "Missing: %d global tensors\n", missing_tensors);
    if (!layers_ok) fprintf(stderr, "Some layers have missing tensors\n");
    return -1;
}

fprintf(stderr, "\n✓ All tensors validated successfully!\n\n");
```

### Patch 1d: Improve vocab_size detection (replace lines 385-399)

```c
if (cfg->vocab_size == 0) {
    fprintf(stderr, "Detecting vocab_size from token_embd.weight...\n");
    for (uint64_t i = 0; i < n_tensors; i++) {
        if (str_eq(tinfos[i].name, "token_embd.weight")) {
            if (tinfos[i].n_dims >= 2) {
                int d0 = (int)tinfos[i].dims[0];
                int d1 = (int)tinfos[i].dims[1];
                // token_embd shape is [vocab_size, embedding_dim]
                cfg->vocab_size = (d0 == cfg->n_embd) ? d1 : d0;
                fprintf(stderr, "  ✓ Detected vocab_size: %d\n", cfg->vocab_size);
                
                // Qwen has very large vocab - validate reasonable range
                if (cfg->vocab_size < 1000) {
                    fprintf(stderr, "  ⚠️  WARNING: vocab_size %d is unusually small (expected >1000)\n", 
                            cfg->vocab_size);
                }
                if (cfg->vocab_size > 250000) {
                    fprintf(stderr, "  ⚠️  WARNING: vocab_size %d is very large (>250k)\n", 
                            cfg->vocab_size);
                }
            }
            break;
        }
    }
}

if (cfg->vocab_size == 0 && m->tok_n_tokens > 0) {
    cfg->vocab_size = (int)m->tok_n_tokens;
    fprintf(stderr, "  ✓ Using tokenizer vocab_size: %d\n", cfg->vocab_size);
}
```

---

## Testing Commands

### Test 1: Inspect Your Qwen 3.5 GGUF File

```bash
# Method A: Using strings (works on any system)
strings your_qwen35.gguf | grep -E "^(qwen|llama|general)" | head -30

# Method B: Using python-gguf (more detailed)
pip install gguf
python3 << 'EOF'
import gguf
reader = gguf.GGUFReader("your_qwen35.gguf")
print("=== METADATA ===")
for key in sorted(reader.fields.keys()):
    if key.startswith(('qwen', 'llama', 'general')):
        print(f"{key}")

print("\n=== FIRST TENSORS ===")
for i, t in enumerate(reader.tensors[:20]):
    print(f"{t.name} {t.tensor.shape}")
EOF
```

### Test 2: Build and Check for Errors

```bash
cd picolm
make clean
make native

# Run with verbose debug output
./picolm /path/to/qwen35.gguf -p "Hello" -n 5 2>&1 | head -150
```

**Expected output**:
```
✓ Found qwen35.embedding_length = 1024
✓ Found qwen35.attention.head_count = 16
...
╔══════════════════════════════════════════════════════╗
║         Configuration Validation (Qwen 3.5)         ║
╚══════════════════════════════════════════════════════╝

✓ All required parameters loaded:
  Embedding dim: 1024
  FFN hidden: 2816 (2.75x expansion)
  ...
```

### Test 3: Full Generation Test

```bash
# Test with simple prompt
echo "The capital of France is" | ./picolm qwen35.gguf -n 20 -t 0.5

# Should output something like:
# The capital of France is Paris. It is the largest city...
```

---

## Debugging Flowchart

```
Model fails to load
    ↓
Check error message
    ├→ "n_embd not loaded"
    │   └→ Run: strings your_qwen35.gguf | grep embedding_length
    │      Then update metadata parsing with correct key
    │
    ├→ "Tensor validation failed"
    │   └→ Check tensor names in GGUF
    │      Update tensor matching in parse_gguf()
    │
    └→ Other error
        └→ Look at debug output, search GitHub issues
```

---

## Quick Reference: What Each Config Parameter Means

```
n_embd (embedding dimension):
  - Size of vector representing each token
  - TinyLlama: 2048, Qwen 3.5-0.8B: 1024
  - Larger = more model capacity, more memory

n_ffn (feed-forward hidden size):
  - Size of hidden layer in FFN blocks
  - Usually 2-4x n_embd
  - Qwen 3.5: 2816 (2.75x n_embd)

n_heads (attention heads):
  - Number of independent attention mechanisms
  - TinyLlama: 32, Qwen 3.5-0.8B: 16
  - Allows parallel attention computations

n_kv_heads (KV heads, for GQA):
  - Number of SHARED key/value heads
  - n_heads:n_kv_heads = 16:2 = 8:1 (Qwen)
  - Lower = more memory efficient

n_layers (transformer layers):
  - Number of stacked transformer blocks
  - Qwen 3.5-0.8B: 24 layers
  - More layers = more parameters but more expressiveness

vocab_size:
  - Number of tokens in vocabulary
  - TinyLlama: 32K, Qwen 3.5: 152K (larger!)
  - Impacts output projection size

max_seq_len (context length):
  - Maximum number of tokens in input/output
  - TinyLlama: 2048, Qwen 3.5-0.8B: 8192
  - Limits how long the conversation can be

rope_freq_base (RoPE theta):
  - Controls positional encoding
  - TinyLlama: 10000, Qwen 3.5: 1000000
  - Higher allows extrapolation beyond training length
```

