#include "model.h"
#include "tensor.h"
#include "quant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

/* ---- GGUF metadata value types ---- */
enum {
    GGUF_META_UINT8   = 0,
    GGUF_META_INT8    = 1,
    GGUF_META_UINT16  = 2,
    GGUF_META_INT16   = 3,
    GGUF_META_UINT32  = 4,
    GGUF_META_INT32   = 5,
    GGUF_META_FLOAT32 = 6,
    GGUF_META_BOOL    = 7,
    GGUF_META_STRING  = 8,
    GGUF_META_ARRAY   = 9,
    GGUF_META_UINT64  = 10,
    GGUF_META_INT64   = 11,
    GGUF_META_FLOAT64 = 12,
};

/* ---- Helpers for reading GGUF binary format ---- */

typedef struct {
    const uint8_t *data;
    size_t pos;
    size_t size;
} reader_t;

static uint8_t read_u8(reader_t *r) {
    uint8_t v = r->data[r->pos];
    r->pos += 1;
    return v;
}

static uint16_t read_u16(reader_t *r) {
    uint16_t v;
    memcpy(&v, r->data + r->pos, 2);
    r->pos += 2;
    return v;
}

static uint32_t read_u32(reader_t *r) {
    uint32_t v;
    memcpy(&v, r->data + r->pos, 4);
    r->pos += 4;
    return v;
}

static int32_t read_i32(reader_t *r) {
    int32_t v;
    memcpy(&v, r->data + r->pos, 4);
    r->pos += 4;
    return v;
}

static uint64_t read_u64(reader_t *r) {
    uint64_t v;
    memcpy(&v, r->data + r->pos, 8);
    r->pos += 8;
    return v;
}

static float read_f32(reader_t *r) {
    float v;
    memcpy(&v, r->data + r->pos, 4);
    r->pos += 4;
    return v;
}

typedef struct { const char *str; uint64_t len; } gguf_str_t;

static gguf_str_t read_gguf_string(reader_t *r) {
    gguf_str_t s;
    s.len = read_u64(r);
    s.str = (const char *)(r->data + r->pos);
    r->pos += s.len;
    return s;
}

static int str_eq(gguf_str_t s, const char *lit) {
    size_t n = strlen(lit);
    return s.len == n && memcmp(s.str, lit, n) == 0;
}

static uint64_t skip_meta_value(reader_t *r, uint32_t vtype, int *is_numeric) {
    *is_numeric = 1;
    switch (vtype) {
        case GGUF_META_UINT8:   return read_u8(r);
        case GGUF_META_INT8:    return (uint64_t)(int64_t)(int8_t)read_u8(r);
        case GGUF_META_UINT16:  return read_u16(r);
        case GGUF_META_INT16:   return (uint64_t)(int64_t)(int16_t)read_u16(r);
        case GGUF_META_UINT32:  return read_u32(r);
        case GGUF_META_INT32:   return (uint64_t)(int64_t)read_i32(r);
        case GGUF_META_UINT64:  return read_u64(r);
        case GGUF_META_INT64:   return read_u64(r);
        case GGUF_META_FLOAT32: { read_f32(r); *is_numeric = 0; return 0; }
        case GGUF_META_FLOAT64: { r->pos += 8; *is_numeric = 0; return 0; }
        case GGUF_META_BOOL:    return read_u8(r);
        case GGUF_META_STRING:  { read_gguf_string(r); *is_numeric = 0; return 0; }
        case GGUF_META_ARRAY: {
            *is_numeric = 0;
            uint32_t arr_type = read_u32(r);
            uint64_t arr_len  = read_u64(r);
            int dummy;
            for (uint64_t i = 0; i < arr_len; i++) {
                skip_meta_value(r, arr_type, &dummy);
            }
            return 0;
        }
        default:
            fprintf(stderr, "Unknown GGUF metadata type: %u\n", vtype);
            exit(1);
    }
}

/* ---- mmap abstraction ---- */

static int mmap_file(model_t *m, const char *path) {
#ifdef _WIN32
    HANDLE fh = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (fh == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "Cannot open file: %s\n", path);
        return -1;
    }

    LARGE_INTEGER fsize;
    GetFileSizeEx(fh, &fsize);
    m->mmap_size = (size_t)fsize.QuadPart;

    HANDLE mh = CreateFileMappingA(fh, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mh) {
        fprintf(stderr, "CreateFileMapping failed\n");
        CloseHandle(fh);
        return -1;
    }

    void *addr = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (!addr) {
        fprintf(stderr, "MapViewOfFile failed\n");
        CloseHandle(mh);
        CloseHandle(fh);
        return -1;
    }

    m->mmap_addr  = addr;
    m->file_handle = fh;
    m->map_handle  = mh;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Cannot open file: %s\n", path);
        return -1;
    }

    struct stat st;
    fstat(fd, &st);
    m->mmap_size = (size_t)st.st_size;

    void *addr = mmap(NULL, m->mmap_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
        fprintf(stderr, "mmap failed\n");
        close(fd);
        return -1;
    }
    madvise(addr, m->mmap_size, MADV_SEQUENTIAL);

    m->mmap_addr = addr;
    m->fd = fd;
#endif
    return 0;
}

static void munmap_file(model_t *m) {
    if (!m->mmap_addr) return;
#ifdef _WIN32
    UnmapViewOfFile(m->mmap_addr);
    CloseHandle(m->map_handle);
    CloseHandle(m->file_handle);
#else
    munmap(m->mmap_addr, m->mmap_size);
    close(m->fd);
#endif
    m->mmap_addr = NULL;
}

/* ---- GGUF Parser ---- */

static int parse_gguf(model_t *m, int max_seq_len) {
    reader_t r = { .data = (const uint8_t *)m->mmap_addr, .pos = 0, .size = m->mmap_size };
    model_config_t *cfg = &m->config;

    uint32_t magic = read_u32(&r);
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "Invalid GGUF magic: 0x%08X\n", magic);
        return -1;
    }

    uint32_t version = read_u32(&r);
    if (version < 2 || version > 3) {
        fprintf(stderr, "Unsupported GGUF version: %u\n", version);
        return -1;
    }

    uint64_t n_tensors  = read_u64(&r);
    uint64_t n_metadata = read_u64(&r);

    cfg->alignment = 32;
    cfg->rope_freq_base = 10000.0f;
    cfg->max_seq_len = 2048;
    cfg->weight_type = GGUF_TYPE_F16;
    m->tok_bos_id = 1;
    m->tok_eos_id = 2;
    // Read the metadata (n_metadata = kv count)
    for (uint64_t i = 0; i < n_metadata; i++) {
        gguf_str_t key = read_gguf_string(&r);
        uint32_t vtype = read_u32(&r);

        if (str_eq(key, "qwen35.embedding_length") || str_eq(key, "general.embedding_length")) {
            int dummy; cfg->n_embd = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "qwen35.feed_forward_length") || str_eq(key, "general.feed_forward_length")) {
            int dummy; cfg->n_ffn = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "qwen35.attention.head_count")) {
            int dummy; cfg->n_heads = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "qwen35.attention.head_count_kv")) {
            int dummy; cfg->n_kv_heads = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "qwen35.block_count")) {
            int dummy; cfg->n_layers = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "qwen35.rope.freq_base")) {
            if (vtype == GGUF_META_FLOAT32) {
                cfg->rope_freq_base = read_f32(&r);
            } else {
                int dummy; skip_meta_value(&r, vtype, &dummy);
            }
        } else if (str_eq(key, "general.alignment")) {
            int dummy; cfg->alignment = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "qwen35.vocab_size")) {
            int dummy; cfg->vocab_size = (int)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "tokenizer.ggml.bos_token_id")) {
            int dummy; m->tok_bos_id = (uint32_t)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "tokenizer.ggml.eos_token_id")) {
            int dummy; m->tok_eos_id = (uint32_t)skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "tokenizer.ggml.tokens")) {
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
        } else if (str_eq(key, "qwen35.ssm.conv_kernel")) {
            int dummy; /* If your model_config_t gets updated, store here */ skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "qwen35.ssm.state_size")) {
            int dummy; skip_meta_value(&r, vtype, &dummy);
        } else if (str_eq(key, "qwen35.ssm.inner_size")) {
            int dummy; skip_meta_value(&r, vtype, &dummy);
        } else {
            int dummy; skip_meta_value(&r, vtype, &dummy);
        }
    }

    if (max_seq_len > 0 && max_seq_len < cfg->max_seq_len) {
        cfg->max_seq_len = max_seq_len;
    }
    cfg->head_dim = cfg->n_embd / cfg->n_heads;

    /* Parse tensor info entries */
    typedef struct {
        gguf_str_t name;
        uint32_t   n_dims;
        uint64_t   dims[4];
        uint32_t   type;
        uint64_t   offset;
    } tensor_info_t;

    tensor_info_t *tinfos = (tensor_info_t *)malloc(n_tensors * sizeof(tensor_info_t));
    if (!tinfos) { fprintf(stderr, "OOM allocating tensor info\n"); return -1; }

    for (uint64_t i = 0; i < n_tensors; i++) {
        tinfos[i].name   = read_gguf_string(&r);
        tinfos[i].n_dims = read_u32(&r);
        for (uint32_t d = 0; d < tinfos[i].n_dims; d++) {
            tinfos[i].dims[d] = read_u64(&r);
        }
        tinfos[i].type   = read_u32(&r);
        tinfos[i].offset = read_u64(&r);
    }

    size_t alignment = (size_t)cfg->alignment;
    size_t tensor_data_base = (r.pos + alignment - 1) & ~(alignment - 1);

    model_weights_t *w = &m->weights;
    memset(w, 0, sizeof(*w));

    for (uint64_t i = 0; i < n_tensors; i++) {
        const void *ptr = (const uint8_t *)m->mmap_addr + tensor_data_base + tinfos[i].offset;
        gguf_type_t qtype = (gguf_type_t)tinfos[i].type;

        if (str_eq(tinfos[i].name, "token_embd.weight")) {
            w->token_embd = ptr; w->type_token_embd = qtype;
        } else if (str_eq(tinfos[i].name, "output_norm.weight")) {
            w->output_norm = ptr; w->type_output_norm = qtype;
        } else if (str_eq(tinfos[i].name, "output.weight")) {
            w->output = ptr; w->type_output = qtype;
        } else {
            int layer = -1;
            char suffix[64] = {0};

            if (tinfos[i].name.len > 4 && memcmp(tinfos[i].name.str, "blk.", 4) == 0) {
                const char *p = tinfos[i].name.str + 4;
                const char *end = tinfos[i].name.str + tinfos[i].name.len;
                layer = 0;
                while (p < end && *p >= '0' && *p <= '9') {
                    layer = layer * 10 + (*p - '0');
                    p++;
                }
                if (p < end && *p == '.') {
                    p++;
                    size_t slen = (size_t)(end - p);
                    if (slen < sizeof(suffix)) {
                        memcpy(suffix, p, slen);
                        suffix[slen] = '\0';
                    }
                }
            }

            if (layer >= 0 && layer < MAX_LAYERS) {
                layer_weights_t *lw = &w->layers[layer];
                
                // ---- Attention Mechanism ----
                if (strcmp(suffix, "attn_norm.weight") == 0) {
                    lw->attn_norm = ptr; lw->type_attn_norm = qtype;
                } else if (strcmp(suffix, "attn_qkv.weight") == 0) {
                    lw->attn_qkv = ptr; lw->type_attn_qkv = qtype;  /* Combined QKV */
                } else if (strcmp(suffix, "attn_gate.weight") == 0) {
                    lw->attn_gate = ptr; lw->type_attn_gate = qtype;
                    
                // ---- SSM / Mamba Mechanism ----
                } else if (strcmp(suffix, "ssm_norm.weight") == 0) {
                    lw->ssm_norm = ptr; lw->type_ssm_norm = qtype;
                } else if (strcmp(suffix, "ssm_a") == 0) {
                    lw->ssm_a = ptr; lw->type_ssm_a = qtype;
                } else if (strcmp(suffix, "ssm_alpha.weight") == 0) {
                    lw->ssm_alpha = ptr; lw->type_ssm_alpha = qtype;
                } else if (strcmp(suffix, "ssm_beta.weight") == 0) {
                    lw->ssm_beta = ptr; lw->type_ssm_beta = qtype;
                } else if (strcmp(suffix, "ssm_conv1d.weight") == 0) {
                    lw->ssm_conv1d = ptr; lw->type_ssm_conv1d = qtype;
                } else if (strcmp(suffix, "ssm_dt.bias") == 0) {
                    lw->ssm_dt = ptr; lw->type_ssm_dt = qtype;
                } else if (strcmp(suffix, "ssm_out.weight") == 0) {
                    lw->ssm_out = ptr; lw->type_ssm_out = qtype;
                    
                // ---- Feed-Forward Network & Normalization ----
                // } else if (strcmp(suffix, "ffn_norm.weight") == 0) {
                //     lw->ffn_norm = ptr; lw->type_ffn_norm = qtype;
                } else if (strcmp(suffix, "ffn_gate.weight") == 0) {
                    lw->ffn_gate = ptr; lw->type_ffn_gate = qtype;
                } else if (strcmp(suffix, "ffn_down.weight") == 0) {
                    lw->ffn_down = ptr; lw->type_ffn_down = qtype;
                } else if (strcmp(suffix, "ffn_up.weight") == 0) {
                    lw->ffn_up = ptr; lw->type_ffn_up = qtype;
                } else if (strcmp(suffix, "post_attention_norm.weight") == 0) {
                    lw->post_attention_norm = ptr; lw->type_post_attention_norm = qtype;
                }
            }
        }
    }

    if (!w->output) {
        w->output = w->token_embd;
        w->type_output = w->type_token_embd;
    }

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
    if (cfg->vocab_size == 0 && m->tok_n_tokens > 0) {
        cfg->vocab_size = (int)m->tok_n_tokens;
    }

    // FIX: Change type_attn_q to type_attn_qkv
    cfg->weight_type = w->layers[0].type_attn_qkv;

    fprintf(stderr, "Model config:\n");
    fprintf(stderr, "  n_embd=%d, n_ffn=%d, n_heads=%d, n_kv_heads=%d\n",
            cfg->n_embd, cfg->n_ffn, cfg->n_heads, cfg->n_kv_heads);
    fprintf(stderr, "  n_layers=%d, vocab_size=%d, max_seq=%d\n",
            cfg->n_layers, cfg->vocab_size, cfg->max_seq_len);
    fprintf(stderr, "  head_dim=%d, rope_base=%.1f\n", cfg->head_dim, cfg->rope_freq_base);

    free(tinfos);
    return 0;
}

/* ---- Pre-compute RoPE cos/sin lookup tables ---- */

static void init_rope_tables(run_state_t *s, const model_config_t *c) {
    int half_dim = c->head_dim / 2;
    for (int pos = 0; pos < c->max_seq_len; pos++) {
        float *cos_row = s->rope_cos + (size_t)pos * half_dim;
        float *sin_row = s->rope_sin + (size_t)pos * half_dim;
        for (int i = 0; i < half_dim; i++) {
            float theta = (float)pos / powf(c->rope_freq_base, (float)(2 * i) / (float)c->head_dim);
            cos_row[i] = cosf(theta);
            sin_row[i] = sinf(theta);
        }
    }
}

/* ---- Buffer allocation ---- */

static int allocate_run_state(model_t *m) {
    model_config_t *c = &m->config;
    run_state_t *s = &m->state;

    int kv_dim = c->n_kv_heads * c->head_dim;
    int half_dim = c->head_dim / 2;

    /* Calculate sizes for float buffers */
    size_t sz_x      = (size_t)c->n_embd * sizeof(float);
    size_t sz_xb     = (size_t)c->n_embd * sizeof(float);
    size_t sz_xb2    = (size_t)c->n_embd * sizeof(float);
    size_t sz_q      = (size_t)c->n_embd * sizeof(float);
    /* att buffer removed (flash attention) */
    size_t sz_hb     = (size_t)c->n_ffn * sizeof(float);
    size_t sz_hb2    = (size_t)c->n_ffn * sizeof(float);
    size_t sz_logits = (size_t)c->vocab_size * sizeof(float);

    int scratch_dim = c->n_embd > c->n_ffn ? c->n_embd : c->n_ffn;
    
    if (c->vocab_size > scratch_dim) scratch_dim = c->vocab_size;
    size_t sz_scratch = (size_t)scratch_dim * sizeof(float);

    /* RoPE tables: cos and sin for each (position, dim_pair) */
    size_t sz_rope = (size_t)c->max_seq_len * half_dim * sizeof(float) * 2;

    /* Norm weights: (n_layers * 2 + 1) * n_embd floats */
    size_t n_norm = (size_t)(c->n_layers * 2 + 1) * c->n_embd;
    size_t sz_norm = n_norm * sizeof(float);

    int ssm_inner_dim = c->n_embd * 2; 
    int ssm_state_size = 16;
    int ssm_conv_kernel = 4;

    size_t sz_ssm_state = (size_t)c->n_layers * ssm_inner_dim * ssm_state_size * sizeof(float);
    size_t sz_ssm_conv  = (size_t)c->n_layers * ssm_inner_dim * ssm_conv_kernel * sizeof(float);

    // Add these sizes to your total calculation
    size_t total = sz_x + sz_xb + sz_xb2 + sz_q +
                   sz_hb + sz_hb2 + sz_logits +
                   sz_scratch + sz_rope + sz_norm + 
                   sz_ssm_state + sz_ssm_conv; // <-- Added

    // size_t total = sz_x + sz_xb + sz_xb2 + sz_q +
    //                sz_hb + sz_hb2 + sz_logits +
    //                sz_scratch + sz_rope + sz_norm;

    /* FP16 KV cache: separate allocation */
    size_t kv_elements = (size_t)c->n_layers * c->max_seq_len * kv_dim;
    size_t sz_kv = kv_elements * sizeof(uint16_t) * 2; /* key + val */

    fprintf(stderr, "Allocating %.2f MB for runtime state (+ %.2f MB FP16 KV cache)\n",
            (double)total / (1024.0 * 1024.0),
            (double)sz_kv / (1024.0 * 1024.0));

    s->mem_block = calloc(1, total);
    if (!s->mem_block) {
        fprintf(stderr, "OOM: cannot allocate %zu bytes\n", total);
        return -1;
    }
    s->mem_size = total + sz_kv;

    /* Allocate FP16 KV cache separately */
    s->kv_block = calloc(1, sz_kv);
    if (!s->kv_block) {
        fprintf(stderr, "OOM: cannot allocate %zu bytes for KV cache\n", sz_kv);
        free(s->mem_block);
        return -1;
    }
    s->kv_size = sz_kv;
    /* Carve float pointers */
    float *p = (float *)s->mem_block;
    s->x      = p; p += c->n_embd;
    s->xb     = p; p += c->n_embd;
    s->xb2    = p; p += c->n_embd;
    s->q      = p; p += c->n_embd;
    s->hb     = p; p += c->n_ffn;
    s->hb2    = p; p += c->n_ffn;
    s->logits = p; p += c->vocab_size;
    s->dequant_scratch = p; p += scratch_dim;
    
    /* RoPE tables */
    s->rope_cos = p; p += (size_t)c->max_seq_len * half_dim;
    s->rope_sin = p; p += (size_t)c->max_seq_len * half_dim;
    
    /* Norm weights */
    s->norm_weights = p;
    
    /* FP16 KV cache pointers */
    uint16_t *kp = (uint16_t *)s->kv_block;
    s->key_cache = kp; kp += kv_elements;
    s->val_cache = kp;
     printf("Checkpoint0\n");
    /* Pre-dequantize norm weights */
    float *nw = s->norm_weights;
    for (int l = 0; l < c->n_layers; l++) {
        printf("Dequantizing norm weights for layer %d\n", l);
        s->attn_norm_w[l] = nw;
        dequantize_row(m->weights.layers[l].attn_norm, nw, c->n_embd,
            m->weights.layers[l].type_attn_norm);
        printf("Dequantizing post_attention_norm weights for layer %d\n", l);
        nw += c->n_embd;
        printf("Dequantizing ssm_norm weights for layer %d\n", l); 
        s->ssm_norm_w[l] = nw;
        dequantize_row(m->weights.layers[l].ssm_norm, nw, c->n_embd,
            m->weights.layers[l].type_ssm_norm);
        nw += c->n_embd;
        // printf("Dequantizing ffn_norm weights for layer %d\n", l); 
        // s->ffn_norm_w[l] = nw;
        // dequantize_row(m->weights.layers[l].ffn_norm, nw, c->n_embd,
        //     m->weights.layers[l].type_ffn_norm);
        nw += c->n_embd;
    }
     printf("Checkpoint1\n");
    s->output_norm_w = nw;
    dequantize_row(m->weights.output_norm, nw, c->n_embd, m->weights.type_output_norm);
    nw += c->n_embd;
     
    /* Allocate the persistent blocks from the tail end of the pointer block */
    s->ssm_state = (float*)nw; 
    nw += (c->n_layers * ssm_inner_dim * ssm_state_size);
    s->conv_state = (float*)nw;
    nw += (c->n_layers * ssm_inner_dim * ssm_conv_kernel);
     
    /* Initialize SSM state buffers to zero */
    memset(s->ssm_state, 0, (size_t)c->n_layers * ssm_inner_dim * ssm_state_size * sizeof(float));
    memset(s->conv_state, 0, (size_t)c->n_layers * ssm_inner_dim * ssm_conv_kernel * sizeof(float));
     
    /* Init tensor scratch */
    tensor_init_scratch(s->dequant_scratch, scratch_dim);

    /* Pre-compute RoPE tables (eliminates powf/cosf/sinf from hot path) */
    init_rope_tables(s, c);
    fprintf(stderr, "Pre-computed RoPE tables for %d positions and %d dim pairs\n",
            c->max_seq_len, half_dim);

    return 0;
}

/* ---- Public API ---- */

int model_load(model_t *m, const char *path, int max_seq_len) {
    memset(m, 0, sizeof(*m));

    if (mmap_file(m, path) != 0) return -1;
    if (parse_gguf(m, max_seq_len) != 0) return -1;
    if (allocate_run_state(m) != 0) return -1;

    return 0;
}

/* ================================================================
 * Forward pass with:
 *   - FP16 KV cache (halves memory bandwidth in attention)
 *   - Flash attention / online softmax (single pass, no score buffer)
 *   - Pre-computed RoPE tables (table lookup instead of trig)
 * ================================================================ */

// 

float *model_forward(model_t *m, int token, int pos) {
    model_config_t *c = &m->config;
    model_weights_t *w = &m->weights;
    run_state_t *s = &m->state;

    int dim    = c->n_embd;
    int n_ffn  = c->n_ffn;
    int n_heads = c->n_heads;
    int n_kv_heads = c->n_kv_heads;
    int head_dim = c->head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int kv_mul = n_heads / n_kv_heads;
    int seq_len = c->max_seq_len;
    int half_dim = head_dim / 2;

    /* RoPE table pointers for this position */
    const float *cos_pos = s->rope_cos + (size_t)pos * half_dim;
    const float *sin_pos = s->rope_sin + (size_t)pos * half_dim;

    /* 1. Embedding lookup */
    {
        size_t row_bytes = gguf_type_row_size(w->type_token_embd, dim);
        const void *embd_row = (const uint8_t *)w->token_embd + (size_t)token * row_bytes;
        dequantize_row(embd_row, s->x, dim, w->type_token_embd);
    }

    /* 2. Transformer layers */
    for (int l = 0; l < c->n_layers; l++) {
        layer_weights_t *lw = &w->layers[l];

        /* ---- Attention ---- */
        rmsnorm(s->xb, s->x, s->attn_norm_w[l], dim);

        /* * FIX: Qwen3.5 uses a combined QKV weight block.
         * The total output dimension is: dim (for Q) + kv_dim (for K) + kv_dim (for V)
         */
        int total_qkv_dim = dim + (2 * kv_dim);
        
        // Compute all projections directly into our dequant_scratch buffer
        matmul(s->dequant_scratch, s->xb, lw->attn_qkv, dim, total_qkv_dim, lw->type_attn_qkv);

        // Slice out temporary floating point pointers from our scratch space layout
        float *q_src = s->dequant_scratch;
        float *k_src = s->dequant_scratch + dim;
        float *v_src = s->dequant_scratch + dim + kv_dim;

        // Copy Q directly to its runtime location
        memcpy(s->q, q_src, dim * sizeof(float));

        /* Key and Value Cache Calculations */
        uint16_t *kcache_layer = s->key_cache + (size_t)l * seq_len * kv_dim;
        uint16_t *vcache_layer = s->val_cache + (size_t)l * seq_len * kv_dim;
        
        uint16_t *key_pos_fp16 = kcache_layer + (size_t)pos * kv_dim;
        uint16_t *val_pos_fp16 = vcache_layer + (size_t)pos * kv_dim;

        /* Apply RoPE to Q and our sliced K buffer */
        rope(s->q, k_src, head_dim, n_heads, n_kv_heads, cos_pos, sin_pos);

        /* Quantize and compress K & V slices to FP16 Cache Blocks */
        for (int d = 0; d < kv_dim; d++) {
            key_pos_fp16[d] = fp32_to_fp16(k_src[d]);
            val_pos_fp16[d] = fp32_to_fp16(v_src[d]);
        }

        /* ---- Flash Attention (online softmax) ---- */
        for (int h = 0; h < n_heads; h++) {
            float *qh = s->q + h * head_dim;
            int kv_h = h / kv_mul;
            float *xbh = s->xb + h * head_dim;

            float max_score = -1e30f;
            float sum_exp = 0.0f;
            float acc[256]; 
            memset(acc, 0, (size_t)head_dim * sizeof(float));

            for (int t = 0; t <= pos; t++) {
                const uint16_t *kt = kcache_layer + (size_t)t * kv_dim + kv_h * head_dim;
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += qh[d] * fp16_to_fp32(kt[d]);
                }
                score /= sqrtf((float)head_dim);

                const uint16_t *vt = vcache_layer + (size_t)t * kv_dim + kv_h * head_dim;

                if (score > max_score) {
                    float correction = expf(max_score - score);
                    sum_exp = sum_exp * correction + 1.0f;
                    for (int d = 0; d < head_dim; d++) {
                        acc[d] = acc[d] * correction + fp16_to_fp32(vt[d]);
                    }
                    max_score = score;
                } else {
                    float w = expf(score - max_score);
                    sum_exp += w;
                    for (int d = 0; d < head_dim; d++) {
                        acc[d] += w * fp16_to_fp32(vt[d]);
                    }
                }
            }

            float inv_sum = 1.0f / sum_exp;
            for (int d = 0; d < head_dim; d++) {
                xbh[d] = acc[d] * inv_sum;
            }
        }

        /* Attention Output Projection Step */
        // Note: Qwen models use attn_gate instead of traditional output projections 
        /* Attention Output Projection Step */
        matmul(s->xb2, s->xb, lw->attn_gate, dim, dim, lw->type_attn_gate);
        vec_add(s->x, s->xb2, dim);

        // ================================================================
        // CHÈN THÊM: KHỐI TÍNH TOÁN SSM / MAMBA PIPELINE TẠI ĐÂY
        // ================================================================
        if (lw->ssm_norm != NULL) {
            int ssm_inner_dim = dim * 2;
            int ssm_state_size = 16;
            int ssm_conv_kernel = 4;
            
            // 1. Chuẩn hóa RMSNorm trước khi vào SSM
            rmsnorm(s->xb, s->x, s->ssm_norm_w[l], dim);
            
            // 2. Dự chiếu trạng thái (Sử dụng tạm vùng nhớ hb và hb2 làm scratchpad)
            // Dự chiếu ssm_alpha và ssm_beta từ xb
            matmul(s->hb,  s->xb, lw->ssm_alpha, dim, ssm_inner_dim, lw->type_ssm_alpha);
            matmul(s->hb2, s->xb, lw->ssm_beta,  dim, ssm_inner_dim, lw->type_ssm_beta);
            
            // 3. Cập nhật trạng thái Convolution 1D & Tích lũy State Space
            // Khối này dịch chuyển cửa sổ conv_state và nhân với trọng số ssm_conv1d
            // Do ssm_a lưu ở dạng F32, ta tính toán trực tiếp không qua dequantize
            float *layer_ssm_state = s->ssm_state + (size_t)l * ssm_inner_dim * ssm_state_size;
            float *layer_conv_state = s->conv_state + (size_t)l * ssm_inner_dim * ssm_conv_kernel;
            
            // Gọi hàm xử lý lõi toán học SSM (hoặc viết vòng lặp tích lũy trạng thái tuyến tính)
            // Ở mức đơn giản nhất để test mạch dữ liệu, ta cập nhật trạng thái lịch sử:
            for (int i = 0; i < ssm_inner_dim; i++) {
                float dt = s->hb[i] + ((float*)lw->ssm_dt)[i % 16]; // dt_bias cũng là F32
                float a_val = ((float*)lw->ssm_a)[i];                  // ssm_a là F32
                
                // Công thức tính toán dịch chuyển trạng thái Mamba: X_t = A * X_{t-1} + B * U_t
                layer_ssm_state[i * ssm_state_size] = layer_ssm_state[i * ssm_state_size] * expf(dt * a_val) + dt * s->hb2[i];
            }
            
            // 4. Nhân với ma trận đầu ra ssm_out đưa về kích thước gốc (dim)
            matmul(s->xb2, s->hb, lw->ssm_out, ssm_inner_dim, dim, lw->type_ssm_out);
            
            // 5. Cộng trực tiếp vào đường truyền tắt (Residual Connection)
            vec_add(s->x, s->xb2, dim);
        }
        // ================================================================

        /* ---- FFN (SwiGLU) ---- */
        rmsnorm(s->xb, s->x, s->ffn_norm_w[l], dim);

        matmul(s->hb,  s->xb, lw->ffn_gate, dim, n_ffn, lw->type_ffn_gate);
        matmul(s->hb2, s->xb, lw->ffn_up,   dim, n_ffn, lw->type_ffn_up);

        silu(s->hb, n_ffn);
        elemwise_mul(s->hb, s->hb, s->hb2, n_ffn);

        matmul(s->xb, s->hb, lw->ffn_down, n_ffn, dim, lw->type_ffn_down);
        vec_add(s->x, s->xb, dim);
    }

    /* 3. Final RMSNorm */
    rmsnorm(s->x, s->x, s->output_norm_w, dim);

    /* 4. Output projection -> logits */
    matmul(s->logits, s->x, w->output, dim, c->vocab_size, w->type_output);

    return s->logits;
}

void model_free(model_t *m) {
    if (m->state.mem_block) {
        free(m->state.mem_block);
        m->state.mem_block = NULL;
    }
    if (m->state.kv_block) {
        free(m->state.kv_block);
        m->state.kv_block = NULL;
    }
    munmap_file(m);
}

/* ================================================================
 * KV Cache Persistence — save/load KV state to skip prompt prefill
 *
 * File format:
 *   [4 bytes] magic: KVCACHE_MAGIC
 *   [4 bytes] n_pos: number of cached positions
 *   [4 bytes] n_layers
 *   [4 bytes] kv_dim (n_kv_heads * head_dim)
 *   [N bytes] key_cache FP16 data (n_layers * n_pos * kv_dim * sizeof(uint16_t))
 *   [N bytes] val_cache FP16 data (same size)
 * ================================================================ */

int kvcache_save(const model_t *m, const char *path, int n_pos) {
    const model_config_t *c = &m->config;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int seq_len = c->max_seq_len;

    if (n_pos <= 0 || n_pos > seq_len) return -1;

    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "kvcache_save: cannot open %s\n", path);
        return -1;
    }

    uint32_t header[4] = {
        KVCACHE_MAGIC,
        (uint32_t)n_pos,
        (uint32_t)c->n_layers,
        (uint32_t)kv_dim
    };
    fwrite(header, sizeof(uint32_t), 4, f);

    /* Write KV cache for each layer, only the first n_pos positions */
    size_t row_size = (size_t)kv_dim * sizeof(uint16_t);
    for (int l = 0; l < c->n_layers; l++) {
        const uint16_t *kcache_l = m->state.key_cache + (size_t)l * seq_len * kv_dim;
        for (int p = 0; p < n_pos; p++) {
            fwrite(kcache_l + (size_t)p * kv_dim, 1, row_size, f);
        }
    }
    for (int l = 0; l < c->n_layers; l++) {
        const uint16_t *vcache_l = m->state.val_cache + (size_t)l * seq_len * kv_dim;
        for (int p = 0; p < n_pos; p++) {
            fwrite(vcache_l + (size_t)p * kv_dim, 1, row_size, f);
        }
    }

    fclose(f);
    fprintf(stderr, "KV cache saved: %d positions to %s\n", n_pos, path);
    return 0;
}

int kvcache_load(model_t *m, const char *path) {
    const model_config_t *c = &m->config;
    int kv_dim = c->n_kv_heads * c->head_dim;
    int seq_len = c->max_seq_len;

    FILE *f = fopen(path, "rb");
    if (!f) return 0;

    uint32_t header[4];
    if (fread(header, sizeof(uint32_t), 4, f) != 4) {
        fclose(f);
        return 0;
    }

    if (header[0] != KVCACHE_MAGIC) {
        fprintf(stderr, "kvcache_load: invalid magic\n");
        fclose(f);
        return 0;
    }

    int n_pos = (int)header[1];
    int file_layers = (int)header[2];
    int file_kv_dim = (int)header[3];

    if (file_layers != c->n_layers || file_kv_dim != kv_dim) {
        fprintf(stderr, "kvcache_load: model mismatch (layers=%d/%d, kv_dim=%d/%d)\n",
                file_layers, c->n_layers, file_kv_dim, kv_dim);
        fclose(f);
        return 0;
    }
    if (n_pos > seq_len) {
        fprintf(stderr, "kvcache_load: cached %d positions exceeds max_seq_len %d\n",
                n_pos, seq_len);
        fclose(f);
        return 0;
    }

    size_t row_size = (size_t)kv_dim * sizeof(uint16_t);
    for (int l = 0; l < c->n_layers; l++) {
        uint16_t *kcache_l = m->state.key_cache + (size_t)l * seq_len * kv_dim;
        for (int p = 0; p < n_pos; p++) {
            if (fread(kcache_l + (size_t)p * kv_dim, 1, row_size, f) != row_size) {
                fclose(f);
                return 0;
            }
        }
    }
    for (int l = 0; l < c->n_layers; l++) {
        uint16_t *vcache_l = m->state.val_cache + (size_t)l * seq_len * kv_dim;
        for (int p = 0; p < n_pos; p++) {
            if (fread(vcache_l + (size_t)p * kv_dim, 1, row_size, f) != row_size) {
                fclose(f);
                return 0;
            }
        }
    }

    fclose(f);
    fprintf(stderr, "KV cache loaded: %d positions from %s\n", n_pos, path);
    return n_pos;
}
