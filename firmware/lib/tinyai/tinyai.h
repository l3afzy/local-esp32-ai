// tinyai: a direct-answer language model small enough for an ESP32.
//
//   tai_model m;
//   tai_load(&m, model_blob, model_blob_len);
//   char out[64];
//   tai_ask(&m, "how much does a hummingbird weigh", out, sizeof out);
//   // out = "Between 2 and 20 grams."
//
// Pure C99, no dependencies. Builds for ESP32 (Arduino / ESP-IDF) and for a PC.
#ifndef TINYAI_H
#define TINYAI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TAI_MAGIC 0x31494154u  // "TAI1"
#define TAI_VERSION 3
#define TAI_GROUP 32           // int4 weights: one scale per 32 inputs
#define TAI_MAX_Q 64           // question chars kept, matches train/common.py
#define TAI_MAX_A 48           // answer chars, hard cap
#define TAI_EOS 0
#define TAI_SEP 1

typedef struct {        // int8 matrix, one scale per row (embeddings)
    const int8_t *q;
    const float *s;
    int rows, cols;
} tai_q8;

typedef struct {        // int4 matrix, one scale per row per 32 columns
    const uint8_t *q;   // two weights per byte, low nibble first, stored +8
    const float *s;
    int rows, cols;
} tai_q4;

typedef struct {
    const float *n1, *n2;
    tai_q4 wq, wk, wv, wo, w1, w2;
} tai_layer;

// Optional multi-core hook: run job(ctx, lo, hi) over [0, n), split however
// the platform likes, and return when all of it is done. The firmware sets
// one on dual-core ESP32s; without it everything runs on the calling core.
typedef void (*tai_job)(void *ctx, int lo, int hi);
typedef void (*tai_parallel_fn)(tai_job job, void *ctx, int n);

typedef struct {
    int vocab, ctx, dim, layers, heads, kv_heads, hidden;
    tai_q8 tok, pos;
    tai_layer *layer;
    const float *norm;
    // What the model knows. facts: one canonical question per fact.
    int n_facts;
    const char *facts;
    // Gate index (see train/gate_index.py): a sorted dictionary of content
    // words, and for every accepted phrasing its fact, question type,
    // length and content-word ids.
    int n_words, n_known;
    const char *words;
    const uint32_t *word_off;
    const uint8_t *word_len;
    const uint16_t *known_fact;
    const uint8_t *known_qtype, *known_nwords, *known_len;
    const uint16_t *known_words;
    const char *small_talk;  // text of phrasings with no content words
    // runtime state (heap)
    float *x, *xb, *hb, *q, *k, *v, *att, *logits;
    int8_t *xq;
    int32_t *xsum;       // per-group sums of xq, for the int4 kernel
    float xs;
    int8_t **kc, **vc;   // [layer][ctx * kv_dim]
    float **ks, **vs;    // [layer][ctx * kv_heads]
    tai_parallel_fn parallel;
} tai_model;

// Borrow `blob` (must stay valid and 4-byte aligned). Returns 0 on success.
int tai_load(tai_model *m, const uint8_t *blob, size_t len);
void tai_free(tai_model *m);
// Split matmuls across cores (see tai_parallel_fn). NULL = single core.
void tai_set_parallel(tai_model *m, tai_parallel_fn fn);

// Full pipeline: calculator -> knowledge gate -> model. Always writes a
// short, direct answer to `out`. Returns 0 = model, 1 = calculator,
// 2 = refused (unknown topic).
int tai_ask(tai_model *m, const char *question, char *out, int out_len);

// Model only, no gate: generate from `question` as typed. For experiments.
void tai_generate(tai_model *m, const char *question, char *out, int out_len);

// Building blocks (exposed for tests).
int tai_normalize(const char *in, char *out, int out_len);
int tai_calc(const char *normalized, char *out, int out_len);     // arithmetic + tai_convert
int tai_convert(const char *normalized, char *out, int out_len);  // units, bases, primes
// 0 = unknown topic, else 1 + index of the matched fact.
int tai_gate(const tai_model *m, const char *normalized);
// "<qtype code> <content words>" as the gate parses `normalized` (for tests).
int tai_gate_words(const char *normalized, char *out, int out_len);
// Canonical question of fact `i` (0-based).
const char *tai_fact(const tai_model *m, int i);

#ifdef __cplusplus
}
#endif
#endif
