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
#define TAI_MAX_Q 64           // question chars kept, matches train/common.py
#define TAI_MAX_A 48           // answer chars, hard cap
#define TAI_EOS 0
#define TAI_SEP 1

typedef struct {
    const int8_t *q;  // quantized rows
    const float *s;   // one scale per row
    int rows, cols;
} tai_qmat;

typedef struct {
    const float *n1, *n2;
    tai_qmat wq, wk, wv, wo, w1, w2;
} tai_layer;

typedef struct {
    int vocab, ctx, dim, layers, heads, hidden;
    tai_qmat tok, pos;
    tai_layer *layer;
    const float *norm;
    // gate: the questions the model was trained on, normalized, NUL-separated
    int n_known;
    const char *known;
    // runtime state (heap)
    float *x, *xb, *hb, *q, *k, *v, *att, *logits;
    int8_t *xq;
    float xs;
    int8_t **kc, **vc;   // [layer][ctx * dim]
    float **ks, **vs;    // [layer][ctx * heads]
} tai_model;

// Borrow `blob` (must stay valid and 4-byte aligned). Returns 0 on success.
int tai_load(tai_model *m, const uint8_t *blob, size_t len);
void tai_free(tai_model *m);

// Full pipeline: calculator -> knowledge gate -> model. Always writes a
// short, direct answer to `out`. Returns 0 = model, 1 = calculator,
// 2 = refused (unknown topic).
int tai_ask(tai_model *m, const char *question, char *out, int out_len);

// Model only, no gate. Useful for experiments.
void tai_generate(tai_model *m, const char *question, char *out, int out_len);

// Building blocks (exposed for tests).
int tai_normalize(const char *in, char *out, int out_len);
int tai_calc(const char *normalized, char *out, int out_len);
// 0 = unknown topic, else 1 + index of the matched known question.
int tai_gate(const tai_model *m, const char *normalized);

#ifdef __cplusplus
}
#endif
#endif
