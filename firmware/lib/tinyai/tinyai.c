// Inference for the TinyGPT in train/model.py.
//
// Weights are int4 (groups of 32 share a scale), activations are quantized
// to int8 on the fly, and every matmul is an integer dot product. On an
// ESP32 the weights are read straight from memory-mapped flash; flash
// bandwidth is the bottleneck, so fewer bits per weight means faster tokens.
#include "tinyai.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------- text

static int prompt_char(char c) {
    return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || strchr(" '+-*/.,%^()=:", c);
}

// Same rules as normalize() in train/common.py.
int tai_normalize(const char *in, char *out, int out_len) {
    char tmp[512];
    int n = 0, prev_space = 1;
    for (; *in && n < (int)sizeof tmp - 1; in++) {
        char c = *in;
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
        if (!prompt_char(c)) c = ' ';
        if (c == ' ') {
            if (prev_space) continue;
            prev_space = 1;
        } else {
            prev_space = 0;
        }
        tmp[n++] = c;
    }
    while (n > 0 && strchr(" ?!.,", tmp[n - 1])) n--;
    int start = n > TAI_MAX_Q ? n - TAI_MAX_Q : 0;
    while (start < n && tmp[start] == ' ') start++;
    int len = n - start;
    if (len > out_len - 1) len = out_len - 1;
    memcpy(out, tmp + start, (size_t)len);
    out[len] = 0;
    return len;
}

static int encode_char(char c) { return (c >= 32 && c <= 126) ? c - 30 : 2; }

const char *tai_fact(const tai_model *m, int i) {
    const char *f = m->facts;
    while (i-- > 0) f += strlen(f) + 1;
    return f;
}

// ---------------------------------------------------------------- loading

typedef struct {
    const uint8_t *p, *end;
} reader;

static const void *take(reader *r, size_t bytes) {
    const uint8_t *p = r->p;
    bytes = (bytes + 3) & ~(size_t)3;  // every block is 4-byte aligned
    if ((size_t)(r->end - p) < bytes) return NULL;
    r->p += bytes;
    return p;
}

static int take_q8(reader *r, tai_q8 *w, int rows, int cols) {
    w->rows = rows;
    w->cols = cols;
    w->q = (const int8_t *)take(r, (size_t)rows * cols);
    w->s = (const float *)take(r, (size_t)rows * 4);
    return w->q && w->s ? 0 : -1;
}

static int take_q4(reader *r, tai_q4 *w, int rows, int cols) {
    w->rows = rows;
    w->cols = cols;
    w->q = (const uint8_t *)take(r, (size_t)rows * cols / 2);
    w->s = (const uint16_t *)take(r, (size_t)rows * (cols / TAI_GROUP) * 2);
    return w->q && w->s && cols % TAI_GROUP == 0 ? 0 : -1;
}

static const char *take_strings(reader *r) {
    const uint32_t *len = (const uint32_t *)take(r, 4);
    return len ? (const char *)take(r, *len) : NULL;
}

static void *zalloc(size_t n) { return calloc(1, n); }

int tai_load(tai_model *m, const uint8_t *blob, size_t len) {
    memset(m, 0, sizeof *m);
    if (((uintptr_t)blob & 3) != 0) return -2;
    reader r = {blob, blob + len};
    const uint32_t *h = (const uint32_t *)take(&r, 11 * 4);
    if (!h || h[0] != TAI_MAGIC || h[1] != TAI_VERSION) return -1;
    m->vocab = (int)h[2];
    m->ctx = (int)h[3];
    m->dim = (int)h[4];
    m->layers = (int)h[5];
    m->heads = (int)h[6];
    m->kv_heads = (int)h[7];
    m->hidden = (int)h[8];
    m->n_facts = (int)h[9];
    m->n_known = (int)h[10];
    int D = m->dim, H = m->hidden, KV = m->kv_heads * (D / m->heads);

    if (take_q8(&r, &m->tok, m->vocab, D) || take_q8(&r, &m->pos, m->ctx, D)) return -1;
    m->layer = (tai_layer *)zalloc(sizeof(tai_layer) * (size_t)m->layers);
    if (!m->layer) return -3;
    for (int l = 0; l < m->layers; l++) {
        tai_layer *L = &m->layer[l];
        L->n1 = (const float *)take(&r, (size_t)D * 4);
        if (!L->n1 || take_q4(&r, &L->wq, D, D) || take_q4(&r, &L->wk, KV, D) ||
            take_q4(&r, &L->wv, KV, D) || take_q4(&r, &L->wo, D, D))
            return -1;
        L->n2 = (const float *)take(&r, (size_t)D * 4);
        if (!L->n2 || take_q4(&r, &L->w1, H, D) || take_q4(&r, &L->w2, D, H)) return -1;
    }
    m->norm = (const float *)take(&r, (size_t)D * 4);
    m->facts = take_strings(&r);
    const uint32_t *nw = (const uint32_t *)take(&r, 4);
    if (!m->norm || !m->facts || !nw) return -1;
    m->n_words = (int)*nw;
    m->words = take_strings(&r);
    m->word_off = (const uint32_t *)take(&r, (size_t)m->n_words * 4);
    m->word_len = (const uint8_t *)take(&r, (size_t)m->n_words);
    m->known_fact = (const uint16_t *)take(&r, (size_t)m->n_known * 2);
    m->known_qtype = (const uint8_t *)take(&r, (size_t)m->n_known);
    m->known_nwords = (const uint8_t *)take(&r, (size_t)m->n_known);
    m->known_len = (const uint8_t *)take(&r, (size_t)m->n_known);
    const uint32_t *n_ids = (const uint32_t *)take(&r, 4);
    if (!m->words || !m->word_off || !m->word_len || !m->known_fact || !m->known_qtype ||
        !m->known_nwords || !m->known_len || !n_ids)
        return -1;
    m->known_words = (const uint16_t *)take(&r, (size_t)*n_ids * 2);
    m->small_talk = take_strings(&r);
    if (!m->known_words || !m->small_talk) return -1;

    int big = H > D ? H : D;
    m->x = (float *)zalloc((size_t)D * 4);
    m->xb = (float *)zalloc((size_t)D * 4);
    m->q = (float *)zalloc((size_t)D * 4);
    m->k = (float *)zalloc((size_t)KV * 4);
    m->v = (float *)zalloc((size_t)KV * 4);
    m->hb = (float *)zalloc((size_t)big * 4);
    m->att = (float *)zalloc((size_t)m->ctx * 4);
    m->logits = (float *)zalloc((size_t)m->vocab * 4);
    m->xq = (int8_t *)zalloc((size_t)big);
    m->xsum = (int32_t *)zalloc((size_t)(big / TAI_GROUP + 1) * 4);
    m->kc = (int8_t **)zalloc(sizeof(int8_t *) * (size_t)m->layers);
    m->vc = (int8_t **)zalloc(sizeof(int8_t *) * (size_t)m->layers);
    m->ks = (float **)zalloc(sizeof(float *) * (size_t)m->layers);
    m->vs = (float **)zalloc(sizeof(float *) * (size_t)m->layers);
    if (!m->x || !m->xb || !m->q || !m->k || !m->v || !m->hb || !m->att || !m->logits || !m->xq ||
        !m->xsum || !m->kc || !m->vc || !m->ks || !m->vs)
        return -3;
    // KV cache is int8 too: one allocation per layer keeps each block small
    // enough for the ESP32's fragmented heap.
    for (int l = 0; l < m->layers; l++) {
        m->kc[l] = (int8_t *)zalloc((size_t)m->ctx * KV);
        m->vc[l] = (int8_t *)zalloc((size_t)m->ctx * KV);
        m->ks[l] = (float *)zalloc((size_t)m->ctx * m->kv_heads * 4);
        m->vs[l] = (float *)zalloc((size_t)m->ctx * m->kv_heads * 4);
        if (!m->kc[l] || !m->vc[l] || !m->ks[l] || !m->vs[l]) return -3;
    }
    return 0;
}

void tai_free(tai_model *m) {
    if (m->kc)
        for (int l = 0; l < m->layers; l++) {
            free(m->kc[l]);
            free(m->vc[l]);
            free(m->ks[l]);
            free(m->vs[l]);
        }
    free(m->kc); free(m->vc); free(m->ks); free(m->vs);
    free(m->x); free(m->xb); free(m->q); free(m->k); free(m->v);
    free(m->hb); free(m->att); free(m->logits); free(m->xq); free(m->xsum); free(m->layer);
    memset(m, 0, sizeof *m);
}

// ---------------------------------------------------------------- math

static void rmsnorm(float *o, const float *x, const float *w, int n) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + 1e-5f);
    for (int i = 0; i < n; i++) o[i] = x[i] * ss * w[i];
}

// Symmetric int8 quantization of a vector; returns the scale.
static float quantize(int8_t *q, const float *x, int n) {
    float amax = 0;
    for (int i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    float s = amax / 127.0f;
    float inv = s > 0 ? 1.0f / s : 0;
    for (int i = 0; i < n; i++) q[i] = (int8_t)lrintf(x[i] * inv);
    return s;
}

// Quantizes the matmul input once; the per-group sums let the int4 kernel
// use unsigned nibbles: sum((w - 8) * x) = sum(w * x) - 8 * sum(x).
static void set_input(tai_model *m, const float *x, int n) {
    m->xs = quantize(m->xq, x, n);
    for (int g = 0; g < n / TAI_GROUP; g++) {
        int32_t s = 0;
        for (int i = 0; i < TAI_GROUP; i++) s += m->xq[g * TAI_GROUP + i];
        m->xsum[g] = s;
    }
}

// Every block in the model file is 4-byte aligned, and so is each int4 row
// (cols/2 bytes, cols a multiple of 32). Saying so lets GCC emit a single
// 32-bit load instead of four byte loads.
#if defined(__GNUC__)
#define ALIGNED4(p) __builtin_assume_aligned((p), 4)
#else
#define ALIGNED4(p) (p)
#endif

typedef struct {
    float *o;
    const tai_model *m;
    const tai_q4 *w;
} mm_job;

// IEEE half -> float. Scales are positive normals (export keeps them there);
// zero and subnormals are handled anyway.
static float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 31, man = h & 1023, bits;
    if (exp == 0) {
        float f = (float)man * (1.0f / 16777216.0f);  // 2^-24
        return sign ? -f : f;
    }
    bits = exp == 31 ? sign | 0x7f800000u | (man << 13) : sign | ((exp + 112) << 23) | (man << 13);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// Rows [lo, hi) of o = W x for int4 W, with x already set by set_input().
static void matmul4_rows(void *ctx, int lo, int hi) {
    const mm_job *j = (const mm_job *)ctx;
    const tai_q4 *w = j->w;
    int groups = w->cols / TAI_GROUP;
    const uint8_t *p = w->q + (size_t)lo * (w->cols / 2);
    const uint16_t *s = w->s + (size_t)lo * groups;
    for (int r = lo; r < hi; r++) {
        const int8_t *x = j->m->xq;
        float acc = 0;
        for (int g = 0; g < groups; g++, s++) {
            int32_t dot = 0;
            for (int i = 0; i < TAI_GROUP / 8; i++, p += 4, x += 8) {
                uint32_t b;
                memcpy(&b, ALIGNED4(p), 4);  // one flash read for 8 weights
                dot += (int32_t)(b & 15) * x[0] + (int32_t)((b >> 4) & 15) * x[1] +
                       (int32_t)((b >> 8) & 15) * x[2] + (int32_t)((b >> 12) & 15) * x[3] +
                       (int32_t)((b >> 16) & 15) * x[4] + (int32_t)((b >> 20) & 15) * x[5] +
                       (int32_t)((b >> 24) & 15) * x[6] + (int32_t)(b >> 28) * x[7];
            }
            acc += (float)(dot - 8 * j->m->xsum[g]) * half_to_float(*s);
        }
        j->o[r] = acc * j->m->xs;
    }
}

static void matmul4(float *o, const tai_model *m, const tai_q4 *w) {
    mm_job j = {o, m, w};
    if (m->parallel && w->rows >= 32) m->parallel(matmul4_rows, &j, w->rows);
    else matmul4_rows(&j, 0, w->rows);
}

void tai_set_parallel(tai_model *m, tai_parallel_fn fn) { m->parallel = fn; }

static void mm4(float *o, tai_model *m, const float *x, const tai_q4 *w) {
    set_input(m, x, w->cols);
    matmul4(o, m, w);
}

static float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

static void softmax(float *x, int n) {
    float mx = x[0], sum = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > mx) mx = x[i];
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - mx);
        sum += x[i];
    }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

// ---------------------------------------------------------------- forward

static void forward(tai_model *m, int token, int pos) {
    int D = m->dim, H = m->hidden, NH = m->heads, NKV = m->kv_heads, hd = D / NH;
    int KV = NKV * hd, rep = NH / NKV;
    const int8_t *te = m->tok.q + (size_t)token * D, *pe = m->pos.q + (size_t)pos * D;
    float ts = m->tok.s[token], ps = m->pos.s[pos];
    for (int i = 0; i < D; i++) m->x[i] = te[i] * ts + pe[i] * ps;

    for (int l = 0; l < m->layers; l++) {
        const tai_layer *L = &m->layer[l];
        rmsnorm(m->xb, m->x, L->n1, D);
        set_input(m, m->xb, D);
        matmul4(m->q, m, &L->wq);
        matmul4(m->k, m, &L->wk);
        matmul4(m->v, m, &L->wv);
        for (int h = 0; h < NKV; h++) {
            size_t off = (size_t)pos * KV + (size_t)h * hd;
            m->ks[l][pos * NKV + h] = quantize(m->kc[l] + off, m->k + h * hd, hd);
            m->vs[l][pos * NKV + h] = quantize(m->vc[l] + off, m->v + h * hd, hd);
        }
        float scale = 1.0f / sqrtf((float)hd);
        for (int h = 0; h < NH; h++) {
            int kvh = h / rep;  // grouped-query attention: heads share K/V
            const float *q = m->q + h * hd;
            for (int t = 0; t <= pos; t++) {
                const int8_t *k = m->kc[l] + (size_t)t * KV + kvh * hd;
                float s = 0;
                for (int i = 0; i < hd; i++) s += q[i] * k[i];
                m->att[t] = s * m->ks[l][t * NKV + kvh] * scale;
            }
            softmax(m->att, pos + 1);
            float *o = m->xb + h * hd;
            for (int i = 0; i < hd; i++) o[i] = 0;
            for (int t = 0; t <= pos; t++) {
                const int8_t *v = m->vc[l] + (size_t)t * KV + kvh * hd;
                float a = m->att[t] * m->vs[l][t * NKV + kvh];
                for (int i = 0; i < hd; i++) o[i] += a * v[i];
            }
        }
        mm4(m->q, m, m->xb, &L->wo);  // m->q reused as scratch
        for (int i = 0; i < D; i++) m->x[i] += m->q[i];

        rmsnorm(m->xb, m->x, L->n2, D);
        mm4(m->hb, m, m->xb, &L->w1);
        for (int i = 0; i < H; i++) m->hb[i] = gelu(m->hb[i]);
        mm4(m->xb, m, m->hb, &L->w2);
        for (int i = 0; i < D; i++) m->x[i] += m->xb[i];
    }
    rmsnorm(m->xb, m->x, m->norm, D);
    // Output head tied to the int8 embedding.
    for (int v = 0; v < m->vocab; v++) {
        const int8_t *e = m->tok.q + (size_t)v * D;
        float s = 0;
        for (int i = 0; i < D; i++) s += e[i] * m->xb[i];
        m->logits[v] = s * m->tok.s[v];
    }
}

static void generate_normalized(tai_model *m, const char *norm, char *out, int out_len) {
    int pos = 0, n = 0, tok;
    for (const char *c = norm; *c && pos < m->ctx - 1; c++) forward(m, encode_char(*c), pos++);
    tok = TAI_SEP;
    // Greedy decoding: the most likely answer, every time. No sampling, no
    // creativity, no rambling. Stops at EOS or the hard length cap.
    while (pos < m->ctx && n < TAI_MAX_A && n < out_len - 1) {
        forward(m, tok, pos++);
        int best = 0;
        for (int i = 1; i < m->vocab; i++)
            if (m->logits[i] > m->logits[best]) best = i;
        if (best == TAI_EOS || best == TAI_SEP) break;
        out[n++] = (char)(best + 30);
        tok = best;
    }
    out[n] = 0;
}

void tai_generate(tai_model *m, const char *question, char *out, int out_len) {
    char norm[TAI_MAX_Q + 1];
    tai_normalize(question, norm, sizeof norm);
    generate_normalized(m, norm, out, out_len);
}

static void copy_out(char *out, int out_len, const char *s) {
    int n = (int)strlen(s);
    if (n > out_len - 1) n = out_len - 1;
    memcpy(out, s, (size_t)n);
    out[n] = 0;
}

int tai_ask(tai_model *m, const char *question, char *out, int out_len) {
    char norm[TAI_MAX_Q + 1];
    tai_normalize(question, norm, sizeof norm);
    if (!norm[0]) {
        copy_out(out, out_len, "Ask a question.");
        return 2;
    }
    if (tai_calc(norm, out, out_len)) return 1;
    int fact = tai_gate(m, norm);
    if (!fact) {
        copy_out(out, out_len, "I don't know.");
        return 2;
    }
    // The model answers the matched fact's canonical key (its shortest
    // phrasing), not the raw text: it only ever sees inputs it was trained on.
    generate_normalized(m, tai_fact(m, fact - 1), out, out_len);
    return 0;
}
