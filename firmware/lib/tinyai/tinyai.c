// Inference for the TinyGPT in train/model.py.
// Weights are int8 with one float scale per row; activations are quantized
// to int8 on the fly so every matmul is an int8 x int8 -> int32 dot product.
#include "tinyai.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------- text

static int prompt_char(char c) {
    return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || strchr(" '+-*/.,%^()=", c);
}

// Same rules as normalize() in train/common.py.
int tai_normalize(const char *in, char *out, int out_len) {
    char tmp[512];
    int n = 0, prev_space = 1;
    for (; *in && n < (int)sizeof tmp - 1; in++) {
        char c = *in;
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
        if (!c || !prompt_char(c)) c = ' ';
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

static int take_qmat(reader *r, tai_qmat *w, int rows, int cols) {
    w->rows = rows;
    w->cols = cols;
    w->q = (const int8_t *)take(r, (size_t)rows * cols);
    w->s = (const float *)take(r, (size_t)rows * 4);
    return w->q && w->s ? 0 : -1;
}

static void *zalloc(size_t n) { return calloc(1, n); }

int tai_load(tai_model *m, const uint8_t *blob, size_t len) {
    memset(m, 0, sizeof *m);
    if (((uintptr_t)blob & 3) != 0) return -2;
    reader r = {blob, blob + len};
    const uint32_t *h = (const uint32_t *)take(&r, 9 * 4);
    if (!h || h[0] != TAI_MAGIC || h[1] != 1) return -1;
    m->vocab = (int)h[2];
    m->ctx = (int)h[3];
    m->dim = (int)h[4];
    m->layers = (int)h[5];
    m->heads = (int)h[6];
    m->hidden = (int)h[7];
    m->n_known = (int)h[8];
    int D = m->dim, H = m->hidden;

    if (take_qmat(&r, &m->tok, m->vocab, D) || take_qmat(&r, &m->pos, m->ctx, D)) return -1;
    m->layer = (tai_layer *)zalloc(sizeof(tai_layer) * (size_t)m->layers);
    if (!m->layer) return -3;
    for (int l = 0; l < m->layers; l++) {
        tai_layer *L = &m->layer[l];
        L->n1 = (const float *)take(&r, (size_t)D * 4);
        if (!L->n1 || take_qmat(&r, &L->wq, D, D) || take_qmat(&r, &L->wk, D, D) ||
            take_qmat(&r, &L->wv, D, D) || take_qmat(&r, &L->wo, D, D))
            return -1;
        L->n2 = (const float *)take(&r, (size_t)D * 4);
        if (!L->n2 || take_qmat(&r, &L->w1, H, D) || take_qmat(&r, &L->w2, D, H)) return -1;
    }
    m->norm = (const float *)take(&r, (size_t)D * 4);
    const uint32_t *klen = (const uint32_t *)take(&r, 4);
    if (!m->norm || !klen) return -1;
    m->known = (const char *)take(&r, *klen);
    if (!m->known) return -1;

    int big = H > D ? H : D;
    m->x = (float *)zalloc((size_t)D * 4);
    m->xb = (float *)zalloc((size_t)D * 4);
    m->q = (float *)zalloc((size_t)D * 4);
    m->k = (float *)zalloc((size_t)D * 4);
    m->v = (float *)zalloc((size_t)D * 4);
    m->hb = (float *)zalloc((size_t)big * 4);
    m->att = (float *)zalloc((size_t)m->ctx * 4);
    m->logits = (float *)zalloc((size_t)m->vocab * 4);
    m->xq = (int8_t *)zalloc((size_t)big);
    m->kc = (int8_t **)zalloc(sizeof(int8_t *) * (size_t)m->layers);
    m->vc = (int8_t **)zalloc(sizeof(int8_t *) * (size_t)m->layers);
    m->ks = (float **)zalloc(sizeof(float *) * (size_t)m->layers);
    m->vs = (float **)zalloc(sizeof(float *) * (size_t)m->layers);
    if (!m->x || !m->xb || !m->q || !m->k || !m->v || !m->hb || !m->att || !m->logits || !m->xq ||
        !m->kc || !m->vc || !m->ks || !m->vs)
        return -3;
    // KV cache is int8 too: one allocation per layer keeps each block small
    // enough for the ESP32's fragmented heap.
    for (int l = 0; l < m->layers; l++) {
        m->kc[l] = (int8_t *)zalloc((size_t)m->ctx * D);
        m->vc[l] = (int8_t *)zalloc((size_t)m->ctx * D);
        m->ks[l] = (float *)zalloc((size_t)m->ctx * m->heads * 4);
        m->vs[l] = (float *)zalloc((size_t)m->ctx * m->heads * 4);
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
    free(m->hb); free(m->att); free(m->logits); free(m->xq); free(m->layer);
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

static int32_t dot8(const int8_t *a, const int8_t *b, int n) {
    int32_t acc = 0;
    for (int i = 0; i < n; i++) acc += (int32_t)a[i] * b[i];
    return acc;
}

// o = W x, with x already quantized in m->xq / m->xs
static void matmul(float *o, const tai_model *m, const tai_qmat *w) {
    for (int r = 0; r < w->rows; r++)
        o[r] = (float)dot8(w->q + (size_t)r * w->cols, m->xq, w->cols) * w->s[r] * m->xs;
}

static void qmm(float *o, tai_model *m, const float *x, const tai_qmat *w) {
    m->xs = quantize(m->xq, x, w->cols);
    matmul(o, m, w);
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
    int D = m->dim, H = m->hidden, NH = m->heads, hd = D / NH;
    const int8_t *te = m->tok.q + (size_t)token * D, *pe = m->pos.q + (size_t)pos * D;
    float ts = m->tok.s[token], ps = m->pos.s[pos];
    for (int i = 0; i < D; i++) m->x[i] = te[i] * ts + pe[i] * ps;

    for (int l = 0; l < m->layers; l++) {
        const tai_layer *L = &m->layer[l];
        rmsnorm(m->xb, m->x, L->n1, D);
        m->xs = quantize(m->xq, m->xb, D);
        matmul(m->q, m, &L->wq);
        matmul(m->k, m, &L->wk);
        matmul(m->v, m, &L->wv);
        for (int h = 0; h < NH; h++) {
            size_t off = (size_t)pos * D + (size_t)h * hd;
            m->ks[l][pos * NH + h] = quantize(m->kc[l] + off, m->k + h * hd, hd);
            m->vs[l][pos * NH + h] = quantize(m->vc[l] + off, m->v + h * hd, hd);
        }
        float scale = 1.0f / sqrtf((float)hd);
        for (int h = 0; h < NH; h++) {
            const float *q = m->q + h * hd;
            for (int t = 0; t <= pos; t++) {
                const int8_t *k = m->kc[l] + (size_t)t * D + h * hd;
                float s = 0;
                for (int i = 0; i < hd; i++) s += q[i] * k[i];
                m->att[t] = s * m->ks[l][t * NH + h] * scale;
            }
            softmax(m->att, pos + 1);
            float *o = m->xb + h * hd;
            for (int i = 0; i < hd; i++) o[i] = 0;
            for (int t = 0; t <= pos; t++) {
                const int8_t *v = m->vc[l] + (size_t)t * D + h * hd;
                float a = m->att[t] * m->vs[l][t * NH + h];
                for (int i = 0; i < hd; i++) o[i] += a * v[i];
            }
        }
        qmm(m->q, m, m->xb, &L->wo);  // m->q reused as scratch
        for (int i = 0; i < D; i++) m->x[i] += m->q[i];

        rmsnorm(m->xb, m->x, L->n2, D);
        qmm(m->hb, m, m->xb, &L->w1);
        for (int i = 0; i < H; i++) m->hb[i] = gelu(m->hb[i]);
        qmm(m->xb, m, m->hb, &L->w2);
        for (int i = 0; i < D; i++) m->x[i] += m->xb[i];
    }
    rmsnorm(m->xb, m->x, m->norm, D);
    qmm(m->logits, m, m->xb, &m->tok);  // output head tied to the embedding
}

static void generate_normalized(tai_model *m, const char *norm, char *out, int out_len) {
    int pos = 0, n = 0, tok = 0;
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
    if (!tai_gate(m, norm)) {
        copy_out(out, out_len, "I don't know.");
        return 2;
    }
    generate_normalized(m, norm, out, out_len);
    return 0;
}
