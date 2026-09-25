// Knowledge gate. A 700k-parameter model has no idea what it does not know:
// ask it for the capital of Bolivia and it will confidently say something.
// So before generating, check that the question's content words match a
// question the model was trained on. If not, answer "I don't know." —
// a direct answer beats a wrong one.
#include <string.h>

#include "tinyai.h"

#define MAX_WORDS 16
#define MAX_WLEN 24

static const char *const STOP[] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "do", "does", "did", "of", "in", "on",
    "at", "to", "for", "from", "by", "with", "and", "or", "what", "whats", "who", "whos", "how",
    "when", "where", "which", "why", "tell", "me", "please", "you", "u", "your", "can", "could",
    "would", "will", "i", "im", "want", "know", "wondering", "question", "quick", "hey", "hi",
    "yo", "ok", "so", "exactly", "roughly", "approximately", "about", "again", "quickly",
    "much", "many", "there", "it", "its", "that", "this", "we", "my", "have", "has", "called",
    "just", "like", "um", "uh", "hmm", "now", "actually", "really", "some", "any", "all",
};

// Words that add context but never change which fact is meant
// ("capital *city* of france", "tallest mountain on *earth*"). The input may
// contain them without the known question having them.
static const char *const SOFT[] = {
    "city", "country", "planet", "earth", "world", "our", "one", "book", "language", "man",
    "woman", "person", "people", "human", "solar", "system", "away", "number", "total",
    "average", "normal", "typical", "usually", "generally", "name", "thing", "kind", "type",
    "exact", "approx", "approximate", "grown", "adult", "full", "whole", "known",
};

static int in_list(const char *w, const char *const *list, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (!strcmp(w, list[i])) return 1;
    return 0;
}

static int is_stop(const char *w) { return in_list(w, STOP, sizeof STOP / sizeof *STOP); }
static int is_soft(const char *w) { return in_list(w, SOFT, sizeof SOFT / sizeof *SOFT); }

// Splits normalized text into content words (letters/digits only, lowercased,
// apostrophes dropped, stop words removed).
static int content_words(const char *s, char w[MAX_WORDS][MAX_WLEN]) {
    int n = 0;
    while (*s && n < MAX_WORDS) {
        while (*s && !((*s >= 'a' && *s <= 'z') || (*s >= '0' && *s <= '9'))) s++;
        int len = 0;
        while (*s && ((*s >= 'a' && *s <= 'z') || (*s >= '0' && *s <= '9') || *s == '\'')) {
            if (*s != '\'' && len < MAX_WLEN - 1) w[n][len++] = *s;
            s++;
        }
        w[n][len] = 0;
        if (len && !is_stop(w[n])) n++;
    }
    return n;
}

static int edit_distance(const char *a, const char *b) {
    int la = (int)strlen(a), lb = (int)strlen(b);
    int row[MAX_WLEN + 1];
    for (int j = 0; j <= lb; j++) row[j] = j;
    for (int i = 1; i <= la; i++) {
        int diag = row[0];
        row[0] = i;
        for (int j = 1; j <= lb; j++) {
            int up = row[j];
            int best = diag + (a[i - 1] != b[j - 1]);
            if (row[j] + 1 < best) best = row[j] + 1;
            if (row[j - 1] + 1 < best) best = row[j - 1] + 1;
            row[j] = best;
            diag = up;
        }
    }
    return row[lb];
}

static int has_digit(const char *s) {
    for (; *s; s++)
        if (*s >= '0' && *s <= '9') return 1;
    return 0;
}

// Same word, allowing for typos and plurals. Numbers must match exactly.
static int word_match(const char *a, const char *b) {
    if (!strcmp(a, b)) return 1;
    if (has_digit(a) || has_digit(b)) return 0;
    int la = (int)strlen(a), lb = (int)strlen(b);
    int shorter = la < lb ? la : lb;
    if (shorter <= 4) {
        // cat/cats: only a trailing 's' may differ. Short words are too close
        // to each other for typo tolerance (bake/cake/take).
        return (la == lb + 1 && a[la - 1] == 's' && !strncmp(a, b, (size_t)lb)) ||
               (lb == la + 1 && b[lb - 1] == 's' && !strncmp(a, b, (size_t)la));
    }
    int d = edit_distance(a, b);
    return d <= (shorter >= 8 ? 2 : 1);
}

static int found(const char *w, char o[MAX_WORDS][MAX_WLEN], int no) {
    for (int j = 0; j < no; j++)
        if (word_match(w, o[j])) return 1;
    return 0;
}

// Counts words of `w` present in `o`. "humming bird" also matches
// "hummingbird". With `hard_only`, soft words are skipped (counted as present).
static int covered(char w[MAX_WORDS][MAX_WLEN], int n, char o[MAX_WORDS][MAX_WLEN], int no,
                   int hard_only) {
    int c = 0;
    for (int i = 0; i < n; i++) {
        if ((hard_only && is_soft(w[i])) || found(w[i], o, no)) {
            c++;
            continue;
        }
        if (i + 1 < n && strlen(w[i]) + strlen(w[i + 1]) < MAX_WLEN) {
            char joined[MAX_WLEN];
            strcpy(joined, w[i]);
            strcat(joined, w[i + 1]);
            if (found(joined, o, no)) {
                c += 2;
                i++;
            }
        }
    }
    return c;
}

// 0 = refuse, otherwise 1 + index of the matching known question.
int tai_gate(const tai_model *m, const char *norm) {
    char in[MAX_WORDS][MAX_WLEN], kw[MAX_WORDS][MAX_WLEN];
    int n = content_words(norm, in);
    if (n == 0) return 1;  // pure small talk ("hi", "how are you"): harmless
    const char *k = m->known;
    for (int i = 0; i < m->n_known; i++, k += strlen(k) + 1) {
        int nk = content_words(k, kw);
        if (nk == 0) continue;
        // Every content word of the input must be known, and the known
        // question must be mostly present in the input.
        if (covered(in, n, kw, nk, 1) == n && 2 * covered(kw, nk, in, n, 0) >= nk) return i + 1;
    }
    return 0;
}
