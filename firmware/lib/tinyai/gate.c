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
    "today", "tomorrow", "tonight", "currently", "right",
};

// Same meaning, one spelling.
static const char *const SYNONYMS[][2] = {
    {"begin", "start"}, {"began", "start"}, {"begins", "start"}, {"started", "start"},
    {"starts", "start"}, {"ended", "end"}, {"ends", "end"}, {"finish", "end"},
    {"finished", "end"}, {"biggest", "largest"}, {"quickest", "fastest"},
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
        for (size_t i = 0; i < sizeof SYNONYMS / sizeof *SYNONYMS; i++)
            if (!strcmp(w[n], SYNONYMS[i][0])) strcpy(w[n], SYNONYMS[i][1]);
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

// 2 = exact word present, 1 = typo/plural match, 0 = absent.
static int found(const char *w, char o[MAX_WORDS][MAX_WLEN], int no) {
    int best = 0;
    for (int j = 0; j < no && best < 2; j++)
        if (!strcmp(w, o[j])) best = 2;
        else if (word_match(w, o[j])) best = 1;
    return best;
}

typedef struct {
    int hard_missing;  // content words of `w` not found (soft words excused)
    int matched;       // words found
    int quality;       // sum of found() scores
} coverage;

// How well the words of `w` are covered by `o`. Split words are joined:
// "humming bird" matches "hummingbird".
static coverage covered(char w[MAX_WORDS][MAX_WLEN], int n, char o[MAX_WORDS][MAX_WLEN], int no) {
    coverage c = {0, 0, 0};
    for (int i = 0; i < n; i++) {
        int f = found(w[i], o, no);
        if (!f && i + 1 < n && strlen(w[i]) + strlen(w[i + 1]) < MAX_WLEN) {
            char joined[MAX_WLEN];
            strcpy(joined, w[i]);
            strcat(joined, w[i + 1]);
            if ((f = found(joined, o, no)) != 0) {
                c.matched++;
                c.quality += f;
                i++;
            }
        }
        if (f) {
            c.matched++;
            c.quality += f;
        } else if (!is_soft(w[i])) {
            c.hard_missing++;
        }
    }
    return c;
}

// Question type: "who", "when", "how many", "how far", ... or "" if none.
// A "how many" question must never be answered by a "what" fact:
// "how many moons does jupiter have" is not "what is jupiter's largest moon".
static void qtype(const char *s, char *t, int t_len) {
    t[0] = 0;
    char w[MAX_WLEN];
    while (*s) {
        int len = 0;
        while (*s == ' ') s++;
        while (*s && *s != ' ') {
            if (*s != '\'' && len < MAX_WLEN - 1) w[len++] = *s;
            s++;
        }
        w[len] = 0;
        const char *type = NULL;
        while (*s == ' ') s++;
        if ((!strcmp(w, "what") || !strcmp(w, "which")) && !strncmp(s, "year", 4) &&
            (s[4] == ' ' || s[4] == 0))
            type = "when";  // "what year did ww2 end" asks when
        else if (!strcmp(w, "what") || !strcmp(w, "whats") || !strcmp(w, "which")) type = "what";
        else if (!strcmp(w, "who") || !strcmp(w, "whos") || !strcmp(w, "whose")) type = "who";
        else if (!strcmp(w, "when") || !strcmp(w, "where") || !strcmp(w, "why")) type = w;
        if (type) {
            strncpy(t, type, (size_t)t_len - 1);
            t[t_len - 1] = 0;
            return;
        }
        if (!strcmp(w, "how")) {
            static const char *const HOW[] = {"many", "much", "long", "far", "fast", "old", "big",
                                              "tall", "heavy", "hot", "cold", "deep", "high",
                                              "often", "smart"};
            strcpy(t, "how");
            for (size_t i = 0; i < sizeof HOW / sizeof *HOW; i++) {
                size_t n = strlen(HOW[i]);
                if (!strncmp(s, HOW[i], n) && (s[n] == ' ' || s[n] == 0)) {
                    strcat(t, " ");
                    strcat(t, HOW[i]);
                    break;
                }
            }
            return;
        }
    }
}

// Character-bigram overlap (Dice coefficient) of two strings, 0..1000.
static int dice(const char *a, const char *b) {
    int la = (int)strlen(a), lb = (int)strlen(b), common = 0;
    if (la < 2 || lb < 2) return strcmp(a, b) ? 0 : 1000;
    unsigned char used[TAI_MAX_Q + 1] = {0};
    for (int i = 0; i + 1 < la; i++)
        for (int j = 0; j + 1 < lb && j < TAI_MAX_Q; j++)
            if (!used[j] && a[i] == b[j] && a[i + 1] == b[j + 1]) {
                used[j] = 1;
                common++;
                break;
            }
    return 2000 * common / (la - 1 + lb - 1);
}

// Picks the known question closest to the input. Every content word of the
// input must be accounted for, the question types must agree, and the known
// question must be at least half present in the input. Among those, exact
// words beat typo matches and leftover unmatched words cost points.
// 0 = refuse, else 1 + fact index.
int tai_gate(const tai_model *m, const char *norm) {
    char in[MAX_WORDS][MAX_WLEN], kw[MAX_WORDS][MAX_WLEN], tin[16], tk[16];
    int n = content_words(norm, in);
    qtype(norm, tin, sizeof tin);
    long best_score = -1;
    int best = 0;
    const char *k = m->known;
    for (int i = 0; i < m->n_known; i++, k += strlen(k) + 1) {
        int nk = content_words(k, kw);
        long score;
        qtype(k, tk, sizeof tk);
        if (tin[0] && tk[0] && strcmp(tin, tk)) continue;
        if (n == 0 || nk == 0) {
            // Small talk ("hi", "how are you"): nothing to fact-check, so
            // match the whole phrase instead.
            if (n != nk) continue;
            int d = dice(norm, k);
            if (d < 500) continue;
            score = d;
        } else {
            coverage a = covered(in, n, kw, nk), b = covered(kw, nk, in, n);
            if (a.hard_missing || 2 * b.matched < nk) continue;
            score = 1000L * (a.quality + b.quality - 2 * (nk - b.matched)) + dice(norm, k);
        }
        if (score > best_score) {
            best_score = score;
            best = m->known_fact[i] + 1;
        }
    }
    return best;
}
