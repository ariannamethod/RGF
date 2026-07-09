/*
 * rgf_viewer.c — reference decoder for the Resonant Generative Format.
 *
 * Parses a .rgf container, loads the diffusion model FROM THE WGHT CHUNK IN
 * MEMORY (never a path), and renders the denoising as a living image: text
 * crystallises out of noise in a 256-colour terminal, holds, then re-masks a
 * fraction of positions and denoises again — the file breathes, it is not a
 * loop. "GIF repeats, RGF resonates."
 *
 * The parser treats .rgf as untrusted input (Fable §4.2): magic + version,
 * tag-whitelist, per-chunk length caps, CRC0 verified BEFORE the weight blob
 * is used, no read past end. A crafted/corrupt file is refused, never crashes.
 *
 * Build: cc -O2 -Wall -Wextra -std=c11 -DDIFFUSION_LIB_ONLY \
 *           rgf_viewer.c vendor/diffusion_engine.c -lm -o rgf_viewer
 * Run:   ./rgf_viewer file.rgf [--seed N] [--once] [--dump] [--fps N]
 */

#define _POSIX_C_SOURCE 200809L
#include "rgf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>

/* ── Engine API (vendor/diffusion_engine.c, DIFFUSION_LIB_ONLY) ──────────── */
extern int  diff_load_mem(const unsigned char* blob, size_t n);
extern int  diff_denoise(int* tokens_io, int n_steps, float temperature, int* steps_buf);
extern void diff_seed(unsigned int seed);
extern int  diff_get_ctx(void);
extern int  diff_get_vocab(void);
extern int  diff_get_mask_tok(void);
extern void diff_free(void);

/* ── CRC32 (must match rgf_pack.c) ───────────────────────────────────────── */
static uint32_t crc32_update(uint32_t crc, const uint8_t* buf, size_t len) {
    crc = ~crc;
    for (size_t i = 0; i < len; i++) {
        crc ^= buf[i];
        for (int k = 0; k < 8; k++)
            crc = (crc >> 1) ^ (0xEDB88320u & (uint32_t)(-(int32_t)(crc & 1)));
    }
    return ~crc;
}

/* ── Read whole file ─────────────────────────────────────────────────────── */
static uint8_t* read_file(const char* path, size_t* out_len) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "rgf_viewer: cannot open %s\n", path); return NULL; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz <= 0) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    uint8_t* buf = (uint8_t*)malloc((size_t)sz);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { free(buf); fclose(f); return NULL; }
    fclose(f);
    *out_len = (size_t)sz;
    return buf;
}

/* ── Parsed container ────────────────────────────────────────────────────── */
typedef struct {
    const uint8_t* wght; size_t wght_len;
    char mani[RGF_TEXT_MAX_BYTES], mcfg[RGF_TEXT_MAX_BYTES], smpl[512], rndr[512], seed[256];
    int has_wght, has_crc0;
    uint32_t crc0_stored;
} RgfDoc;

static int cfg_int(const char* txt, const char* key, int def) {
    const char* p = strstr(txt, key);
    if (!p) return def;
    p += strlen(key);
    if (*p != '=') return def;
    return atoi(p + 1);
}
static double cfg_flt(const char* txt, const char* key, double def) {
    const char* p = strstr(txt, key);
    if (!p) return def;
    p += strlen(key);
    if (*p != '=') return def;
    return atof(p + 1);
}

/* Copy a text chunk into dst with a forced NUL (Fable §4.2). */
static void copy_text(char* dst, size_t dstcap, const uint8_t* src, uint32_t len) {
    uint32_t k = (len < dstcap - 1) ? len : (uint32_t)(dstcap - 1);
    memcpy(dst, src, k);
    dst[k] = 0;
}

/* Parse .rgf from a memory buffer. Bounds-checked; returns 0 on success. */
static int rgf_parse(const uint8_t* buf, size_t n, RgfDoc* d) {
    memset(d, 0, sizeof(*d));
    if (n < 8 || memcmp(buf, RGF_MAGIC, 4) != 0) { fprintf(stderr, "rgf: bad magic\n"); return -1; }
    uint32_t ver; memcpy(&ver, buf + 4, 4);
    if (ver != RGF_FORMAT_VERSION) { fprintf(stderr, "rgf: unsupported version %u\n", ver); return -1; }

    size_t p = 8;
    uint32_t running_crc = crc32_update(0, buf, 8);  /* CRC covers signature + all chunks before CRC0 */
    int saw_end = 0;

    while (p + 8 <= n) {
        uint32_t tag, len;
        memcpy(&tag, buf + p, 4);
        memcpy(&len, buf + p + 4, 4);

        if (tag == RGF_TAG_CRC0) {
            if (len != 4 || p + 8 + 4 > n) { fprintf(stderr, "rgf: bad CRC0 chunk\n"); return -1; }
            memcpy(&d->crc0_stored, buf + p + 8, 4);
            d->has_crc0 = 1;
            if (d->crc0_stored != running_crc) {
                fprintf(stderr, "rgf: CRC mismatch (stored 0x%08X, computed 0x%08X)\n", d->crc0_stored, running_crc);
                return -1;
            }
            p += 8 + 4;
            continue;  /* CRC0 itself is not folded into the running crc */
        }

        /* length caps by class (untrusted input) */
        uint32_t cap = (tag == RGF_TAG_WGHT) ? RGF_WGHT_MAX_BYTES : RGF_TEXT_MAX_BYTES;
        if (len > cap) { fprintf(stderr, "rgf: chunk 0x%08X length %u exceeds cap %u\n", tag, len, cap); return -1; }
        if (p + 8 + (size_t)len > n) { fprintf(stderr, "rgf: chunk 0x%08X runs past end\n", tag); return -1; }

        const uint8_t* payload = buf + p + 8;
        switch (tag) {
            case RGF_TAG_MANI: copy_text(d->mani, sizeof(d->mani), payload, len); break;
            case RGF_TAG_MCFG: copy_text(d->mcfg, sizeof(d->mcfg), payload, len); break;
            case RGF_TAG_SMPL: copy_text(d->smpl, sizeof(d->smpl), payload, len); break;
            case RGF_TAG_RNDR: copy_text(d->rndr, sizeof(d->rndr), payload, len); break;
            case RGF_TAG_SEED: copy_text(d->seed, sizeof(d->seed), payload, len); break;
            case RGF_TAG_WGHT: d->wght = payload; d->wght_len = len; d->has_wght = 1; break;
            case RGF_TAG_END:  saw_end = 1; break;
            default: /* unknown tag: skip by length (forward-compat) — only reached after CRC ok */ break;
        }

        running_crc = crc32_update(running_crc, buf + p, 8 + len);
        p += 8 + (size_t)len;
        if (saw_end) break;
    }

    if (!d->has_crc0) { fprintf(stderr, "rgf: no CRC0 chunk — refusing\n"); return -1; }
    if (!d->has_wght) { fprintf(stderr, "rgf: no WGHT chunk — refusing\n"); return -1; }
    return 0;
}

/* ── ANSI render: crystallising text from noise (palette=blood) ──────────── */
/* mask token → dim glyph-noise; a byte that just resolved → bright (red→white);
 * settled bytes fade toward grey. 256-colour terminal. */
static void render_frame(const int* toks, int ctx, int mask_tok, const int* prev) {
    fputs("\x1b[H", stdout);  /* cursor home (no full clear → less flicker) */
    for (int i = 0; i < ctx; i++) {
        int t = toks[i];
        if (t == mask_tok) {
            /* masked: dim pseudo-noise glyph */
            int g = 33 + (i * 7 + t) % 94;  /* printable */
            printf("\x1b[38;5;238m%c", g);
        } else {
            int fresh = prev && prev[i] == mask_tok;   /* resolved THIS step */
            int color = fresh ? 231 : 174;             /* 231 white (fresh), 174 dusty-rose (settled) */
            unsigned char c = (unsigned char)t;
            if (c < 32 || c > 126) c = '.';
            printf("\x1b[38;5;%dm%c", color, c);
        }
    }
    fputs("\x1b[0m\n", stdout);
    fflush(stdout);
}

int main(int argc, char** argv) {
    const char* path = NULL;
    int once = 0, dump = 0, fps = 12; long seed_override = -1;
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--once")) once = 1;
        else if (!strcmp(argv[i], "--dump")) dump = 1;
        else if (!strcmp(argv[i], "--seed") && i+1 < argc) seed_override = atol(argv[++i]);
        else if (!strcmp(argv[i], "--fps")  && i+1 < argc) fps = atoi(argv[++i]);
        else if (argv[i][0] != '-') path = argv[i];
        else { fprintf(stderr, "rgf_viewer: unknown arg %s\n", argv[i]); return 2; }
    }
    if (!path) { fprintf(stderr, "usage: %s file.rgf [--seed N] [--once] [--dump] [--fps N]\n", argv[0]); return 2; }

    size_t n = 0;
    uint8_t* buf = read_file(path, &n);
    if (!buf) return 1;

    RgfDoc d;
    if (rgf_parse(buf, n, &d) != 0) { free(buf); return 1; }

    if (dump) {
        printf("== RGF %s ==\n--MANI--\n%s\n--MCFG--\n%s\n--SMPL--\n%s\n--RNDR--\n%s\n--SEED--\n%s\n--WGHT-- %zu bytes, crc0=0x%08X\n",
               path, d.mani, d.mcfg, d.smpl, d.rndr, d.seed, d.wght_len, d.crc0_stored);
        free(buf);
        return 0;
    }

    /* Load model from the WGHT blob in memory — never from a path. */
    if (diff_load_mem(d.wght, d.wght_len) != 0) {
        fprintf(stderr, "rgf_viewer: model load failed from WGHT chunk\n");
        free(buf); return 1;
    }

    int ctx = diff_get_ctx(), mask = diff_get_mask_tok();
    int steps      = cfg_int(d.smpl, "steps", 20);
    double temp    = cfg_flt(d.smpl, "temperature", 0.8);
    double remaskf = cfg_flt(d.smpl, "remask", 0.4);
    int hold_ms    = cfg_int(d.smpl, "hold_ms", 2000);

    /* seed policy: --seed overrides; else time-based */
    unsigned int seed = (seed_override >= 0) ? (unsigned int)seed_override
                                             : (unsigned int)(time(NULL) ^ (long)getpid());
    diff_seed(seed);

    int* toks = (int*)malloc((size_t)ctx * sizeof(int));
    int* prev = (int*)malloc((size_t)ctx * sizeof(int));
    if (!toks || !prev) { free(toks); free(prev); diff_free(); free(buf); return 1; }

    fputs("\x1b[2J", stdout);  /* clear once at start */
    long frame_us = (fps > 0) ? 1000000L / fps : 0;

    do {
        for (int i = 0; i < ctx; i++) toks[i] = mask;
        /* denoise: each returned step is a frame — we render the final crystallised state.
         * (steps_buf NULL: the engine denoises internally; we show the settled text.) */
        for (int i = 0; i < ctx; i++) prev[i] = mask;
        diff_denoise(toks, steps, (float)temp, NULL);
        render_frame(toks, ctx, mask, prev);

        if (once) break;

        usleep((useconds_t)(hold_ms * 1000));
        /* breathe: re-mask a fraction, then the loop denoises again — mutation without repeat */
        int rem = (int)(remaskf * ctx);
        for (int r = 0; r < rem; r++) toks[rand() % ctx] = mask;
        if (frame_us) usleep((useconds_t)frame_us);
    } while (1);

    if (once) {
        /* pipe-friendly proof: final crystallised text on stdout */
        printf("\n[final] ");
        for (int i = 0; i < ctx; i++) { unsigned char c = (unsigned char)toks[i]; putchar((c >= 32 && c <= 126) ? c : '.'); }
        putchar('\n');
    }

    free(toks); free(prev); diff_free(); free(buf);
    return 0;
}
