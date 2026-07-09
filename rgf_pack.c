/*
 * rgf_pack.c — pack a notorch weight dump + params into a .rgf file.
 *
 * Usage:
 *   rgf_pack --weights diffusion.bin --out dracula.rgf --title "Dracula Resonance"
 *            [--author NAME] [--organism dracula-diffusion-v1] [--corpus dracula.txt]
 *            [--seed-policy time|fixed:N|random] [--steps 20] [--temp 0.8] [--remask 0.4]
 *
 * Before writing, the weight blob is validated by a full walk: magic, tensor
 * count == RGF_N_TENSORS, and every tensor's ndim+shape compared against the
 * fixed RGF_TENSOR_SHAPES table. Garbage is never sealed into the format.
 * The WGHT chunk is the verbatim notorch dump (not re-encoded).
 */

#include "rgf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ── CRC32 (IEEE 802.3, standard poly 0xEDB88320) ────────────────────────── */

static uint32_t crc32_update(uint32_t crc, const uint8_t* buf, size_t len) {
    crc = ~crc;
    for (size_t i = 0; i < len; i++) {
        crc ^= buf[i];
        for (int k = 0; k < 8; k++)
            crc = (crc >> 1) ^ (0xEDB88320u & (uint32_t)(-(int32_t)(crc & 1)));
    }
    return ~crc;
}

/* ── Read a whole file into memory ───────────────────────────────────────── */

static uint8_t* read_file(const char* path, size_t* out_len) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "rgf_pack: cannot open %s\n", path); return NULL; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    uint8_t* buf = (uint8_t*)malloc((size_t)sz);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { free(buf); fclose(f); return NULL; }
    fclose(f);
    *out_len = (size_t)sz;
    return buf;
}

/* ── Validate the notorch weight blob against the fixed shape table ──────── */
/* Layout: u32 magic, i32 count, then per-tensor: i32 ndim, i32 shape[ndim], f32 data[prod(shape)]. */

static int validate_blob(const uint8_t* b, size_t n) {
    size_t p = 0;
    if (n < 8) { fprintf(stderr, "rgf_pack: blob too small\n"); return -1; }
    uint32_t magic; memcpy(&magic, b + p, 4); p += 4;
    if (magic != RGF_WGHT_MAGIC) {
        fprintf(stderr, "rgf_pack: bad blob magic 0x%08X (want 0x%08X)\n", magic, RGF_WGHT_MAGIC);
        return -1;
    }
    int32_t count; memcpy(&count, b + p, 4); p += 4;
    if (count != RGF_N_TENSORS) {
        fprintf(stderr, "rgf_pack: tensor count %d != %d\n", count, RGF_N_TENSORS);
        return -1;
    }
    for (int t = 0; t < RGF_N_TENSORS; t++) {
        const rgf_shape* want = &RGF_TENSOR_SHAPES[t];
        if (p + 4 > n) { fprintf(stderr, "rgf_pack: truncated at tensor %d ndim\n", t); return -1; }
        int32_t ndim; memcpy(&ndim, b + p, 4); p += 4;
        if (ndim != want->ndim) {
            fprintf(stderr, "rgf_pack: tensor %d ndim %d != %d\n", t, ndim, want->ndim);
            return -1;
        }
        long len = 1;
        for (int d = 0; d < ndim; d++) {
            if (p + 4 > n) { fprintf(stderr, "rgf_pack: truncated at tensor %d dim %d\n", t, d); return -1; }
            int32_t dim; memcpy(&dim, b + p, 4); p += 4;
            if (dim != want->dims[d]) {
                fprintf(stderr, "rgf_pack: tensor %d dim %d = %d != %d\n", t, d, dim, want->dims[d]);
                return -1;
            }
            len *= dim;
        }
        size_t bytes = (size_t)len * 4;
        if (p + bytes > n) { fprintf(stderr, "rgf_pack: truncated tensor %d data\n", t); return -1; }
        p += bytes;
    }
    if (p != n) fprintf(stderr, "rgf_pack: warning — %zu trailing bytes after tensors\n", n - p);
    return 0;
}

/* ── Chunk writers ───────────────────────────────────────────────────────── */

/* Write one chunk (tag + length + payload) AND fold it into the running CRC. */
static void write_chunk(FILE* f, uint32_t* crc, uint32_t tag, const void* payload, uint32_t len) {
    uint32_t hdr[2] = { tag, len };
    fwrite(hdr, 4, 2, f);
    *crc = crc32_update(*crc, (const uint8_t*)hdr, 8);
    if (len) {
        fwrite(payload, 1, len, f);
        *crc = crc32_update(*crc, (const uint8_t*)payload, len);
    }
}

int main(int argc, char** argv) {
    const char *weights = NULL, *out = NULL, *title = "Untitled";
    const char *author = "Arianna Method", *organism = "dracula-diffusion-v1", *corpus = "dracula.txt";
    const char *seed_policy = "time"; long seed_value = 0;
    int steps = 20; double temp = 0.8, remask = 0.4;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--weights") && i+1 < argc) weights = argv[++i];
        else if (!strcmp(argv[i], "--out")     && i+1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--title")   && i+1 < argc) title = argv[++i];
        else if (!strcmp(argv[i], "--author")  && i+1 < argc) author = argv[++i];
        else if (!strcmp(argv[i], "--organism")&& i+1 < argc) organism = argv[++i];
        else if (!strcmp(argv[i], "--corpus")  && i+1 < argc) corpus = argv[++i];
        else if (!strcmp(argv[i], "--seed-policy") && i+1 < argc) seed_policy = argv[++i];
        else if (!strcmp(argv[i], "--steps")   && i+1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temp")    && i+1 < argc) temp = atof(argv[++i]);
        else if (!strcmp(argv[i], "--remask")  && i+1 < argc) remask = atof(argv[++i]);
        else { fprintf(stderr, "rgf_pack: unknown arg %s\n", argv[i]); return 2; }
    }
    if (!weights || !out) {
        fprintf(stderr, "usage: rgf_pack --weights W.bin --out F.rgf --title T [opts]\n");
        return 2;
    }

    size_t blob_len = 0;
    uint8_t* blob = read_file(weights, &blob_len);
    if (!blob) return 1;
    if (blob_len > RGF_WGHT_MAX_BYTES) {
        fprintf(stderr, "rgf_pack: weight blob %zu exceeds cap %u\n", blob_len, RGF_WGHT_MAX_BYTES);
        free(blob); return 1;
    }
    if (validate_blob(blob, blob_len) != 0) { free(blob); return 1; }

    FILE* f = fopen(out, "wb");
    if (!f) { fprintf(stderr, "rgf_pack: cannot write %s\n", out); free(blob); return 1; }

    uint32_t crc = 0;
    /* signature: magic + format_version (part of CRC coverage) */
    fwrite(RGF_MAGIC, 1, 4, f);              crc = crc32_update(crc, (const uint8_t*)RGF_MAGIC, 4);
    uint32_t ver = RGF_FORMAT_VERSION;
    fwrite(&ver, 4, 1, f);                   crc = crc32_update(crc, (const uint8_t*)&ver, 4);

    char buf[RGF_TEXT_MAX_BYTES];
    int m;
    m = snprintf(buf, sizeof(buf), "title=%s\nauthor=%s\norganism=%s\ncorpus=%s\ncreated=build\n",
                 title, author, organism, corpus);
    if (m < 0 || (size_t)m >= sizeof(buf)) { fprintf(stderr, "rgf_pack: MANI overflow\n"); fclose(f); free(blob); return 1; }
    write_chunk(f, &crc, RGF_TAG_MANI, buf, (uint32_t)m);

    m = snprintf(buf, sizeof(buf),
                 "arch=%s\nvocab=%d\nembed=%d\nheads=%d\nffn=%d\nctx=%d\nlayers=%d\nt_max=%d\ndtype=f32\n",
                 organism, RGF_V, RGF_E, RGF_HEADS, RGF_FFN, RGF_CTX, RGF_N_LAYERS, RGF_T_MAX);
    if (m < 0 || (size_t)m >= sizeof(buf)) { fprintf(stderr, "rgf_pack: MCFG overflow\n"); fclose(f); free(blob); return 1; }
    write_chunk(f, &crc, RGF_TAG_MCFG, buf, (uint32_t)m);

    m = snprintf(buf, sizeof(buf), "steps=%d\ntemperature=%.3f\nremask=%.3f\nhold_ms=2000\n", steps, temp, remask);
    write_chunk(f, &crc, RGF_TAG_SMPL, buf, (uint32_t)m);

    m = snprintf(buf, sizeof(buf), "mode=NOISE_TO_TEXT\npalette=blood\nfps=12\n");
    write_chunk(f, &crc, RGF_TAG_RNDR, buf, (uint32_t)m);

    m = snprintf(buf, sizeof(buf), "policy=%s\nvalue=%ld\n", seed_policy, seed_value);
    write_chunk(f, &crc, RGF_TAG_SEED, buf, (uint32_t)m);

    write_chunk(f, &crc, RGF_TAG_WGHT, blob, (uint32_t)blob_len);

    /* CRC0 chunk carries the crc of everything written before it. */
    uint32_t crc_hdr[2] = { RGF_TAG_CRC0, 4 };
    fwrite(crc_hdr, 4, 2, f);
    fwrite(&crc, 4, 1, f);

    write_chunk(f, &crc, RGF_TAG_END, NULL, 0);

    fclose(f);
    free(blob);
    printf("rgf_pack: wrote %s (%zu-byte blob, %d tensors validated, crc=0x%08X)\n",
           out, blob_len, RGF_N_TENSORS, crc);
    return 0;
}
