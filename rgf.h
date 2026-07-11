/*
 * rgf.h — Resonant Generative Format v0.1: constants + tensor shape table.
 *
 * RGF stores the conditions for the birth of frames (mini-diffusion weights +
 * config), not the frames. A native decoder denoises text from noise — each
 * denoise step is a frame. "GIF repeats, RGF resonates."
 *
 * This header is the contract: the model config mirrors the Dracula Diffusion
 * engine (diffusion_engine.c) and RGF_TENSOR_SHAPES is the exact, architecture-
 * fixed shape table for all 60 tensors. Validation compares against it — it does
 * not guess. A crafted/corrupt .rgf cannot smuggle a mismatched blob past this.
 */

#ifndef RGF_H
#define RGF_H

#include <stdint.h>

/* ── Container ───────────────────────────────────────────────────────────── */

#define RGF_MAGIC          "RGF1"        /* 4 bytes at offset 0 */
#define RGF_FORMAT_VERSION 1u

/* Chunk tags (FourCC, little-endian layout: u32 tag + u32 length + payload). */
#define RGF_TAG_MANI 0x494E414D  /* "MANI" — manifest (title/author/organism/corpus/created) */
#define RGF_TAG_MCFG 0x4746434D  /* "MCFG" — model config */
#define RGF_TAG_SMPL 0x4C504D53  /* "SMPL" — sampler (steps/temp/remask/hold_ms) */
#define RGF_TAG_RNDR 0x52444E52  /* "RNDR" — render (mode/palette/fps) */
#define RGF_TAG_SEED 0x44454553  /* "SEED" — seed policy */
#define RGF_TAG_MRGS 0x5347524D  /* "MRGS" — BPE merges (integer "id_a id_b" text, verbatim) */
#define RGF_TAG_WGHT 0x54484757  /* "WGHT" — weight blob (verbatim notorch dump) */
#define RGF_TAG_CRC0 0x30435243  /* "CRC0" — u32 crc32 of all bytes before this chunk */
#define RGF_TAG_END  0x00444E45  /* "END\0" — terminator (length 0) */

/* Chunk length caps — untrusted input, refuse absurd sizes (Fable §4.2). */
#define RGF_WGHT_MAX_BYTES (64u * 1024u * 1024u)  /* 64 MB — f32 blob ~15 MB, headroom */
#define RGF_TEXT_MAX_BYTES (64u * 1024u)          /* 64 KB — text chunks */
#define RGF_MRGS_MAX_BYTES (256u * 1024u)         /* 256 KB — BPE merges text (1792 merges ~14 KB, headroom) */

/* ── Model config (MUST match diffusion_engine.c D_* — mirrored, verified) ── */

#define RGF_V         2049  /* vocab (2048 BPE merges-vocab + MASK)   */
#define RGF_E         288   /* embedding dim                         */
#define RGF_HEADS     6     /* attention heads                       */
#define RGF_FFN       1152  /* feed-forward dim                      */
#define RGF_CTX       128   /* context length (bytes)                */
#define RGF_N_LAYERS  6     /* transformer layers                    */
#define RGF_T_MAX     1000  /* diffusion timesteps                   */
#define RGF_N_TENSORS 60    /* = 2 + 2 + N_LAYERS*9 + 2              */

/* Weight blob magic (notorch dump: magic + i32 count + per-tensor). */
#define RGF_WGHT_MAGIC 0x4E544F52u

/* ── Tensor shape table (exact, order == diffusion_engine.c load_weights) ─── */

typedef struct { int ndim; int dims[2]; } rgf_shape;

#define RGF_MAT(r, c) { 2, { (r), (c) } }
#define RGF_VEC(n)    { 1, { (n), 0 } }

/* 60 tensors: [wte wpe t_proj1 t_proj2] + 6×[rms1 wq wk wv wo rms2 w_gate w_up w_down] + [rms_f head] */
static const rgf_shape RGF_TENSOR_SHAPES[RGF_N_TENSORS] = {
    RGF_MAT(2049, 288), /* wte   [V, E]   */
    RGF_MAT(128, 288),  /* wpe   [CTX, E] */
    RGF_MAT(288, 288),  /* t_proj1 [E, E] */
    RGF_MAT(288, 288),  /* t_proj2 [E, E] */
    /* layer 0 */ RGF_VEC(288), RGF_MAT(288,288), RGF_MAT(288,288), RGF_MAT(288,288), RGF_MAT(288,288), RGF_VEC(288), RGF_MAT(1152,288), RGF_MAT(1152,288), RGF_MAT(288,1152),
    /* layer 1 */ RGF_VEC(288), RGF_MAT(288,288), RGF_MAT(288,288), RGF_MAT(288,288), RGF_MAT(288,288), RGF_VEC(288), RGF_MAT(1152,288), RGF_MAT(1152,288), RGF_MAT(288,1152),
    /* layer 2 */ RGF_VEC(288), RGF_MAT(288,288), RGF_MAT(288,288), RGF_MAT(288,288), RGF_MAT(288,288), RGF_VEC(288), RGF_MAT(1152,288), RGF_MAT(1152,288), RGF_MAT(288,1152),
    /* layer 3 */ RGF_VEC(288), RGF_MAT(288,288), RGF_MAT(288,288), RGF_MAT(288,288), RGF_MAT(288,288), RGF_VEC(288), RGF_MAT(1152,288), RGF_MAT(1152,288), RGF_MAT(288,1152),
    /* layer 4 */ RGF_VEC(288), RGF_MAT(288,288), RGF_MAT(288,288), RGF_MAT(288,288), RGF_MAT(288,288), RGF_VEC(288), RGF_MAT(1152,288), RGF_MAT(1152,288), RGF_MAT(288,1152),
    /* layer 5 */ RGF_VEC(288), RGF_MAT(288,288), RGF_MAT(288,288), RGF_MAT(288,288), RGF_MAT(288,288), RGF_VEC(288), RGF_MAT(1152,288), RGF_MAT(1152,288), RGF_MAT(288,1152),
    RGF_VEC(288),       /* rms_f [E]     */
    RGF_MAT(2049, 288), /* head  [V, E]  */
};

#endif /* RGF_H */
