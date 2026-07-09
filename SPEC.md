# RGF v0.1 — Format Specification

Little-endian, chunked (in the spirit of GIF89a: signature + blocks). Extension
`.rgf`, MIME `application/x-rgf`. The reference decoder is `rgf_viewer.c`; the
packer is `rgf_pack.c`. Model constants and the tensor shape table live in `rgf.h`.

## Container

```
offset 0:  magic  "RGF1"            (4 bytes)
offset 4:  u32    format_version    (= 1)
then a sequence of chunks:  u32 tag (FourCC) + u32 length + payload[length]
```

The signature (magic + version) and every chunk up to `CRC0` are covered by the
CRC. `CRC0` stores that value; the decoder verifies it **before** using the weight
blob. Unknown tags are skipped by length (forward-compat) — but only after the CRC
check passes.

## Chunks

| tag    | payload                                                                 | required |
|--------|-------------------------------------------------------------------------|----------|
| `MANI` | manifest, `key=value\n`: title, author, organism, corpus, created        | yes |
| `MCFG` | model config: arch, vocab, embed, heads, ffn, ctx, layers, t_max, dtype  | yes |
| `SMPL` | sampler: steps, temperature, remask, hold_ms                             | yes |
| `RNDR` | render: mode (`NOISE_TO_TEXT`), palette, fps                             | yes |
| `SEED` | seed policy: policy (`time`\|`fixed`\|`random`), value                    | yes |
| `WGHT` | weight blob — verbatim notorch dump (magic `0x4E544F52` + 60 tensors)     | yes |
| `CRC0` | u32 crc32 (IEEE, poly `0xEDB88320`) of all bytes before this chunk        | yes |
| `END`  | terminator, length 0                                                      | yes |

Text chunks are plain `key=value\n` (no JSON — trivial to parse in C, readable via
`strings`). `dtype` in `MCFG` is present from birth: v0.1 writes `f32` (~15 MB),
v0.2 adds `q8` (~3.8 MB) without breaking the format.

## Untrusted-input hardening (viewer, format-level)

`.rgf` opens with other people's bytes. The parser enforces:

- magic + version check;
- per-chunk length caps: `WGHT ≤ 64 MB`, text chunks `≤ 64 KB`;
- no read past end of buffer;
- `CRC0` verified before the weight blob is used;
- forced NUL-termination on text chunks;
- the weight loader guards `ndim` and every shape (the engine's `load_mat`/`load_vec`).

## Weight blob & shape table (format-level validation)

The `WGHT` payload is a verbatim notorch dump: `u32 magic (0x4E544F52)`, `i32 count`
(must be `60`), then per tensor `i32 ndim`, `i32 shape[ndim]`, `f32 data[∏shape]`.
`rgf_pack` validates the blob by a full walk against `RGF_TENSOR_SHAPES` in `rgf.h`
— the architecture-fixed shape of all 60 tensors, in `load_weights` order:

```
wte[256,192] wpe[128,192] t_proj1[192,192] t_proj2[192,192]
× 6 layers: rms1[192] wq/wk/wv/wo[192,192] rms2[192] w_gate/w_up[768,192] w_down[192,768]
rms_f[192] head[256,192]
```

A blob with a mismatched shape is refused, not sealed into the format.

## Lifecycle (viewer)

```
parse container → verify CRC → load WGHT from MEMORY (not a path)
→ seed by policy → tokens[ctx] = all MASK → denoise steps (each step = a frame)
→ hold_ms → re-mask (remask × ctx) random positions → denoise again → … while open
```

Re-masking a fraction (not a full reset) gives continuous mutation without repeat —
the anti-GIF. `--once` does one crystallisation and prints the final text
(pipe-friendly proof). `--dump` prints the parsed chunks (proof the model is from
the file).

Part of the Arianna Method.
