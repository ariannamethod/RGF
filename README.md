# RGF — Resonant Generative Format

A file format for living images. An `.rgf` stores the **conditions for the birth
of frames** — the weights of a tiny text-diffusion model plus config — not the
frames themselves. A native decoder denoises text out of noise: each denoise step
is a frame. **GIF repeats, RGF resonates.**

## What this is

GIF stores frames. RGF stores what grows them. Open an `.rgf` with its decoder and
a model inside the file crystallises text from noise, holds, re-masks a fraction of
positions, and denoises again — the file breathes, it is not a loop. Each open is a
different birth (time-seeded).

The format carries **no executable code** — only weight and config chunks. The
renderer mode is an enum, not a script; the viewer as a class has no network, fs, or
shell access. Sandbox by construction.

## Build & run

```
make                                       # rgf_pack + rgf_viewer
./rgf_pack --weights diffusion.bin --out dracula.rgf --title "Dracula Resonance"
./rgf_viewer dracula.rgf                    # live: text crystallises from noise, breathes
./rgf_viewer dracula.rgf --once             # one crystallisation, final text to stdout
./rgf_viewer dracula.rgf --dump             # print parsed chunks (the model IS in the file)
make asan                                   # AddressSanitizer build for fuzzing the parser
```

An `.rgf` is not self-executing in v0.1 — it needs its decoder, like a GIF needs a
GIF decoder. The difference: the decoder does not replay frames, it **generates**
them.

## Status — v0.1 (proof-of-life)

The full vertical is verified end-to-end on the real Dracula Diffusion model (3.74M
params, byte-level, trained to loss ~0.08 — deep overfit by design): `train →
weights → pack → .rgf → viewer → parse → load-from-memory → denoise → visual`. The
parser is hardened against untrusted input (magic, tag-whitelist, length caps, CRC32
verified before the blob is used, ndim guards on the weight loader) — corrupt files
are refused, never crashed (fuzz-checked under ASan).

**Honest scope:** the denoised text is *fabric, not prose* — an English-textured
ripple of the corpus, not readable sentences. This is proof-of-life of the medium
(the denoise-animation is alive), not literature. Coherence is a tuning/scale
question, not a claim this release makes.

## Spec

See [SPEC.md](SPEC.md) — container layout, chunk table, the 60-tensor shape table
the packer validates against.

## Roadmap

- **v0.2** — q8-quantised WGHT (~3.8 MB vs ~15 MB f32), SDL-window render backend,
  coherence tuning (sampler beyond argmax).
- **v0.3** — RGF-HEBREW: γ-guidance (θ=ε+γ+αδ) in the denoise, RTL render.
- **v2** — RGF-VLM: a poster frame in the file is read by the file's own VLM; the
  caption is born from what the model sees in itself. A closed loop:
  image → perception → speech.
- A second reference decoder: the browser engine reads the same `.rgf`.

Part of the Arianna Method. Co-authored by Oleg Ataeff and Claude.
