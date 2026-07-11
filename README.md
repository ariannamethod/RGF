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

The full vertical is proven end-to-end (`train → weights → pack → .rgf → viewer →
parse → load-from-memory → BPE-decode → denoise → visual`) on a real BPE Dracula
Diffusion model (V=2049 = 2048 merges + MASK, E288/FFN1152/6L, ~9.35M params). The
container carries its own tokenizer (`MRGS` chunk = the merge table), so the viewer
detokenises without any external file. A passage-overfit demo model **reveals a
recognisable Dracula passage from pure noise** — the memorised skeleton reads through
the MaskGIT denoise (*"which I got of it from the train … from the station, as we …
depth, took us am"*). The parser is hardened against untrusted input (magic,
tag-whitelist, length caps, CRC32 verified before use, `MRGS`/`WGHT` rejected after
`CRC0` or when duplicated, merges validated causal so decode can't recurse unboundedly)
— corrupt files are refused, never crashed.

**Honest scope:** the demo model is *overfit on one passage* — it breathes that
passage, recognisably, with cold-start filler noise where the reveal commits before
context accrues. Cleaner prose is a seeding / longer-training knob; full-corpus
coherence is a separate training-scale question (a byte/small model plateaus at the
marginal until it is trained past it — the wall was a permanent optimiser freeze,
now removed, plus scale). What v0.1 proves: the medium is real — conditioned weights
in a file denoise into legible text, and the container is a hardened, self-contained
tokeniser+model that any decoder can breathe.

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
