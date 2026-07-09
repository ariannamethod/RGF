# RGF — Resonant Generative Format. Build.
CC      = cc
CFLAGS  = -O2 -Wall -Wextra -std=c11
LDFLAGS = -lm
ENGINE  = vendor/diffusion_engine.c

.PHONY: all pack viewer asan clean
all: rgf_pack rgf_viewer

pack:   rgf_pack
viewer: rgf_viewer

rgf_pack: rgf_pack.c rgf.h
	$(CC) $(CFLAGS) -I. rgf_pack.c $(LDFLAGS) -o rgf_pack

rgf_viewer: rgf_viewer.c rgf.h $(ENGINE)
	$(CC) $(CFLAGS) -DDIFFUSION_LIB_ONLY -I. -Ivendor rgf_viewer.c $(ENGINE) $(LDFLAGS) -o rgf_viewer

# AddressSanitizer build of the viewer — for fuzzing the .rgf parser.
asan: rgf_viewer.c rgf.h $(ENGINE)
	$(CC) -O1 -g -fsanitize=address -DDIFFUSION_LIB_ONLY -I. -Ivendor rgf_viewer.c $(ENGINE) $(LDFLAGS) -o rgf_viewer_asan

clean:
	rm -f rgf_pack rgf_viewer rgf_viewer_asan
