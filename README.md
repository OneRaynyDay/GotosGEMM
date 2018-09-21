# GotosGEMM

**Implemented GEMM via GEPP & GEBP, which is the fastest theoretical GEMM variant according to Kazushige Goto.** To use stack-allocated, fixed size arrays in C++, use:

    static_gemm.hpp
    
For heap-allocated, dynamic size arrays in C++, use:

    dynamic_gemm.hpp
    
To reproduce the benchmarks, set the macro preprocessor directive `DYNAMIC` on or off inside of `main.cpp`.

The below times are measured on the following platform:

    Intel(R) Core(TM) i7-6660U CPU @ 2.40GHz

which runs `AVX2` instructions. Compiled using gcc-8, version 8.2.0, on a Macbook Pro.

---

## Results

Against Apple's Accelerate framework's BLAS:

    $ ./main
    Time elapsed : 37.4626 (naive)
    Time elapsed : 1.67072 (gotos impl)
    Time elapsed : 1.5553 (Apple BLAS)

On a docker container with `ATLAS` installed, linked to cblas:

    $ ./main
    Time elapsed : 74.8051 (naive)
    Time elapsed : 1.93216 (gotos impl)
    Time elapsed : 3.06775 (ATLAS)
    
proof it's linked to `ATLAS`'s cblas:

    $ ldd main
	linux-vdso.so.1 (0x00007fffcbbd5000)
	libcblas.so.3 => /usr/lib/x86_64-linux-gnu/libcblas.so.3 (0x00007f1005cc8000)
	libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007f100593a000)
	libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f100559c000)
	libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007f1005384000)
	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007f1004f93000)
	libatlas.so.3 => /usr/lib/x86_64-linux-gnu/libatlas.so.3 (0x00007f1004a0a000)
	/lib64/ld-linux-x86-64.so.2 (0x00007f100f9da000)

## Extra Observations

I tried playing with gebp's block size, and it seems like for 2048x1024 x 1024 x 2048 matrices, a single float seems to be the best for performance under `-O2` and no vectorized instructions. This is a bit strange, since I thought bigger blocks could reside inside the tlb/L1/L2 caches.

    1 float:

    Time elapsed : 2.33989

    8 floats:
    
    Time elapsed : 4.34408
    
    16 floats:
    
    Time elapsed : 4.522
    
    32 floats:
    
    Time elapsed : 6.42583
    
    64 floats:
    
    Time elapsed : 7.35193
    
    128 floats:
    
    Time elapsed : 8.18502
    
    256 floats:
    
    Time elapsed : 8.1686

However, here's something _very interesting, with vectorized instructions:_

    1 float:
    
    Time elapsed : 1.3284
    
    2 floats:
    
    Time elapsed : 1.62332
    
    4 floats:
    
    Time elapsed : 0.421444
    
    8 floats:
    
    Time elapsed : 0.527425
    
    16 floats:
    
    Time elapsed : 4.03877

... and it gets worse from here. It seems like the **gcc auto-vectorizer seems to have aggressively optimized for special cases of 4 and 8 for AVX and AVX2 instructions.** 

The performance of this header under `-O3` aggressive optimization flags rivals that of ATLAS.
