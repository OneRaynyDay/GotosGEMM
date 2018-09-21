# GotosGEMM

**Implemented GEMM via GEPP & GEBP, which is the fastest theoretical GEMM variant according to Kazushige Goto.**

The below times are measured on the following platform:

    Intel(R) Core(TM) i7-6660U CPU @ 2.40GHz

which runs `AVX2` instructions.

---

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
