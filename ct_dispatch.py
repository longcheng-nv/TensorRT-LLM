# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-Python transcription of the frozen GVR CUDA dispatch (gvr_topk_launch).

Source of truth: ../src_cuda/kernel.cu (3197 lines).  route(b, n, npad, k) is a
PURE function of its four ints -- no env knobs, no GPU, stdlib only.

Branch map (kernel.cu line citations):
  constants          NB L16, QUADC L21, SNB L170, CMPC L2372, BLKC L2374
  reg-block prologue L2757-2822: wide=(b<=148) L2757; n4=n>>2 L2759;
                     CMP=min(n,2560) L2764; QC=(b>148?1024:QUADC) L2768;
                     CURE L2775; DEGE L2788; DEG widens CMP to n L2791;
                     NBSEL L2820; IMGOFF=NBSEL L2821; smem=(NBSEL+2*CMP)*4 L2822
  LAUNCH_REG2/DEG/REG macros L2823-2847 (KPT ladder 1/2/4; DEG forces KPT=1,
                     CUR=CURE both places); IMGW/smi/IMGE L2852-2854;
                     LAUNCH_REGIMG L2861-2863 -> gvr_topk_reg<...,KPT=1,CUR=true,
                     DEG=false,IMG=true,NBH=2*NB> via launch_regimg L2672-2686
  n4 rungs           n4<=256 L2864; n4<=512 L2865; n4<=1024 wide/img/else L2866-2884
  clustered reg path L2897-2940: gate n4>4096 && n4<=8*BLKC*4 && k<=BLKC L2897;
                     av/amax L2898-2899; two-pass cs=8 co-residency veto
                     (pass==0 && c==8 && b>15 -> skip) L2917-2923; 64-bit product
                     (long long)c*BLKC*v < n4 L2919; smc=(3*NB+2*CMPC)*4 L2926;
                     grid dim3(cs,b) L2666
  wide 4k fallback   n4<=4096 && wide -> LAUNCH_REG(1024,4,1,2*NB) L2945-2947
  streaming R        L2959-2975 (b<=32: R=min(148/b, ((n>>2)+1023)/1024));
                     r11 shallow split b<=74 && n4>=16384 && k<=1024 -> R=2 L2985;
                     cluster clamp R->pow2, useclus, only if 2<=R<=8 && k<=1024 L2994
  big/SCAP/CMP       big=(b*R<=148) L2995; SCAP L3009-3010; CMP L3011
  aim                L3039-3040; sqrt floor r=int(0.5+sqrt(6LL*n)) L3041-3042;
                     SFAC L3072-3073; amin L3079-3080; clamps L3081-3082
  sample geometry    small_dense gate L3091 ((k>1024)&&!big&&n<=SCAP&&n>2*k);
                     PAIR form (sel>>3, half=n4s>>1, SMP*8) L3092-3109;
                     clus QUAD override (sel>>4, quarter=n4s>>2, SMP*16),
                     gated n>SCAP only, L3115-3128
  Q                  Q=(n4s+R-1)/R L3110
  clus launch        smc=SNB*8+(SCAP+4)*8+CMP*8 L3130; U ladder per=Q>>10
                     L3132-3142; CS=R in {2,4,8} L3143-3145; grid dim3(CS,b) L2704
  main launch        smem=(SCAP+4)*((R>1||b<=296)?8:4)+(CMP+1)*8 L3149;
                     KPT ladder 1/2/4/8 L3150-3169; big: per=Q>>10 U ladder,
                     SPLIT=(R>1), grid dim3(R,b) L3173-3185 + L2750;
                     b<=296 -> (512,2,8,false) L3193; else (256,4,8,false) L3194

rt carries the FULL runtime scalar list each kernel receives, in signature
order, always starting with (n, npad, k) -- every launch site passes them
(L2666-2667 reg_clus, L2684-2685 regimg, L2704-2705 clus, L2726-2727 reg,
L2750-2751 main).  [dispatch x-check 2026-08-13: rt previously omitted the
leading n/npad/k; fixed for full-ABI parity with the independent spec
transcription.]

Dead ABI-parity args: gvr_main's 7th/8th params are declared `int SCAP_, int CMP_`
(kernel.cu L381) and are NEVER read by the kernel body -- it recomputes SCPB/CMPB
as constexprs of (BLK, SPLIT, KBIG) (L413-424) that mirror the host formulas
bit-identically.  They are kept in rt under their source names 'SCAP_'/'CMP_'
purely for ABI parity.  gvr_clus's SCAP/CMP (L1798) are LIVE runtime args.
`aim` and `SFAC` are host-side intermediates only (never cross the ABI), so they
do not appear in rt.

C-semantics notes encoded here:
  * every `/` on ints is C truncating division -> Python `//` (all operands
    are non-negative on every reachable path);
  * `sel = (long long)SFAC * n / aim` and the TGT/TGT2 products are 64-bit in C;
    Python ints are exact, so `//` reproduces them;
  * `int r = (int)(0.5 + sqrt((double)(6LL*n)))` truncates toward zero after
    the +0.5 -> `int(0.5 + math.sqrt(float(6*n)))`;
  * `IMGW = (n + 3) & ~3` four-element float4 round-up;
  * the reg-block CMP (possibly widened to n by DEGE) is scoped to the braces
    at L2758-2949; the streaming path re-derives its own CMP.
"""

import math

# ---- constants lifted from kernel.cu ---------------------------------------
NB = 1024        # L16   register-path histogram bins
QUADC = 96       # L21   crossing-bin O(mc^2) rank gate (streaming/reg paths)
SNB = 256        # L170  streaming-path bin count
CMPC = 4096      # L2372 crossing-bin slots per CTA, clustered register path
BLKC = 1024      # L2374 CTA size of the clustered register path


def route(b, n, npad, k):
    """Mirror of gvr_topk_launch (kernel.cu L2754-3197). Pure. See module doc."""
    wide = b <= 148                                             # L2757

    # ================= register-resident block (L2758-2949) =================
    n4 = n >> 2                                                 # L2759
    CMP = n if n < 2560 else 2560                               # L2764
    QC = 1024 if b > 148 else QUADC                             # L2768
    CURE = not (n < 2 * k and b > 148)                          # L2775
    DEGE = (n <= 3 * k) or (n <= 4 * k + 64)                    # L2788
    if DEGE and CMP < n:                                        # L2791
        CMP = n
    NBSEL = (2 * NB) if (n4 > 512 and not (n4 <= 1024 and not wide)) else NB  # L2820
    IMGOFF = NBSEL                                              # L2821
    smem_reg = (NBSEL + 2 * CMP) * 4                            # L2822

    def _reg(BLK, VPT, MINB, NBH):
        # LAUNCH_REG (L2844-2847): DEG wins, else CUR flag; KPT ladder L2823-2834.
        if DEGE:
            tpl = (BLK, VPT, MINB, 1, CURE, True, False, NBH)   # LAUNCH_DEG L2836-2843
        else:
            kpt = 1 if k <= BLK else (2 if k <= 2 * BLK else 4)
            tpl = (BLK, VPT, MINB, kpt, CURE, False, False, NBH)
        return {
            'kernel': 'reg', 'tpl': tpl,
            'rt': {'n': n, 'npad': npad, 'k': k,                # L2726-2727 full ABI
                   'CMP': CMP, 'IMGOFF': IMGOFF, 'QC': QC},
            'grid': (b, 1), 'cluster': 1, 'block': BLK,
            'smem': smem_reg, 'ws': False,
        }

    IMGW = (n + 3) & ~3                                         # L2852
    smi = (NBSEL + (2 * CMP if 2 * CMP > IMGW else IMGW)) * 4   # L2853
    IMGE = wide and (not DEGE) and k <= 1024                    # L2854

    if n4 <= 256:                                               # L2864
        return _reg(256, 1, 8, NB)
    if n4 <= 512:                                               # L2865
        return _reg(512, 1, 4, NB)
    if n4 <= 1024:                                              # L2866-2884
        if wide:
            if IMGE:                                            # LAUNCH_REGIMG(1024,1,2) L2872
                # launch_regimg<1024,1,2,NBV=2*NB,KPTV=1> -> gvr_topk_reg
                # <1024,1,2,1,true,false,true,2048>  (L2672-2686)
                return {
                    'kernel': 'regimg',
                    'tpl': (1024, 1, 2, 1, True, False, True, 2 * NB),
                    'rt': {'n': n, 'npad': npad, 'k': k,        # L2684-2685 full ABI
                           'CMP': CMP, 'IMGOFF': IMGOFF, 'QC': QC},
                    'grid': (b, 1), 'cluster': 1, 'block': 1024,
                    'smem': smi, 'ws': False,
                }
            return _reg(1024, 1, 2, 2 * NB)                     # L2872 else-arm
        return _reg(512, 2, 4, NB)                              # L2883

    # ---- clustered register-resident path (L2897-2940) ----
    if n4 > 4096 and n4 <= 8 * BLKC * 4 and k <= BLKC:          # L2897
        av = 148 // (b if b > 0 else 1)                         # L2898 truncating
        amax = 1                                                # L2899
        while (amax << 1) <= av and amax < 8:
            amax <<= 1
        vsel = 0
        cs = 0
        if amax >= 2:                                           # L2901
            # knife5 (layer 9): UNCONDITIONAL cs=8 co-residency veto --
            # the L2w pass-1 rescue is deleted; 512k b>15 falls through to
            # streaming, made retry-safe by TSH-floor staging (S1) and the
            # gvr_clus veto (S2).
            for v in (1, 2, 4):
                c = 1                                           # 64-bit product
                while c * BLKC * v < n4:
                    c <<= 1
                if c == 8 and b > 15:                           # THE VETO
                    continue
                if c <= amax:
                    vsel = v
                    cs = c
                    break
        if vsel and cs >= 2:                                    # L2925
            smc = (3 * NB + 2 * CMPC) * 4                       # L2926
            return {
                'kernel': 'reg_clus', 'tpl': (BLKC, vsel, cs),
                'rt': {'n': n, 'npad': npad, 'k': k},           # dims only, L2666-2667
                'grid': (cs, b), 'cluster': cs, 'block': BLKC,
                'smem': smc, 'ws': False,
            }

    if n4 <= 4096 and wide:                                     # L2945-2947
        return _reg(1024, 4, 1, 2 * NB)

    # ================= streaming / collect path (L2950-3196) =================
    R = 1                                                       # L2959
    if b <= 32:                                                 # L2960-2975
        r1 = 148 // b
        if r1 < 1:
            r1 = 1
        r2 = ((n >> 2) + 1023) // 1024                          # L2972
        if r2 < 1:
            r2 = 1
        R = r1 if r1 < r2 else r2
        if R < 1:
            R = 1
    elif b <= 74 and (n >> 2) >= 16384 and k <= 1024:           # L2985 r11 split
        R = 2

    useclus = False                                             # L2993-2994
    if 2 <= R <= 8 and k <= 1024:
        p2 = 1
        while (p2 << 1) <= R:
            p2 <<= 1
        # knife5 (layer 8): gvr_clus cs=8 hits the same GPC packing wall as
        # the clustered register path; same veto, same b>15 threshold.
        if p2 == 8 and b > 15:
            p2 = 4
        R = p2
        useclus = True

    big = b * R <= 148                                          # L2995
    SCAP = (16384 if R == 1 else 8192) if big \
        else (8192 if k > 1024 else 4096)                       # L3009-3010
    CMP = (4096 if k > 1024 else 2048) if big else 1024         # L3011

    aim = ((4 * k if k >= 1024 else 2 * k) if R == 1 else 2 * k) if big \
        else ((11 * k) // 8 if k >= 1024 else (3 * k) // 2)     # L3039-3040
    q = 6 * n                                                   # L3041: 6LL * n
    r = int(0.5 + math.sqrt(float(q)))                          # L3041 C cast trunc
    if r > aim:                                                 # L3042
        aim = r
    SFAC = (32 if R == 2 else (48 if k > 1024 else 16)) if R > 1 \
        else (64 if k >= 1024 else 32)                          # L3072-3073
    amin = 3 * k if R == 2 else (7 * k) // 2                    # L3079
    if R > 1 and aim < amin:                                    # L3080
        aim = amin
    if aim > (SCAP >> 1):                                       # L3081
        aim = SCAP >> 1
    if aim < k:                                                 # L3082
        aim = k

    n4s = n >> 2                                                # L3084
    SMP, SS2, TGT, TGT2 = 0, 1, 0, 0                            # L3085
    small_dense = (k > 1024) and (not big) and n <= SCAP and n > 2 * k  # L3091
    if (n > SCAP or small_dense) and n4s >= 4:                  # L3092: PAIR sample
        sel = SFAC * n // aim                                   # L3095 64-bit
        if sel < 256:                                           # L3096
            sel = 256
        if sel > n // 2:                                        # L3097
            sel = n // 2
        pairs = sel >> 3                                        # L3098
        if pairs < 1:
            pairs = 1
        half = n4s >> 1                                         # L3099
        if half < 1:
            half = 1
        if pairs > half:                                        # L3100
            pairs = half
        SS2 = half // pairs                                     # L3101
        if SS2 < 1:
            SS2 = 1
        SMP = half // SS2                                       # L3102
        if SMP < 1:
            SMP = 1
        TGT = (aim * (SMP * 8)) // n                            # L3103 64-bit
        if TGT < 1:                                             # L3104
            TGT = 1
        TGT2 = (k * (SMP * 8)) // n                             # L3107 64-bit
        if TGT2 < 1:                                            # L3108
            TGT2 = 1
    Q = (n4s + R - 1) // R                                      # L3110

    if useclus:                                                 # L3111-3147
        if n > SCAP and n4s >= 4:                               # L3115: QUAD override
            sel = SFAC * n // aim                               # L3116
            if sel < 256:
                sel = 256
            if sel > n // 2:
                sel = n // 2
            quads = sel >> 4                                    # L3119
            if quads < 1:
                quads = 1
            quarter = n4s >> 2                                  # L3120
            if quarter < 1:
                quarter = 1
            if quads > quarter:                                 # L3121
                quads = quarter
            SS2 = quarter // quads                              # L3122
            if SS2 < 1:
                SS2 = 1
            SMP = quarter // SS2                                # L3123
            if SMP < 1:
                SMP = 1
            TGT = (aim * (SMP * 16)) // n                       # L3124
            if TGT < 1:
                TGT = 1
            TGT2 = (k * (SMP * 16)) // n                        # L3126
            if TGT2 < 1:
                TGT2 = 1
        smc = SNB * 8 + (SCAP + 4) * 8 + CMP * 8                # L3130
        per = Q >> 10                                           # L3131
        U = 8 if per >= 8 else (4 if per >= 4 else (2 if per >= 2 else 1))  # L3134-3141
        CS = 2 if R == 2 else (4 if R == 4 else 8)              # L3143-3145
        return {
            'kernel': 'clus', 'tpl': (1024, U, 1, SNB, CS),
            'rt': {'n': n, 'npad': npad, 'k': k,                # L2704-2705 ABI (live)
                   'SCAP': SCAP, 'CMP': CMP, 'SMP': SMP, 'TGT': TGT,
                   'Q': Q, 'SS2': SS2, 'TGT2': TGT2},
            'grid': (CS, b), 'cluster': CS, 'block': 1024,
            'smem': smc, 'ws': False,
        }

    smem_main = (SCAP + 4) * (8 if (R > 1 or b <= 296) else 4) + (CMP + 1) * 8  # L3149

    def _main(BLK, MINB, U, SPLIT):
        # LAUNCH_MAIN KPT ladder 1/2/4/8 (L3150-3169); grid dim3(gx=R, gy=b) L2750.
        kpt = 1 if k <= BLK else (2 if k <= 2 * BLK else (4 if k <= 4 * BLK else 8))
        # knife5 (layer 7) TSH-floor staging gate.  CUDA form: grid-uniform
        # RUNTIME gate gridDim.y > 15 && k <= 1024 && (n >> 2) <= 32768 with
        # a dual scan-instantiation branch.  Here: compile-time key -- the
        # ungated variant IS the pre-knife5 kernel; per-launch semantics are
        # identical because the gate is uniform over the grid.
        tshg = bool(SPLIT) and b > 15 and k <= 1024 and (n >> 2) <= 32768
        return {
            'kernel': 'main', 'tpl': (BLK, U, MINB, SNB, kpt, SPLIT, tshg),
            # SCAP_/CMP_ are DEAD ABI-parity args: gvr_main (L381) never reads
            # them, it uses constexpr SCPB/CMPB (L413-424).  Kept for ABI parity.
            'rt': {'n': n, 'npad': npad, 'k': k,                # L2750-2751 full ABI
                   'SCAP_': SCAP, 'CMP_': CMP, 'R': R, 'SMP': SMP, 'TGT': TGT,
                   'Q': Q, 'SS2': SS2, 'TGT2': TGT2},
            'grid': (R, b), 'cluster': 1, 'block': BLK,
            'smem': smem_main, 'ws': True,
        }

    if big:                                                     # L3173-3185
        per = Q >> 10                                           # L3174
        U = 8 if per >= 8 else (4 if per >= 4 else (2 if per >= 2 else 1))
        return _main(1024, 1, U, R > 1)                         # SPLIT iff R>1
    if b <= 296:                                                # L3193
        return _main(512, 2, 8, False)
    return _main(256, 4, 8, False)                              # L3194


if __name__ == '__main__':
    smoke = [
        # (b, n, npad, k)                          expected family
        (64,   1024,    1024,    512),   # reg   n4<=256 rung (DEG: n<=3k)
        (64,   2048,    2048,    512),   # reg   n4<=512 rung
        (1024, 4096,    4096,    1024),  # reg   n4<=1024, b>148 -> (512,2,4)
        (64,   4096,    4096,    512),   # regimg wide !DEGE k<=1024
        (64,   4096,    4096,    1024),  # reg   wide but DEGE (n<=4k+64)
        (8,    65536,   65536,   1024),  # reg_clus (vsel=2, cs=8; b<=15 no veto)
        (16,   131072,  131072,  512),   # knife5: veto fall-through -> SPLIT slab, tshg=True
        (64,   16384,   16384,   1024),  # reg   wide 4k fallback (1024,4,1)
        (64,   262144,  262144,  1024),  # clus  r11 R=2 shallow cluster split
        (1,    1048576, 1048576, 1024),  # main  deep slab SPLIT R=148
        (20,   262144,  262144,  2048),  # main  k>1024 split (no useclus)
        (512,  131072,  131072,  1024),  # main  b>296 BLK=256
        (256,  6144,    6144,    2048),  # main  small_dense sample gate
        (256,  262144,  262144,  2048),  # main  v32 KBIG-domain, BLK=512 KPT=4
    ]
    for shp in smoke:
        print(shp, '->', route(*shp))
