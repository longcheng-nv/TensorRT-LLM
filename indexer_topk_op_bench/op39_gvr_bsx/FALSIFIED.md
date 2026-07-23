# op39 falsification ledger (inherits op38/op37/op20 GVR history — read project_gvr_topk_falsification_history)
- (r3_v11 (TB,CS,MAXV,AR,HS) ladder cannot reach 1.0x on 133/137 BS16-1024 loss cells; domain: §7b fp32 BS 16-1024, B200; evidence: nsys+event ladder) [op38] class: structural-wall. Revival: new arm shape only.
- (sample rank r < ~64 with 256-cluster sampling undershoots the tail -> resort storms; domain: npad >= 262144 fp32 real captures; evidence: nsys a2 (r=40) AND a5 (r=48) — falsified TWICE, second time by re-proposing inside the falsified domain) [op39 iter5/iter9] class: statistical-floor. Revival: only with >=1024 independent clusters or an exact count-feedback pass.
- CDP2 tail-launch K2 (device-side conditional rescue launch): -rdc=true costs
  15-20% globally on this reg-starved kernel (device-runtime reserve) >> the
  1.6-5.7% K2-free bound. Do not re-propose device launch in any arm_v2-family
  kernel; the CDP code stays behind -DARM39_CDP for reference.
- K2 grid-diet (1 CTA/row grid-stride rescue): rejected without measurement —
  rescue-storm cells (low-hit big-N) would regress chunks-fold.
