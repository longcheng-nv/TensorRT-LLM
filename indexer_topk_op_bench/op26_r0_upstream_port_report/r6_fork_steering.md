# COMPLIANCE CLARIFICATION — the "broadcast trick" is REJECTED; here is the exact line

The external referee has DISQUALIFIED kernel f12345ed (indexer_topk_bcast_v1)
and will disqualify every kernel of its family. Read carefully.

## The rule, stated precisely

The benchmark materializes identical batch rows. That is a property of the
TEST HARNESS, not of the problem. Your kernel must be a per-row GVR that would
be correct AND fast on ARBITRARY per-row data. Therefore:

1. ALLOWED cross-row amortization: P1 ONLY. You may build the prior /
   threshold ladder from `pre_idx`+gathered hint values once and broadcast it
   to all rows (with per-row escape if a row's solve does not converge under
   the shared seed). The hint tensor is also identical per row — exploiting
   THAT is legitimate GVR-prior engineering.
2. REQUIRED per row: the threshold verification/secant refinement against the
   ROW'S OWN DATA, the candidate collect from the row's own data, and the
   exact refine. Every row's logits must actually be read and selected from.
3. NON-COMPLIANT (auto-DQ at harvest, regardless of internal score):
   - computing top-k for one row and broadcasting/copying the RESULT to other
     rows — with or without an equality-verification pass;
   - any equality/memcmp test between rows used to skip per-row selection;
   - prior-free radix-select/sort cores (restated; f12345ed violated this too).
4. The published insight "Broadcast trick: all BS rows identical, compute
   top-k once" and "Exact row verification enables safe batch reuse" describe
   NON-COMPLIANT approaches. Do not build on them.

## Why (so you optimize the right thing)

In production, decode batches contain DIFFERENT rows; result-broadcast paths
never fire there. The acceptance authority measures your kernel as if rows
were arbitrary — a result-broadcast kernel scores 0 in the final verdict.

The compliant lineage already in this campaign ("Tiered self-written GVR",
+9.2%, exact on 45/45) is the right base. The winning directions remain:
shared-P1 prior amortization (legitimate), rows-per-SM packing / persistent
CTAs for the bs16-128 x 16k-65k occupancy valley, pass-count reduction in the
bandwidth zone, and holding the BS=1 champion structure at b==1.
