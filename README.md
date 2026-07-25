# KF R6 compliant champion `indexer_topk_gvr_r5_v14` (fb1e6848)

Strict per-row GVR (preIdx prior -> secant+log solve -> exact refine), CUDA C++
sm_100a, batched BS 1-1024. Final compliant winner of campaign gvr-topk-bs40-v2
(pq3hwx7eh94k1arcf0hwmn7wem, round 5; campaign cut by fable-5 quota outage).
All 9 dispatched paths consume pre_idx (launcher-level verified).
Verdict (nsys cold-L2 paired vs PR#16457 head @04a0900ff7, real §7b data):
1615/1615 exact; BS=1 865-cell gm 0.8331 (bs1 probe band 1.036);
BS2-1024 750-case gm 0.6706. Campaign bars (per-K gm>=1.40, min>=0.95) NOT met
— see R6_CLOSEOUT.md. Complementary asset: 15e80901 (better bs128-1024).
Entry: main.cpp::run(logits[b,npad] f32, pre_idx[b,k] i32, n_valid, indices out).
