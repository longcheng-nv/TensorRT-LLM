# COST.md — op42
| phase | GPU-h | notes |
|---|---|---|
| P0 setup + anchors | 0.3 | b200-073 GPU0-3 |
| iter1-2 screens (nsys 12+7 cells) | ~2.5 | 4-GPU sharded |
| iter4-6 tp verdicts (9 cells x3) | ~3.5 | 4-GPU sharded |
| gates (smoke) | ~0.5 | single GPU |
