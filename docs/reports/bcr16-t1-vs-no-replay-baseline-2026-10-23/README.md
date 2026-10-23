# BCR16 T1 versus the no-replay T1 baseline

Comparison collected on October 23, 2026. Both runs are stopped; the BCR16 T1 recovery monitor was deleted.

- [PDF report](report.pdf): 16 slides covering config/code differences, validation trajectories, reward means, and matched examples.
- [Markdown report](report.md): results, exact metric keys, counts, and comparison limits.
- [Evidence](evidence.json): machine-readable histories, provenance, matched examples, and figure manifest.
- [Editable slide narrative](slides.md) and [report builder](build_report.py).
- [Source evidence](sources/) and [reusable figures](figures/).
- [PDF validation](pdf_validation.json).

At shared validation step 600, BCR16 T1 improves LibriSpeech clean and other, slightly improves OpenML bad, and slightly regresses Earnings. The baseline has replay disabled at temperature 1.0. Replay changes fresh-data exposure, and the reward divisor also changes from 4 to 3; this is a descriptive comparison rather than an isolated replay ablation.

To rebuild the report from the archived evidence, run `python build_report.py` from this directory. The collection scripts document the original evidence retrieval; rebuilding does not require rerunning them. Historical paths inside evidence identify the original collection locations.
