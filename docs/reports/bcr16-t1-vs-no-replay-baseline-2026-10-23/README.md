# BCR16 T1 versus the no-replay T1 baseline

Comparison collected on October 23, 2026. Both runs are stopped; the BCR16 T1 recovery monitor was deleted.

- [Compact PDF report](report-compact.pdf): five pages covering the main findings, validation trajectories, reward means, and matched examples.
- [PDF report](report.pdf): 16 slides covering config/code differences, validation trajectories, reward means, and matched examples.
- [Markdown report](report.md): results, exact metric keys, counts, comparison limits, and embedded figures.

At shared validation step 600, BCR16 T1 improves LibriSpeech clean and other, slightly improves OpenML bad, and slightly regresses Earnings. The baseline has replay disabled at temperature 1.0. Replay changes fresh-data exposure, and the reward divisor also changes from 4 to 3; this is a descriptive comparison rather than an isolated replay ablation.

The `figures/` folder contains only images used by the Markdown report.
