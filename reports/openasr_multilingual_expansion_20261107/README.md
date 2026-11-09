# OpenASR multilingual expansion Excel report

`openasr_multilingual_expansion.xlsx` opens on the `multilingual_expansion`
sheet: six supplementary datasets, pooled CORAAL, all eight CORAAL locations,
and error-reduction formulas against the freshly evaluated 2609r2 baseline.
Mandarin rows use CER; the remaining rows use WER. Fractions are stored at full
precision and displayed as percentages. No mixed CER/WER average is calculated.

The original canonical table and existing metadata cells are preserved.
The six new datasets and location diagnostics do not enter headline averages.
Fresh pooled CORAAL is shown separately from the earlier canonical CORAAL result.
All 42 exact detail paths and three evaluation identities are appended to the
metadata sheets under labels ending in `[multilingual expansion]`.

The attached `comparison.json` contains counts, exact checkpoints, pinned data
revisions, provenance hashes, and the 2611r2 runtime-recovery record.
`comparison.md` is the readable comparison. Evidence was collected on 2026-11-07;
the workbook was generated on 2026-11-09.

Template: `openasr_2611r2_vs_2609r2_and_2611s2.xlsx` from the preceding full-suite
comparison; SHA-256 `6cd8b77562f08e4f2b95a1c0c94ba72b0c1f44e3db21340f3cdf183fdc5a9d45`.

Regenerate with the preceding canonical workbook as input:

```bash
/home/boren/.virtualenvs/openai/bin/python \
  scripts/build_openasr_multilingual_expansion_xlsx.py \
  --comparison reports/openasr_multilingual_expansion_20261107/comparison.json \
  --template /path/to/preceding_openasr_report.xlsx \
  --out reports/openasr_multilingual_expansion_20261107/openasr_multilingual_expansion.xlsx
```

Verification reopened the output, matched every fresh metric to the saved
counts, checked error-reduction formulas and positive/negative cases, verified
all model/result metadata, and compared every original cell value and style.
