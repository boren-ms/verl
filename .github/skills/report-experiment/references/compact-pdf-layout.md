# Compact PDF Layout

Use this layout for the default `report.pdf`. The PDF is a compact reading
surface; `report.md` and `evidence.json` retain full run identity, provenance,
limitations, and detailed findings.

## Allowed content

The PDF contains only these sections, in this order:

1. **Configuration differences**
   - Difference-only effective-settings table.
   - Omit identical controls; keep them in `evidence.json` and
     `data/shared_controls.csv`.
   - Add one short multivariable-comparison caveat on the same page.

2. **Validation**
   - Overlaid observed `p_err` trajectories using actual optimizer/validation
     steps and stable run colors.
   - Exact shared-step table when aligned points exist.
   - A compact progression or latest/best table only when it adds information
     not already readable from the figure.
   - Put unequal-budget, provisional-run, or provenance caveats beside the
     affected figure/table.

3. **Reward-core**
   - Group compatible primary components into one readable figure page.
   - Group compatible secondary components into another figure page only when
     needed for legibility.
   - Include exactly one canonical exact-shared-window mean table when a common
     window exists.
   - State that these are training-batch diagnostics and give the verified
     direction; do not add a separate interpretation page.

## Page budget

For a one-dataset comparison with up to three runs, target four A4 landscape
pages:

1. configuration differences;
2. validation figure plus aligned/progression tables;
3. primary `reward-core` figure;
4. secondary `reward-core` figure plus shared-window table.

The page count is a target, not a reason to shrink content below readable size.
Multiple datasets, many runs, or a long difference table may continue within
the same three sections. Split dense tables or figures across additional pages
instead of adding cover, executive-summary, run-identity, standalone
interpretation, evidence-limit, appendix, or filler pages.

## Verification

- Use document pages, normally A4 landscape; never use slide dimensions.
- Render every page to an image and inspect a contact sheet.
- Inspect dense tables at full resolution for clipping and legibility.
- Extract PDF text and confirm the three section headings are present.
- Confirm no forbidden standalone section/page was introduced.
- Record page count, rendering method, extracted-text count, page-render paths,
  contact-sheet path, and `pptx_generated: false` in `evidence.json`.
