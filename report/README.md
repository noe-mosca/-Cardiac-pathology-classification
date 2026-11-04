# Report

LaTeX sources of the report. Upload this directory as an Overleaf project, or compile `main.tex` with pdfLaTeX and BibTeX.
The report uses the official ICLR 2025 conference style (`iclr2025_conference.sty` and `.bst`, `math_commands.tex`, `fancyhdr.sty`, `natbib.sty`, all included). `\iclrfinalcopy` shows the author, and the page header is replaced by "IMA205 challenge report, Télécom Paris".

- `main.tex`: title, author, packages and section assembly.
- `sec_merge/`: abstract, one file per section (`1_intro` to `7_conclusion`), the reproducibility and AI use statements (`statements.tex`, before the references) and the appendix (`8_appendix.tex`, Appendices A to G with their own table of contents).
- `figs/`: figures of the main text; `figs/supp/`: appendix figures.
- `tabs/`: tables of the main text; `tabs/supp/`: full tables of the appendix.
- `references.bib`: bibliography database (28 entries, all cited).

## Regenerating figures and tables

Figures and tables are produced from the experiment outputs (`results/` of the code). From the folder that contains both the code and this directory:

```bash
python3 report/regenerate_figures.py        # figures and tables of the main text (~20 s)
python3 report/regenerate_supplementary.py  # appendix figures and tables (~20 s)
```

Both scripts need the `cardiac` package, the cached features (`cache/features/`) and the segmentations (`data/acdc/`, for the example slices).

`tabs/features.tex` was written by hand from `cardiac/features.py`. All other tables are generated: `tabs/{protocol,benchmark,lv_reconstruction,lv_protocol}.tex` by `regenerate_figures.py`, `tabs/supp/*.tex` by `regenerate_supplementary.py`. The numbers quoted in the text of `sec_merge/` were written by hand from `results/`: if the experiments are rerun, the text must be checked against the regenerated tables.
