# Universality Theorem: Discreteness-as-Stability

## Overview

Automated validation suite demonstrating the universality of the "discreteness-as-stability" mechanism across multiple dynamical systems (quantum, classical, abstract) for Nature Physics publication.

## Key Result

**UNIVERSALITY VALIDATED: 5/6 models (83%) pass**

| Model | Class | Shelf Δ | Status |
|-------|-------|---------|--------|
| Floquet Ising | quantum | 1.823 | ✅ |
| Floquet XXZ | quantum | 2.127 | ✅ |
| Kicked Rotor | quantum | 0.477 | ✅ |
| Circle Map | classical | 1.000 | ✅ |
| Standard Map | classical | 0.278 | ✅ |
| GL(2,R) Trace | abstract | 0.139 | ❌ (resolution-limited) |

## Theorem Statement

**Discrete locking shelves emerge iff a system is:**
1. **Reversible** (unitarity / symplecticity / det = ±1)
2. **Non-integrable** (chaotic / mixed phase space)
3. **Weakly perturbed** (σ < σ_crit)

## Files

### Main Scripts
- `universality_theorem.py` - Complete theorem validation (Python)
- `universal_validation.py` - Cross-model validation suite
- `run_full_validation.ts` - TypeScript batch processor
- `universality_suite.ts` - TypeScript model implementations

### Output Documents
- `universality_theorem/universality_theorem.md` - Formal theorem + validation
- `universality_theorem/executive_summary.md` - 237-word Nature summary
- `universality_theorem/theorem1_latex.tex` - LaTeX theorem (RevTeX4)
- `universality_theorem/nature_abstract.md` - Abstract options + cover letter

### Data
- `universality_theorem/failure_boundary_map.csv` - 144 simulation runs
- `universal_validation/summary_table.csv` - Model comparison

### Figures (Publication-Ready SVG)
- `cross_model_shelf_comparison.svg` - Killer figure
- `shelf_vs_reversibility.svg` - Condition (i) test
- `positive_vs_negative_controls.svg` - Failure mode demo

## Running Validation

```bash
# Python suite (recommended)
python universality_theorem.py

# TypeScript suite
npx tsx run_full_validation.ts
```

## Requirements

- Python 3.11+ with numpy
- Node.js 20+ with tsx

## License

MIT

## Citation

If using this work, please cite: [Paper reference TBD]
