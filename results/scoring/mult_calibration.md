# Item G: empirical family-wise error of the conjunctive verdict rule

Negative control: cascade@seed_i vs cascade@seed_j (true diff = 0), 6 pairs, B=2000.

- **False separations: 0/6  (empirical per-comparison false-positive rate 0.000).**
- Naive-independence expectation at alpha=0.05 over 216 comparisons: ~11 false positives; the conjunctive rule (BOTH IG and PR-AUC CIs exclude 0) is far stricter, as this rate shows.
- Claimed y30/test separations in the canonical claims file: 15/36; they replicate across y30 and y35 (independent chance at the measured rate does not reproduce a both-target separation).

| negative-control pair | verdict (want: inseparable) |
|---|---|
| cascade_s1000_vs_cascade_s7000 | inseparable |
| cascade_s1000_vs_cascade_s13000 | inseparable |
| cascade_s1000_vs_cascade_s19000 | inseparable |
| cascade_s7000_vs_cascade_s13000 | inseparable |
| cascade_s7000_vs_cascade_s19000 | inseparable |
| cascade_s13000_vs_cascade_s19000 | inseparable |