# Operational-b sensitivity: the ML hybrid is unstable, the physics is not

Verdicts (test split) from the pre-specified block-bootstrap rule at three operational b.
b=1.15 is the honest optimum (pre-2024 slope 1.013); 1.10/1.20 bracket it.

## Hybrid vs physics (verdict FLIPS with b)

### y30/test

| pair | b=1.10 | b=1.15 | b=1.20 |
|---|---|---|---|
| hybrid vs cascade | inseparable | inseparable | B_beats_A |
| hybrid vs sv_etas | inseparable | inseparable | B_beats_A |
| hybrid vs modern_etas | inseparable | inseparable | B_beats_A |
| hybrid vs firstgen_etas | B_beats_A | inseparable | B_beats_A |

### y35/test

| pair | b=1.10 | b=1.15 | b=1.20 |
|---|---|---|---|
| hybrid vs cascade | B_beats_A | inseparable | inseparable |
| hybrid vs sv_etas | inseparable | B_beats_A | inseparable |
| hybrid vs modern_etas | B_beats_A | B_beats_A | inseparable |
| hybrid vs firstgen_etas | B_beats_A | B_beats_A | inseparable |

## Physics vs physics (STABLE across b)

### y30/test

| pair | b=1.10 | b=1.15 | b=1.20 |
|---|---|---|---|
| cascade vs sv_etas | inseparable | inseparable | inseparable |
| cascade vs firstgen_etas | inseparable | inseparable | inseparable |
| modern_etas vs firstgen_etas | B_beats_A | B_beats_A | B_beats_A |

### y35/test

| pair | b=1.10 | b=1.15 | b=1.20 |
|---|---|---|---|
| cascade vs sv_etas | inseparable | inseparable | inseparable |
| cascade vs firstgen_etas | inseparable | inseparable | inseparable |
| modern_etas vs firstgen_etas | B_beats_A | B_beats_A | B_beats_A |
