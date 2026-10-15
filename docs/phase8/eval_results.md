# Phase 8 evaluation (synthetic scenes, real tracker code)

Each scene locked 6 times; MegaPose simulated as a random symmetric twin of the truth (+-2 deg, +-3 mm).
`correct` = within 5 deg and 1 cm of ground truth; `flipped` = more than 90 deg off.
`wrong pose inspectable` = locks where inspection would have judged components on a wrong pose.

| Scene | before: correct / flipped | after: correct / flipped / refused | after: confirmed / ambiguous | wrong pose inspectable before -> after | circle error px (after) | lock ms p95 before -> after |
|---|---|---|---|---|---|---|
| turned 0 deg on the desk | 2 / 4 | 6 / 0 / 0 | 6 / 0 | 4 -> 0 | 1.79 | 714 -> 878 |
| turned 90 deg on the desk | 1 / 5 | 6 / 0 / 0 | 6 / 0 | 5 -> 0 | 1.09 | 608 -> 808 |
| turned 180 deg on the desk | 2 / 4 | 6 / 0 / 0 | 6 / 0 | 4 -> 0 | 1.33 | 594 -> 865 |
| turned 270 deg on the desk | 3 / 3 | 6 / 0 / 0 | 6 / 0 | 3 -> 0 | 1.06 | 614 -> 826 |
| camera 35 deg above the desk | 1 / 2 | 5 / 1 / 0 | 1 / 5 | 5 -> 0 | 2.80 | 506 -> 852 |
| camera 55 deg above the desk | 1 / 5 | 6 / 0 / 0 | 6 / 0 | 5 -> 0 | 1.15 | 388 -> 762 |
| camera 75 deg above the desk | 1 / 5 | 6 / 0 / 0 | 6 / 0 | 5 -> 0 | 1.52 | 365 -> 647 |
| distance 0.25 m | 2 / 4 | 6 / 0 / 0 | 6 / 0 | 4 -> 0 | 1.69 | 595 -> 845 |
| distance 0.45 m | 3 / 3 | 6 / 0 / 0 | 6 / 0 | 3 -> 0 | 0.93 | 588 -> 842 |
| moved to the left of the frame | 4 / 2 | 6 / 0 / 0 | 6 / 0 | 2 -> 0 | 4.00 | 515 -> 827 |
| moved to the right of the frame | 2 / 4 | 6 / 0 / 0 | 6 / 0 | 4 -> 0 | 0.82 | 641 -> 877 |
| tilted 12 deg | 2 / 4 | 6 / 0 / 0 | 6 / 0 | 4 -> 0 | 1.30 | 569 -> 843 |
| hand over the circle | 4 / 2 | 4 / 2 / 0 | 0 / 6 | 2 -> 0 | — | 676 -> 721 |
| hand over a corner (circle visible) | 3 / 3 | 6 / 0 / 0 | 6 / 0 | 3 -> 0 | 1.58 | 725 -> 926 |
| low contrast, noisy | 3 / 3 | 6 / 0 / 0 | 6 / 0 | 3 -> 0 | 1.38 | 653 -> 853 |
| plain box (no circle) | 1 / 5 | 1 / 5 / 0 | 0 / 6 | 5 -> 0 | — | 641 -> 728 |

## Tracking: 12 background corrections, each a random twin (box held still)

| | shown-pose flips | worst error deg | guard counters | update ms (mean) |
|---|---|---|---|---|
| before | 8 | 178.7 | {'twin_flips_suppressed': 0, 'corrections_rejected': 0, 'orientation_switches': 0} | 20 |
| after | 0 | 1.1 | {'twin_flips_suppressed': 6, 'corrections_rejected': 0, 'orientation_switches': 0} | 20 |

## Operator turns the box round (legitimate 180 deg change, circle visible)

- before: 126.0 deg off after one correction
- after: 0.0 deg off after one correction
