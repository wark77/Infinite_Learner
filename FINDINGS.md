# Findings log (chronological highlights, with corrections)

All numbers: numpy MLPs, 8 seeds unless noted, `evals` = gradient-evaluation equivalents (fwd+bwd on a batch = 1, forward-only probe = 1/3).
Sequences: 5 tasks, replay from a growing buffer, 3000 evals/task, 64 new + 16 replayed samples per step.

## 1. Known landscapes (ground truth, random rotation, D = 2..64)
* **Corridor** (narrow winding valley): fan variants solve 100% for all D in ~250-800 evals (4-8 steps for the 3-view fan); Adam 1000-2300 evals.
* **Ridge** (thin high wall between basins): Adam 0%; ladder fan / curvature rays 100% (curvature "stop" mode alone 50-75%, 100% with the stuck trigger).
* **Side basin, no gradient hint:** every method fails for D >= 8 (works for D <= 4). Nothing can find a hidden direction without information.
* **Two-task barrier** (old well, joint well, broad hill, Adam gets stuck in a local minimum): Adam 0/0/0 at D=2/8/32.
  Stuck-triggered third axis with a dense retry ladder: **100%** at D=2, 8, 32 (200-276 evals, 3-6 steps). Ablation (no audit): dense ladder
  without third axis 62-100%; third axis with a coarse ladder 0%; always-on third axis 0-100% depending on D.

## 2. Continual-learning sequences (final mean accuracy)
| method | split-digits | permuted-64 | split-Fashion | permuted-16 |
|---|---|---|---|---|
| joint training (ceiling) | 0.979 | 0.942 | 0.827 | 0.816 |
| Adam, random replay, lr 0.01 (early, UNTUNED) | 0.976 | 0.880 | 0.797 | 0.791 |
| Adam, random replay, lr tuned per batch size | - | **0.916-0.922** | - | - |
| 2-axis ladder fan | 0.966 | 0.865 | 0.750 | 0.690 |
| curvature rays (5-scale window) | 0.972 | 0.885 | 0.765 | 0.65-0.69 |
Fine-tuning without replay forgets (0.16-0.57). Straight-line old-task loss barrier at every task transition: 0.000 in all real sequences.

## 3. Batch / data-reuse controls (permuted-64, lr tuned per variant, equal compute per task)
| method | evals/batch | batches (sequential steps) | training items/task | final acc |
|---|---|---|---|---|
| Adam batch x1 / x4 / x16 / x32 | 1 / 4 / 16 / 32 | 3000 / 750 / 188 / 94 | 240k each | 0.916 / 0.922 / 0.917 / 0.900 |
| Adam reusing each batch x4 / x16 / x32 | 4 / 16 / 32 | 750 / 188 / 94 (x4/x16/x32 inner steps = 3000 sequential updates) | 60k / 15k / 7.5k | 0.918 / 0.887 / 0.857 |
| ladder fan (fan2) | 30.6 | 98 | 7.8k | 0.865 |
| curvature rays | 15.4 | 195 | 15.6k | 0.885 |
| curvature rays on a x4 batch | 58.8 | 51 | 16.3k | 0.890 |
Reading: at equal items and equal evals the fan ties Adam-with-reuse (0.885 vs 0.887) but needs ~15x fewer *sequential* updates (195 vs 3000);
Adam with a big batch has the same depth but draws ~15x more items. The fan did not beat tuned Adam on accuracy per compute.

## 4. Other findings
* **Counter ("spaced repetition") replay:** the literal scheme starves older tasks in a growing buffer (0.52-0.59); resetting counters at each task start fixes it, but it is no better than random replay.
* **Smoothed-gradient axis:** equals the gradient (cos 1.000) up to scale ~4 on digits; diverges at scale 8-16 (real structure, two draws agree). Always-on third axes
  (smoothed at scan scale or 2x, step-scaled, elementwise, curvature) cost 2-4x per step and lose 0.01-0.14 accuracy on real sequences. A **stuck-triggered** axis
  never fires on permuted/Fashion (results identical to base) and ~1% of steps on split (+5-8% cost).
* **Negative results:** rays that stop where two gradient scales diverge (0.27-0.65); uphill escape moves; consensus/CVaR selection over per-problem losses;
  fan / tunnelling on parity-8 plateaus (0/12 escapes vs Adam 2/12); low-data full batch (L-BFGS 5 steps / 7 evals vs fan 6 steps / 208+ evals).
* Wall-clock model: with 3 sync rounds per step the fan can win at most ~1.7-1.8x, and only when synchronisation latency >= ~2 gradient evaluations.

## 5. Corrections made along the way
1. Early Adam baselines used a fixed lr 0.01; tuned Adam is stronger (section 3). Claims that the fan "beats Adam" on permuted-64 held only against the untuned baseline.
2. Early barrier-crossing results mixed the third axis with ladder alignment (the hand-tuned ladder contained a point inside the 4.0-6.5 improving region; the default ladder did not).
3. The first "no-leak" side landscape leaked the target direction through a Gaussian tail; replaced by a compact-support well.
4. Smoothed gradient is redundant with the *gradient* on digits, not with momentum.
5. The single-file runner initially dropped an RNG draw (barrier measurement) and therefore did not reproduce earlier runs exactly; restored, now bit-identical.

## 6. Open questions
Dense-ladder range from the scan scale (currently top 12, ratio 1.25); stuck trigger on real plateaus (parity-8); larger nets / GPU / real datasets;
stronger baselines (SAM, L-BFGS, LR schedules, large-batch tricks); probe batch smaller than gradient batch (noisy selection); adaptive allocation of evals between rays, scales and refinement.
