# Machine Learning Vision: a view-based optimizer (exploratory research code, 0BSD)

Instead of one gradient step per batch, the **fan** draws a batch and spends a budget of *forward-only probes* on it:
it builds a small "screen" of candidate directions from informative views (gradient, momentum, optionally a smoothed
gradient), probes each ray at a ladder of distances, and takes the best point -- even through higher points, so it can
jump over hills. Variants: a 5-scale window that follows the feature scale ("curvature rays"), an audit that vetoes steps
that hurt old tasks, and a **stuck-triggered third axis** with a dense retry ladder. Probes are independent, so they batch and parallelise.

Everything lives in one file: `fan_experiments.py` (numpy, scikit-learn, matplotlib).

    python fan_experiments.py                       # ~1 min demo: split-digits + landscapes -> fan_results/*.png
    python fan_experiments.py --exp perm fashion    # other 5-task continual-learning sequences
    python fan_experiments.py --full --exp split perm perm16 fashion landscapes --workers 0   # replication settings
    python fan_experiments.py --exp batch --batch-seqs perm --seeds 8 --budget 3000           # Adam batch / data-reuse controls
    python fan_experiments.py --list-methods

Graphs: loss (all seen / current / earlier tasks) and mean accuracy vs evals; landscape objective vs evals.
"evals" = gradient-evaluation equivalents (forward+backward on a batch = 1, forward-only = 1/3). Runs are deterministic
and independent of `--workers`.

## Honest summary of what the experiments show (see FINDINGS.md for tables, corrections and negative results)
* **Strong:** narrow winding valleys (up to ~40x fewer steps than Adam at similar compute) and a constructed two-task *barrier*
  landscape that Adam never crosses (the stuck-triggered third axis crosses 100% at D=2/8/32).
* **Not (yet) shown:** better accuracy per compute than a *tuned* Adam on small real continual-learning sequences. Against
  Adam with a tuned learning rate (e.g. 0.916-0.922 on permuted-digits) the best fan variants (0.885-0.890) are behind.
* **Distinctive property:** few *sequential* steps AND few training items at once. At equal data items and equal evals the fan
  roughly ties Adam-with-batch-reuse (0.885 vs 0.887) but needs ~15x fewer sequential updates (195 vs 3000 per task);
  Adam with a large batch has the same depth but draws ~15x more items. This trade matters where steps/synchronisation or data are expensive.
* Early comparisons in this project used an untuned Adam (lr 0.01); later controls tune it -- prefer the tuned numbers.

## Caveats
Tiny problems (digits, 14x14 Fashion-MNIST, numpy MLPs), CPU only, 8 seeds, hand-chosen fan hyperparameters. The barrier landscape is
constructed so the new-task gradient points across the hill. No GPU / large-model / large-dataset tests yet.

## License
BSD Zero Clause (0BSD) -- see LICENSE. `archive/` holds the exploratory multi-file scripts and logs the single file was consolidated from.
