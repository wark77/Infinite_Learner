#!/usr/bin/env python3
"""
fan_experiments.py -- ONE self-contained file that replicates the fan experiments and draws loss-vs-evals graphs.

WHAT IT RUNS
  Continual-learning sequences (5 tasks, replay from a growing buffer, a fixed compute budget per task):
      split    split-digits, 2 classes per task            (class-incremental)
      perm     digits with 5 different pixel permutations   (domain-incremental)
      perm16   same, hidden width 16 (capacity-limited)
      fashion  split Fashion-MNIST, 2 classes per task      (downloaded from GitHub on first use; skipped if unreachable)
  Known landscapes with ground truth: corridor, ridge, side_open, and a two-task barrier landscape.
  Methods: Adam (random / counter replay), fine-tuning, the 2-axis ladder fan ("fan2"), curvature-following rays
  ("curv_fixed" / "curv_stop"), frozen-window controls, third-axis variants ("+ss1", "+as2", ...), ray-stopping ("march_*").

WHAT IT DRAWS
  fan_results/loss_vs_evals_<exp>.png     four panels vs evals (gradient-evaluation equivalents; fwd+bwd = 1, forward-only = 1/3):
        loss on all tasks seen so far | loss on the current task | loss on earlier tasks | mean test accuracy
        (task boundaries are dashed vertical lines; bands = +-1 std over seeds)
  fan_results/landscape_loss_vs_evals.png objective vs evals for each landscape (median over seeds, band = inter-quartile range)
  fan_results/results_<exp>.json          raw traces and final metrics
  plus summary tables on the console.

USAGE
  python fan_experiments.py                          # quick demo (about a minute): split-digits + landscapes
  python fan_experiments.py --exp perm fashion       # other sequences
  python fan_experiments.py --full --exp split perm perm16 fashion landscapes --workers 0
                                                     # replication settings (8 seeds, 3000 evals/task); use --workers N for N cores
  python fan_experiments.py --methods adam_rand,curv_fixed,curv_fixed+as2 --seeds 4
  python fan_experiments.py --list-methods

REQUIREMENTS  numpy, scikit-learn, matplotlib (Fashion-MNIST optional, needs internet on first use).
All runs are deterministic: results do not depend on --workers.
"""
import argparse, json, os, sys, time, multiprocessing as mp
import numpy as np


# ======================================================================================================================
# 1. ALGORITHMS  (verbatim from fan.py)
#    Adam | Fan (ladder fan; views grad / mom / smooth / asmooth / ssmooth, Gram-Schmidt, pixel grid, audit veto)
#    CurvFan (curvature-following rays over a window of scales) | MarchFan (rays that stop where two gradient scales diverge)
#    CounterReplay (spaced-repetition replay over all stored problems)
# ======================================================================================================================



class Adam:
    def __init__(self, lr, b1=0.9, b2=0.999):
        self.lr, self.b1, self.b2, self.m, self.v, self.t = lr, b1, b2, None, None, 0

    def step(self, w, P, rng):
        _, g = P.lg(w)
        if self.m is None: self.m, self.v = np.zeros_like(w), np.zeros_like(w)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        return w - self.lr * (self.m / (1 - self.b1 ** self.t)) / (np.sqrt(self.v / (1 - self.b2 ** self.t)) + 1e-8), 1.0


class Fan:
    def __init__(self, views=("grad", "smooth", "mom"), rule="mean", res=3, K=11, tmin=0.003, ratio=2.2,
                 scale=16.0, M=4, mom_decay=0.8, min_resid=0.05, tol=0.05,
                 audit_tol=0.5, audit_abs=0.05, early_steps=0, early_tol=3.0, early_abs=0.5, curv_eps=0.01, use_batch=True,
                 scan_every=10, scan_scales=(0.5, 1, 2, 4, 8, 16, 32), scan_M=2, tau_div=0.97, tau_noise=0.6,
                 asm_mult=1.0, smooth_mult=1.0, skip_redundant=0,
                 stuck_views=None, stuck_mult=2.0, stuck_patience=2, stuck_gain=1e-4, stuck_dense=True, dense_ratio=1.25, dense_top=12.0, stuck_backoff=True):
        """asm_mult: smoothing scale of the 'asmooth' view = asm_mult x the scale found by the scan. smooth_mult: smoothing scale of the
        'ssmooth' view = smooth_mult x the auto-sensed STEP scale (re-centring window centre for CurvFan, else a running average of accepted
        step lengths). skip_redundant: if the smoothed axis was dropped as redundant for this many steps in a row, probe it only every
        that-many-th step (0 = always probe).
        stuck_views: if set (e.g. ('asmooth','grad','mom')), a STUCK-TRIGGERED third axis: normal steps use `views`; when `stuck_patience`
        consecutive steps found no improving pixel or a relative gain < stuck_gain, the same step is retried with these views, the asmooth
        scale multiplied by stuck_mult, and (CurvFan) a fresh wide scale scan. The retry's cost is counted.
        views/rule: see module doc (rule 'mean' = lowest mixed loss; 'constrained' = lowest new-task loss subject to
        old-task loss <= (1+tol)*old). res: pixels per axis. K, tmin, ratio: distance ladder tmin*ratio^k, k<K.
        scale, M: norm-scale and number of antithetic pairs of the smoothed-gradient view.
        audit_*: tolerance of the audit veto; for the first `early_steps` steps of each task it is the permissive
        (early_tol, early_abs) instead."""
        self.views, self.rule, self.res, self.K, self.tol = tuple(views), rule, res, K, tol
        self.ts = tmin * ratio ** np.arange(K)
        self.scale, self.M, self.mom_decay, self.min_resid = scale, M, mom_decay, min_resid
        self.audit_tol, self.audit_abs = audit_tol, audit_abs
        self.early_steps, self.early_tol, self.early_abs, self.curv_eps = early_steps, early_tol, early_abs, curv_eps
        self.use_batch = use_batch       # evaluate all probes of a step in ONE batched forward pass when the problem supports it (P.parts_batch)
        self.mom, self._grids, self.task_steps = None, {}, 0
        self.scan_every, self.scan_scales, self.scan_M, self.tau_div, self.tau_noise = scan_every, tuple(scan_scales), scan_M, tau_div, tau_noise
        self.scale_star, self.last_scan, self.scan_log = None, 0, []
        self.asm_mult, self.smooth_mult, self.skip_redundant = asm_mult, smooth_mult, skip_redundant
        self._step_ema, self._sm_streak = None, 0
        self.stuck_views, self.stuck_mult, self.stuck_patience, self.stuck_gain = stuck_views, stuck_mult, stuck_patience, stuck_gain
        self.stuck_dense, self.dense_ratio, self.dense_top, self.stuck_backoff, self._patience = stuck_dense, dense_ratio, dense_top, stuck_backoff, stuck_patience
        self._stuck_count, self._in_retry, self._force_wide, self._scanned, self._last_gain, self._last_moved = 0, False, False, False, 0.0, False
        self.stats = dict(steps=0, fail=0, accepted=0, axes=0, shrunk=0, vetoed=0, sm_probes=0, sm_kept=0, stuck_retries=0, stuck_success=0)

    def _step_scale(self):
        """Auto-sensed step scale: running average of accepted step lengths (CurvFan overrides with its window centre)."""
        return self._step_ema

    def new_task(self):
        """Call when a new task starts (resets the per-task step counter used by the permissive-early audit)."""
        self.task_steps = 0

    def _grid(self, n):
        if n not in self._grids:
            axis = np.linspace(-1, 1, self.res)
            A = np.stack(np.meshgrid(*([axis] * n), indexing="ij"), -1).reshape(-1, n)
            self._grids[n] = A[np.any(A != 0, axis=1)]                  # drop the zero pixel
        return self._grids[n]

    def _dense_probe(self, w, P, B, A, m0, wo):
        """Stuck retry: probe the pure axis rays (+-each orthonormal view) on a DENSE geometric ladder (ratio dense_ratio, up to dense_top).
        A coarse ladder can step straight over a narrow improving region; the informative rays are cheap to refine (2n rays only)."""
        n = B.shape[1]; K = int(np.ceil(np.log(self.dense_top / self.ts[0]) / np.log(self.dense_ratio))) + 1
        ts = self.ts[0] * self.dense_ratio ** np.arange(K); rays = np.concatenate([B.T, -B.T]); cost = len(rays) * K / 3.0
        Wc = (w[None, None, :] + ts[None, :, None] * rays[:, None, :]).reshape(len(rays) * K, -1)
        if hasattr(P, "parts_batch"): ln, lo = P.parts_batch(Wc)
        else:
            pr = [P.parts(x) for x in Wc]; ln = np.array([p[0] for p in pr]); lo = np.array([p[1] for p in pr])
        m = (1 - wo) * ln + wo * lo; ok = np.isfinite(ln) & np.isfinite(lo) & (m < m0 - 1e-12)
        if not ok.any(): return None, np.inf, None, cost
        idx = int(np.argmin(np.where(ok, m, np.inf))); ri, ti = divmod(idx, K)
        target = np.zeros(n); target[ri % n] = 1.0 if ri < n else -1.0
        row = int(np.where((A == target).all(1))[0][0]) if (A == target).all(1).any() else 0
        self.stats["dense_wins"] = self.stats.get("dense_wins", 0) + 1
        return ts[ti] * rays[ri], float(m[idx]), row, cost

    def _scan(self, w, P, g, rng):
        """Sample the scale of the structure: for each scale s compare the plain gradient with the smoothed gradient
        (two independent draws sa, sb). Returns the SMALLEST scale where the smoothed gradient differs from the gradient
        (cos < tau_div) while the two draws still agree (cos >= tau_noise, i.e. structure, not sampling noise), else None."""
        D = len(w); cost = 0.0; chosen = None
        cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
        for s in self.scan_scales:
            sa, sb = np.zeros(D), np.zeros(D)
            for _ in range(self.scan_M):
                z = rng.standard_normal(D); z /= np.linalg.norm(z); sa += P.lg(w + s * z)[1] + P.lg(w - s * z)[1]
                z = rng.standard_normal(D); z /= np.linalg.norm(z); sb += P.lg(w + s * z)[1] + P.lg(w - s * z)[1]
            cost += 4 * self.scan_M
            if chosen is None and cos(g, sa + sb) < self.tau_div and cos(sa, sb) >= self.tau_noise: chosen = s
        return chosen, cost

    def _probe(self, w, P, U, ln0, lo0, m0, wo, rng):
        """Probe every ray at every ladder distance (forward passes only); best pixel wins, even through higher ones."""
        if self.use_batch and hasattr(P, "parts_batch") and self.rule == "mean":      # all rays x all distances in one pass
            R, K = len(U), len(self.ts)
            Wc = (w[None, None, :] + self.ts[None, :, None] * U[:, None, :]).reshape(R * K, -1)
            ln, lo = P.parts_batch(Wc); m = (1 - wo) * ln + wo * lo
            ok = np.isfinite(ln) & np.isfinite(lo) & (m < m0 - 1e-12)
            if not ok.any(): return None, np.inf, None, R * K / 3.0
            idx = int(np.argmin(np.where(ok, m, np.inf))); ui, ti = divmod(idx, K)
            return self.ts[ti] * U[ui], float(m[idx]), ui, R * K / 3.0
        best, best_key, best_i = None, np.inf, None
        for ui, u in enumerate(U):
            for t in self.ts:
                ln, lo = P.parts(w + t * u)
                if not (np.isfinite(ln) and np.isfinite(lo)): continue
                m = (1 - wo) * ln + wo * lo
                if self.rule == "constrained" and P.has_old:
                    if lo <= lo0 * (1 + self.tol) + 1e-3 and ln < ln0 - 1e-9 and ln < best_key: best, best_key, best_i = t * u, ln, ui
                elif m < m0 - 1e-12 and m < best_key: best, best_key, best_i = t * u, m, ui
        return best, best_key, best_i, len(U) * self.K / 3.0

    def step(self, w, P, rng):
        w2, cost = self._step(w, P, rng)
        if self.stuck_views is None: return w2, cost
        low = (not self._last_moved) or (self._last_gain < self.stuck_gain)
        self._stuck_count = self._stuck_count + 1 if low else 0
        if self._stuck_count >= self._patience and not self._in_retry:               # stuck: retry this step with the third axis added
            self.stats["stuck_retries"] += 1
            saved = (self.views, self.asm_mult); self.views, self.asm_mult = tuple(self.stuck_views), self.stuck_mult
            self._in_retry, self._force_wide = True, True
            try: w3, c3 = self._step(w2, P, rng, count=False)
            finally: self.views, self.asm_mult = saved; self._in_retry = False; self._force_wide = False
            won = self._last_moved and self._last_gain >= self.stuck_gain
            if won: self.stats["stuck_success"] += 1
            if self.stuck_backoff: self._patience = self.stuck_patience if won else min(self._patience * 2, 32)     # exponential backoff on failures
            self._stuck_count = 0
            return w3, cost + c3
        return w2, cost

    def _step(self, w, P, rng, count=True):
        D = len(w)
        if count: self.stats["steps"] += 1; self.task_steps += 1
        (ln0, gn), (lo0, go) = P.lg_parts(w); cost = 1.0; wo = P.wo
        g = (1 - wo) * gn + wo * go; m0 = (1 - wo) * ln0 + wo * lo0

        # 1. views ---------------------------------------------------------------------------------------------
        vecs, vnames = [], []
        for name in self.views:
            if name == "grad": vecs.append(-g); vnames.append(name)
            elif name == "gnew": vecs.append(-gn); vnames.append(name)
            elif name == "gold" and P.has_old: vecs.append(-go); vnames.append(name)
            elif name == "smooth":
                gs = np.zeros(D)
                for _ in range(self.M):                                  # antithetic: odd-order sampling noise cancels
                    u = rng.standard_normal(D); u /= np.linalg.norm(u)
                    gs += P.lg(w + self.scale * u)[1] + P.lg(w - self.scale * u)[1]
                cost += 2 * self.M; vecs.append(-gs); vnames.append(name)
            elif name == "mom" and self.mom is not None: vecs.append(self.mom); vnames.append(name)
            elif name in ("asmooth", "ssmooth"):                         # smoothed view at an AUTO-SENSED scale (scan result, or the step scale), x multiplier
                if name == "asmooth" and ((not self._scanned) or self.task_steps - self.last_scan >= self.scan_every):
                    self.scale_star, c_ = self._scan(w, P, g, rng); cost += c_; self.last_scan = self.task_steps; self.scan_log.append(self.scale_star); self._scanned = True
                base = self.scale_star if name == "asmooth" else self._step_scale()
                mult = self.asm_mult if name == "asmooth" else self.smooth_mult
                skip = self.skip_redundant and self._sm_streak >= self.skip_redundant and self.task_steps % self.skip_redundant != 0
                if base is not None and not skip:
                    sc = mult * base; gs = np.zeros(D)
                    for _ in range(self.M):
                        u = rng.standard_normal(D); u /= np.linalg.norm(u)
                        gs += P.lg(w + sc * u)[1] + P.lg(w - sc * u)[1]
                    cost += 2 * self.M; vecs.append(-gs); vnames.append(name); self.stats["sm_probes"] += 1
            elif name == "cross" and self.mom is not None:               # elementwise product of the two views: where gradient and momentum agree
                vecs.append((-g) * self.mom); vnames.append(name)
            elif name == "curv" and self.mom is not None and np.linalg.norm(self.mom) > 1e-12:   # H.m by finite difference: how the gradient changes along the motion
                e = self.mom / np.linalg.norm(self.mom); eps = self.curv_eps
                vecs.append(-(P.lg(w + eps * e)[1] - g) / eps); vnames.append(name); cost += 1.0

        # 2. Gram-Schmidt: keep only views that add a new direction -------------------------------------------------
        B, bnames = [], []
        for v, vn in zip(vecs, vnames):
            n = np.linalg.norm(v)
            if n < 1e-14: continue
            v = v / n
            for b in B: v = v - (v @ b) * b
            r = np.linalg.norm(v)
            if r >= self.min_resid: B.append(v / r); bnames.append(vn)
        if not B:
            v = rng.standard_normal(D); B = [v / np.linalg.norm(v)]; bnames = ["grad"]
        smn = [n for n in vnames if n in ("asmooth", "ssmooth")]
        if smn and "grad" in vnames:                                              # smoothed axis worth its probe? only if it and the gradient are BOTH independent directions
            kept = smn[0] in bnames and "grad" in bnames; self._sm_streak = 0 if kept else self._sm_streak + 1; self.stats["sm_kept"] += int(kept)

        # 3. screen: pixel grid over the span of the views ----------------------------------------------------------
        B = np.array(B).T; A = self._grid(B.shape[1])
        U = A @ B.T; U /= np.linalg.norm(U, axis=1, keepdims=True); self.stats["axes"] += B.shape[1]

        # 4. probe the screen (overridable): returns the best candidate move
        self._g = g
        best, best_key, best_i, pc = self._probe(w, P, U, ln0, lo0, m0, wo, rng); cost += pc
        if best is None and self._in_retry and self.stuck_dense:                  # coarse ladder found nothing: dense ladder on the pure axis rays
            best, best_key, best_i, pc2 = self._dense_probe(w, P, B, A, m0, wo); cost += pc2

        # 5. audit veto: shrink the step (x1, x1/2, x1/4) until the audit set is not degraded, else reject ----------
        if best is not None and P.audit is not None:
            la0 = P.audit_loss(w); cost += P.audit_cost; ok = None
            tol_, abs_ = ((self.early_tol, self.early_abs) if self.task_steps <= self.early_steps
                          else (self.audit_tol, self.audit_abs))
            for sh in (1.0, 0.5, 0.25):
                la = P.audit_loss(w + sh * best); cost += P.audit_cost
                if la <= la0 * (1 + tol_) + abs_: ok = sh; break
            if ok is None: best = None; self.stats["vetoed"] += 1
            else:
                if ok < 1.0: self.stats["shrunk"] += 1
                best = ok * best

        # 6. move, update momentum -------------------------------------------------------------------------------
        if best is not None:
            w2 = w + best; self.stats["accepted"] += 1
            self._last_moved, self._last_gain = True, (m0 - best_key) / max(abs(m0), 1e-12)
            nb = float(np.linalg.norm(best)); self._step_ema = nb if self._step_ema is None else 0.7 * self._step_ema + 0.3 * nb
            used = [bnames[k] for k in range(len(bnames)) if A[best_i, k] != 0]          # which views the winning pixel used
            for nm in used: self.stats["use_" + nm] = self.stats.get("use_" + nm, 0) + 1
            self.mom = self.mom_decay * (self.mom if self.mom is not None else 0.0) + (1 - self.mom_decay) * (w2 - w)
            return w2, cost
        self.stats["fail"] += 1; self._last_moved, self._last_gain = False, 0.0
        if self.mom is not None: self.mom = self.mom_decay * self.mom
        return w, cost


class MarchFan(Fan):
    """Fan whose rays MARCH outward and STOP where the gradient at two smoothing scales diverges.
    At each probed point p on a ray (distances tmin*ratio^k, ascending): loss (forward) plus the smoothed gradient at the
    small scale s_small and at the big scale s_big, using ONE antithetic pair each and the SAME random direction z
        g_s = g(p + s z) + g(p - s z).
    If cos(g_small, g_big) < tau the two scales disagree (the landscape changes structure at the big scale: an edge,
    a bend, a wall) and the ray stops there; otherwise it keeps going. The best-loss visited point over all rays is the
    candidate. Cost per probed point = 1/3 + 4 gradient evals, much more than the forward-only ladder."""
    def __init__(self, s_small=1.0, s_big=8.0, tau=0.9, K=8, ratio=3.0, rel_delta=None, **kw):
        """rel_delta: if set, stop when the agreement DROPS by more than rel_delta relative to the agreement at the ray's
        origin (measured once per step), instead of when it falls below the absolute threshold tau."""
        super().__init__(K=K, ratio=ratio, **kw)
        self.s_small, self.s_big, self.tau, self.rel_delta = s_small, s_big, tau, rel_delta
        self.stats.update(points=0, stops=0, rays=0, cos_sum=0.0)

    def _probe(self, w, P, U, ln0, lo0, m0, wo, rng):
        D = len(w); best, best_key, best_i, cost = None, np.inf, None, 0.0; a0 = None
        if self.rel_delta is not None:                               # agreement of the two scales at the origin of the rays
            z = rng.standard_normal(D); z /= np.linalg.norm(z)
            gs = P.lg(w + self.s_small * z)[1] + P.lg(w - self.s_small * z)[1]; gb = P.lg(w + self.s_big * z)[1] + P.lg(w - self.s_big * z)[1]; cost += 4.0
            a0 = float(gs @ gb / (np.linalg.norm(gs) * np.linalg.norm(gb) + 1e-30))
        for ui, u in enumerate(U):
            self.stats["rays"] += 1
            for t in self.ts:
                p = w + t * u
                ln, lo = P.parts(p); cost += 1 / 3.0; self.stats["points"] += 1
                if np.isfinite(ln) and np.isfinite(lo):
                    m = (1 - wo) * ln + wo * lo
                    if m < m0 - 1e-12 and m < best_key: best, best_key, best_i = t * u, m, ui
                z = rng.standard_normal(D); z /= np.linalg.norm(z)
                gs = P.lg(p + self.s_small * z)[1] + P.lg(p - self.s_small * z)[1]
                gb = P.lg(p + self.s_big * z)[1] + P.lg(p - self.s_big * z)[1]; cost += 4.0
                a = float(gs @ gb / (np.linalg.norm(gs) * np.linalg.norm(gb) + 1e-30)); self.stats["cos_sum"] += a
                if (a < self.tau) if a0 is None else (a < a0 - self.rel_delta): self.stats["stops"] += 1; break
        return best, best_key, best_i, cost


class CurvFan(Fan):
    """2-axis fan whose rays follow the curvature through a window of feature scales.
    * Scale scan: on the first step of each task (and every `scan_every` steps) the full wide ladder is probed (as in Fan)
      and the window centre h is set to the winning distance.
    * Window: afterwards each ray probes the 5 scales h*rho^k, k=-2..2 (forward only) and the centre re-centres on the winner.
    * Curvature is free to read: the directional derivative along a ray is d = g.u (g = mixed gradient at w), so one forward
      probe at scale t gives kappa = 2*(L(t) - L0 - d*t)/t^2.
    mode 'fixed': candidates are just the window scales ("fixed steps").
    mode 'stop' : (a) if the best scale is the top of the window and still improving, keep going x rho (up to max_ext) until
                  the loss rises = divergence; if the best is the bottom, look at smaller scales; (b) parabolic interpolation
                  through the 3 points around the low; (c) Newton stop t = -d/kappa from the smallest-scale curvature
                  (use_newton). All candidates are probed; the best wins."""
    def __init__(self, mode="stop", rho=3.0, max_ext=3, scan_every=25, h0=0.05, use_newton=True, freeze=False, views=("grad", "mom"), **kw):
        """freeze: control -- keep the window centre fixed at h0 (no wide scan, no re-centring)."""
        super().__init__(views=views, **kw)
        self.freeze = freeze
        self.mode, self.rho, self.max_ext, self.scan_every, self.h0, self.use_newton = mode, rho, max_ext, scan_every, h0, use_newton
        self.h, self.last_scan = None, 0
        self.stats.update(scans=0, bracketed=0, ext_up=0, ext_down=0, won_fixed=0, won_par=0, won_newton=0, rays=0, noimp=0)

    def _step_scale(self):
        return self.h if self.h is not None else self._step_ema

    @staticmethod
    def _parabola(x, y):
        (x1, x2, x3), (y1, y2, y3) = x, y; den = (x1 - x2) * (x1 - x3) * (x2 - x3)
        A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / den
        B = (x3 ** 2 * (y1 - y2) + x2 ** 2 * (y3 - y1) + x1 ** 2 * (y2 - y3)) / den
        return -B / (2 * A) if A > 1e-18 else None

    def _probe(self, w, P, U, ln0, lo0, m0, wo, rng):
        g = self._g; cost = 0.0; best_m, best_t, best_i, kind = np.inf, None, None, None
        def M(t, u):
            nonlocal cost
            ln, lo = P.parts(w + t * u); cost += 1 / 3.0
            return (1 - wo) * ln + wo * lo if (np.isfinite(ln) and np.isfinite(lo)) else np.inf
        if self.freeze and self.h is None: self.h = self.h0
        if not self.freeze and (self.h is None or self._force_wide or self.task_steps - self.last_scan >= self.scan_every):        # wide scan of feature scales
            self.stats["scans"] += 1; self.last_scan = self.task_steps
            if self.use_batch and hasattr(P, "parts_batch"):
                R, K = len(U), len(self.ts); Wc = (w[None, None, :] + self.ts[None, :, None] * U[:, None, :]).reshape(R * K, -1)
                ln, lo = P.parts_batch(Wc); mm = (1 - wo) * ln + wo * lo; cost += R * K / 3.0
                ok = np.isfinite(ln) & np.isfinite(lo) & (mm < m0 - 1e-12)
                if ok.any(): idx = int(np.argmin(np.where(ok, mm, np.inf))); ui, ti = divmod(idx, K); best_m, best_t, best_i = float(mm[idx]), self.ts[ti], ui
            else:
                for ui, u in enumerate(U):
                    for t in self.ts:
                        m = M(t, u)
                        if m < m0 - 1e-12 and m < best_m: best_m, best_t, best_i = m, t, ui
            if best_t is not None: self.h = float(best_t); self.scan_log.append(self.h)
            elif self.h is None: self.h = self.h0
            return (None if best_t is None else best_t * U[best_i]), best_m, best_i, cost
        pre = None
        if self.use_batch and hasattr(P, "parts_batch"):                       # the 5-scale window of every ray in one pass
            sc = self.h * self.rho ** np.arange(-2, 3); R = len(U)
            Wc = (w[None, None, :] + sc[None, :, None] * U[:, None, :]).reshape(R * 5, -1)
            ln, lo = P.parts_batch(Wc); pre = np.where(np.isfinite(ln) & np.isfinite(lo), (1 - wo) * ln + wo * lo, np.inf).reshape(R, 5); cost += R * 5 / 3.0
        for ui, u in enumerate(U):
            self.stats["rays"] += 1; d = float(g @ u)
            ts = list(self.h * self.rho ** np.arange(-2, 3)); vs = [M(t, u) for t in ts] if pre is None else [float(x) for x in pre[ui]]
            cands = [(v, t, "fixed") for v, t in zip(vs, ts)]
            if self.mode == "stop":
                k = int(np.argmin(vs)); ext = 0
                while k == len(ts) - 1 and vs[k] < m0 and ext < self.max_ext:            # still falling at the top: march on to the divergence
                    ts.append(ts[-1] * self.rho); vs.append(M(ts[-1], u)); k = int(np.argmin(vs)); ext += 1; self.stats["ext_up"] += 1
                    cands.append((vs[-1], ts[-1], "fixed"))
                ext = 0
                while k == 0 and vs[0] < m0 and ext < self.max_ext:                       # lowest probe is best: look below the window
                    ts.insert(0, ts[0] / self.rho); vs.insert(0, M(ts[0], u)); k = int(np.argmin(vs)); ext += 1; self.stats["ext_down"] += 1
                    cands.append((vs[0], ts[0], "fixed"))
                if 0 < k < len(ts) - 1:                                                   # bracketed low -> parabolic stop
                    self.stats["bracketed"] += 1
                    tp = self._parabola(ts[k - 1:k + 2], vs[k - 1:k + 2])
                    if tp is not None and ts[k - 1] < tp < ts[k + 1]: cands.append((M(tp, u), tp, "par"))
                if self.use_newton and d < 0:                                              # Newton stop from the smallest-scale curvature
                    kappa = 2 * (vs[0] - m0 - d * ts[0]) / ts[0] ** 2
                    if kappa > 1e-12:
                        tn = min(max(-d / kappa, ts[0] / self.rho), ts[-1] * self.rho)
                        cands.append((M(tn, u), tn, "newton"))
            for v, t, kd in cands:
                if v < m0 - 1e-12 and v < best_m: best_m, best_t, best_i, kind = v, t, ui, kd
        if best_t is None:
            self.stats["noimp"] += 1
            if not self.freeze: self.last_scan = -10 ** 9
            return None, np.inf, None, cost    # nothing found: rescan the wide ladder next step
        self.stats["won_" + kind] += 1
        if not self.freeze: self.h = float(best_t); self.scan_log.append(self.h)                                  # re-centre the window on the winner
        return best_t * U[best_i], best_m, best_i, cost


class CounterReplay:
    """Spaced-repetition replay over ALL stored problems (grows as tasks arrive).
    counter[i] rises each time problem i is re-tested without having degraded and resets to 0 when it has. Selection =
    lowest counters first (random tie-break): verified-stable problems wait until the others have caught up, a degraded
    problem jumps back to the front. ref_mode 'best': degraded if loss_i > best_loss_i*(1+tol_rel)+tol_abs (catches slow
    drift); 'prev': relative to the previous test of that problem."""
    def __init__(self, per_example_loss, ref_mode="best", tol_rel=0.25, tol_abs=0.02, reset_on_task=False, cap=None):
        """reset_on_task: zero every counter when a new task starts (old 'stable' records are stale once the model starts
        changing). cap: maximum counter value (bounded 'Leitner box'), so new problems can catch up quickly.
        Both off = the literal scheme, which starves older tasks in a growing buffer (see seq_cl.py results)."""
        self.ple, self.ref_mode, self.tol_rel, self.tol_abs = per_example_loss, ref_mode, tol_rel, tol_abs
        self.reset_on_task, self.cap = reset_on_task, cap
        self.X = self.y = self.counter = self.ref = None

    def new_task(self):
        if self.reset_on_task and self.counter is not None: self.counter[:] = 0

    def add(self, w, X, y):
        """Add a finished task's problems to the buffer, with their current losses as the reference."""
        ref = self.ple(w, X, y)
        if self.X is None: self.X, self.y, self.counter, self.ref = X, y, np.zeros(len(y), int), ref
        else:
            self.X, self.y = np.concatenate([self.X, X]), np.concatenate([self.y, y])
            self.counter, self.ref = np.concatenate([self.counter, np.zeros(len(y), int)]), np.concatenate([self.ref, ref])

    def select(self, rng, k):
        return np.argsort(self.counter + rng.random(len(self.y)) * 0.99)[:k]

    def update(self, j, losses):
        degraded = losses > self.ref[j] * (1 + self.tol_rel) + self.tol_abs
        self.counter[j] = np.where(degraded, 0, self.counter[j] + 1)
        if self.cap is not None: self.counter[j] = np.minimum(self.counter[j], self.cap)
        self.ref[j] = np.minimum(self.ref[j], losses) if self.ref_mode == "best" else losses
        return int(degraded.sum())


# ======================================================================================================================
# 2. NETWORK, DATA, PROBLEM WRAPPERS
# ======================================================================================================================
SHAPES = []


def set_net(in_dim, hidden):
    """tanh MLP in_dim -> hidden... -> 10 (flat parameter vector)."""
    global SHAPES
    sizes = [in_dim] + list(hidden) + [10]
    SHAPES = [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]


def init_params(rng):
    parts = []
    for a, b in SHAPES:
        parts.append((rng.standard_normal((a, b)) * np.sqrt(1.0 / a)).ravel()); parts.append(np.zeros(b))
    return np.concatenate(parts).astype(np.float64)


def unpack(w):
    out, i = [], 0
    for a, b in SHAPES:
        out.append((w[i:i + a * b].reshape(a, b), w[i + a * b:i + a * b + b])); i += a * b + b
    return out


def logits(w, X):
    h, L = X, unpack(w)
    for k, (W, b) in enumerate(L):
        h = h @ W + b
        if k < len(L) - 1: h = np.tanh(h)
    return h


def per_example_loss(w, X, y):
    h = logits(w, X); h = h - h.max(1, keepdims=True)
    return np.log(np.exp(h).sum(1)) - h[np.arange(len(y)), y]


def ce(w, X, y): return float(per_example_loss(w, X, y).mean())


def ce_grad(w, X, y):
    L = unpack(w); acts, h = [X], X
    for k, (W, b) in enumerate(L):
        h = h @ W + b
        if k < len(L) - 1: h = np.tanh(h)
        acts.append(h)
    z = h - h.max(1, keepdims=True); e = np.exp(z); p = e / e.sum(1, keepdims=True); n = len(y)
    loss = float(np.mean(np.log(e.sum(1)) - z[np.arange(n), y]))
    d = p.copy(); d[np.arange(n), y] -= 1.0; d /= n; grads = []
    for k in range(len(L) - 1, -1, -1):
        grads.append(d.sum(0)); grads.append((acts[k].T @ d).ravel())
        if k > 0: d = (d @ L[k][0].T) * (1.0 - acts[k] ** 2)
    return loss, np.concatenate(grads[::-1])


def ce_batch(Wc, X, y):
    """Mean cross-entropy of n candidate weight vectors Wc (n, D) on one batch: a single batched forward pass."""
    n = Wc.shape[0]; h = X; i = 0
    for k, (a, b) in enumerate(SHAPES):
        W = Wc[:, i:i + a * b].reshape(n, a, b); bias = Wc[:, i + a * b:i + a * b + b][:, None, :]; i += a * b + b
        h = np.matmul(h, W) + bias
        if k < len(SHAPES) - 1: h = np.tanh(h)
    z = h - h.max(-1, keepdims=True); lse = np.log(np.exp(z).sum(-1))
    zy = np.take_along_axis(z, np.broadcast_to(y[None, :, None], (n, len(y), 1)), axis=-1)[..., 0]
    return (lse - zy).mean(-1)


def acc(w, X, y): return float(np.mean(logits(w, X).argmax(1) == y))


class MixedBatch:
    """Current batch = new-task half + optional old (replay) half. The interface the fan optimisers talk to."""
    def __init__(self, w_old=0.5): self.w_old, self.has_old, self.audit = w_old, False, None

    def set(self, Xn, yn, Xo=None, yo=None):
        self.Xn, self.yn, self.Xo, self.yo = Xn, yn, Xo, yo; self.has_old = Xo is not None

    @property
    def wo(self): return self.w_old if self.has_old else 0.0

    def audit_loss(self, w): return ce(w, *self.audit)

    @property
    def audit_cost(self): return len(self.audit[1]) / 128 / 3.0

    def parts(self, w): return ce(w, self.Xn, self.yn), (ce(w, self.Xo, self.yo) if self.has_old else 0.0)

    def parts_batch(self, Wc):
        ln = ce_batch(Wc, self.Xn, self.yn)
        return ln, (ce_batch(Wc, self.Xo, self.yo) if self.has_old else np.zeros(len(Wc)))

    def lg_parts(self, w):
        ln, gn = ce_grad(w, self.Xn, self.yn)
        if not self.has_old: return (ln, gn), (0.0, np.zeros_like(gn))
        return (ln, gn), ce_grad(w, self.Xo, self.yo)

    def lg(self, w):
        (ln, gn), (lo, go) = self.lg_parts(w); wo = self.wo
        return (1 - wo) * ln + wo * lo, (1 - wo) * gn + wo * go

    def loss(self, w):
        ln, lo = self.parts(w); return (1 - self.wo) * ln + self.wo * lo


FMNIST_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"


def load_fashion(n_train=1000, n_test=500, seed=0):
    """Fashion-MNIST (zalandoresearch GitHub repo, downloaded once), 2x2 average-pooled to 14x14 = 196 inputs."""
    import gzip, urllib.request
    root = os.path.join(os.path.expanduser("~"), ".cache", "fashion_mnist"); os.makedirs(root, exist_ok=True)
    def get(name):
        p = os.path.join(root, name)
        if not os.path.exists(p): urllib.request.urlretrieve(FMNIST_URL + name, p)
        with gzip.open(p, "rb") as fh: return fh.read()
    imgs = lambda n: np.frombuffer(get(n), np.uint8, offset=16).reshape(-1, 28, 28)
    labs = lambda n: np.frombuffer(get(n), np.uint8, offset=8)
    Xtr, ytr, Xte, yte = imgs("train-images-idx3-ubyte.gz"), labs("train-labels-idx1-ubyte.gz"), imgs("t10k-images-idx3-ubyte.gz"), labs("t10k-labels-idx1-ubyte.gz")
    rng = np.random.default_rng(seed)
    pick = lambda y, n: np.concatenate([rng.choice(np.where(y == c)[0], n, replace=False) for c in range(10)])
    itr, ite = pick(ytr, n_train), pick(yte, n_test)
    pool = lambda X: X.reshape(-1, 14, 2, 14, 2).mean((2, 4)).reshape(len(X), 196) / 255.0
    Xa, Xb = pool(Xtr[itr]), pool(Xte[ite]); mu, sd = Xa.mean(0), Xa.std(0) + 0.1
    return (Xa - mu) / sd, ytr[itr], (Xb - mu) / sd, yte[ite]


def make_tasks(kind):
    if kind == "fashion":
        Xtr, ytr, Xte, yte = load_fashion(); tasks = []
        for c in range(0, 10, 2):
            mtr, mte = np.isin(ytr, [c, c + 1]), np.isin(yte, [c, c + 1])
            tasks.append(dict(Xtr=Xtr[mtr], ytr=ytr[mtr], Xte=Xte[mte], yte=yte[mte]))
        return tasks
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    X, y = load_digits(return_X_y=True); X = X / 16.0
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 0.1; f = lambda A: ((A - mu) / sd).astype(np.float64)
    Xtr, Xte = f(Xtr), f(Xte); tasks = []
    if kind == "split":
        for c in range(0, 10, 2):
            mtr, mte = np.isin(ytr, [c, c + 1]), np.isin(yte, [c, c + 1])
            tasks.append(dict(Xtr=Xtr[mtr], ytr=ytr[mtr], Xte=Xte[mte], yte=yte[mte]))
    else:
        for t in range(5):
            p = np.arange(64) if t == 0 else np.random.default_rng(t).permutation(64)
            tasks.append(dict(Xtr=Xtr[:, p], ytr=ytr, Xte=Xte[:, p], yte=yte))
    return tasks


# experiment name -> (task kind, input dim, hidden layers)
SEQUENCES = {"split": ("split", 64, [64, 64]), "perm": ("perm", 64, [64, 64]), "perm16": ("perm", 64, [16, 16]), "fashion": ("fashion", 196, [64, 64])}


# ======================================================================================================================
# 3. METHOD REGISTRY (continual-learning sequences)
#    opt: adam | fan;  replay: None | rand | counter;  audit: veto steps that hurt a set of old problems;  sched: CounterReplay options
# ======================================================================================================================
_CR = dict(reset_on_task=True)                      # counter replay with the per-task counter reset (the literal scheme starves old tasks)
METHODS = {
    "ft":           dict(opt="adam", lr=0.003, replay=None),
    "adam_rand":    dict(opt="adam", lr=0.01, replay="rand"),
    "adam_cnt":     dict(opt="adam", lr=0.01, replay="counter", sched=_CR),
    "adam_cnt_lit": dict(opt="adam", lr=0.01, replay="counter", sched=dict()),
    "fan2":         dict(opt="fan", replay="counter", audit=True, fan_hp=dict(views=("grad", "mom")), sched=_CR),
    "curv_fixed":   dict(opt="fan", fan_cls="curv", replay="counter", audit=True, fan_hp=dict(mode="fixed"), sched=_CR),
    "curv_stop":    dict(opt="fan", fan_cls="curv", replay="counter", audit=True, fan_hp=dict(mode="stop"), sched=_CR),
    "curv_rand":    dict(opt="fan", fan_cls="curv", replay="rand", audit=True, fan_hp=dict(mode="fixed"), sched=dict()),      # random replay + random audit
    "frozen_a":     dict(opt="fan", fan_cls="curv", replay="counter", audit=True, fan_hp=dict(mode="fixed", freeze=True, h0=1.2), sched=_CR),    # window frozen at 1.2
    "frozen_c":     dict(opt="fan", fan_cls="curv", replay="counter", audit=True, fan_hp=dict(mode="fixed", freeze=True, h0=0.05), sched=_CR),
    "frozen_d":     dict(opt="fan", fan_cls="curv", replay="counter", audit=True, fan_hp=dict(mode="fixed", freeze=True, h0=10.0), sched=_CR),
    "march_1v8":    dict(opt="fan", fan_cls="march", replay="counter", audit=True, fan_hp=dict(views=("grad", "mom"), s_small=1.0, s_big=8.0, tau=0.9), sched=_CR),
    "march_1v16":   dict(opt="fan", fan_cls="march", replay="counter", audit=True, fan_hp=dict(views=("grad", "mom"), s_small=1.0, s_big=16.0, tau=0.9), sched=_CR),
    "march_1v8_rel":  dict(opt="fan", fan_cls="march", replay="counter", audit=True, fan_hp=dict(views=("grad", "mom"), s_small=1.0, s_big=8.0, rel_delta=0.1), sched=_CR),
    "march_1v16_rel": dict(opt="fan", fan_cls="march", replay="counter", audit=True, fan_hp=dict(views=("grad", "mom"), s_small=1.0, s_big=16.0, rel_delta=0.1), sched=_CR),
    "joint":        dict(opt="joint"),                    # Adam on all tasks mixed: capacity ceiling (no loss trace)
}
# third-axis variants <base>+<kind><mult>: smoothed-gradient axis whose smoothing scale is auto-sensed (ss: typical STEP scale, as: SCAN-found
# scale) times mult (1 or 2); the smoothed view goes FIRST so it stays an exact ray; skipped for a while when redundant with the gradient
for _base, (_cls, _hp0) in (("fan", (None, {})), ("curv_fixed", ("curv", dict(mode="fixed"))), ("curv_stop", ("curv", dict(mode="stop")))):
    for _kind, _view in (("ss", "ssmooth"), ("as", "asmooth")):
        for _mult in (1, 2):
            _hp = dict(_hp0, views=(_view, "grad", "mom"), skip_redundant=5); _hp["smooth_mult" if _kind == "ss" else "asm_mult"] = float(_mult)
            METHODS[f"{_base}+{_kind}{_mult}"] = dict(opt="fan", fan_cls=_cls, replay="counter", audit=True, fan_hp=_hp, sched=_CR)

# STUCK-TRIGGERED third axis: normal steps use the cheap 2-axis screen; after `stuck_patience` stuck steps (no improving pixel, or relative gain < 1e-4)
# the same step is retried with a smoothed-gradient axis (smoothing scale = stuck_mult x the scan-sensed scale) added, a fresh wide scale scan, and a DENSE
# distance ladder on the pure axis rays (a coarse ladder can step straight over a narrow improving region). Failed retries back off exponentially.
#   +stuck  (2x scale)  +stuck1 (1x)  +stuck_noaxis (dense retry, no third axis: ablation)  +stuck_coarse (third axis, coarse ladder: ablation)
for _base, (_cls, _hp0) in (("fan2", (None, dict(views=("grad", "mom")))), ("curv_fixed", ("curv", dict(mode="fixed"))), ("curv_stop", ("curv", dict(mode="stop")))):
    _asm = ("asmooth", "grad", "mom")
    for _nm, _kw in (("stuck", dict(stuck_views=_asm, stuck_mult=2.0)), ("stuck1", dict(stuck_views=_asm, stuck_mult=1.0)),
                     ("stuck_noaxis", dict(stuck_views=("grad", "mom"))), ("stuck_coarse", dict(stuck_views=_asm, stuck_mult=2.0, stuck_dense=False))):
        METHODS[f"{_base}+{_nm}"] = dict(opt="fan", fan_cls=_cls, replay="counter", audit=True, fan_hp=dict(_hp0, **_kw), sched=_CR)

NEW_BS, AUDIT_BS = 64, 64


def make_opt(spec):
    if spec["opt"] == "adam": return Adam(spec["lr"])
    cls = {"march": MarchFan, "curv": CurvFan}.get(spec.get("fan_cls"), Fan)
    return cls(**spec.get("fan_hp", {}))


def line_barrier(w0, w1, X, y, n=21):
    """Old-task loss barrier on the straight line between the weights before and after a task: max(loss) - max(endpoint losses)."""
    L = np.array([ce(w0 + l * (w1 - w0), X, y) for l in np.linspace(0, 1, n)])
    return float(L.max() - max(L[0], L[-1]))


# ======================================================================================================================
# 4. CONTINUAL-LEARNING SEQUENCE RUNNER  (records LOSS and accuracy vs evals while training)
# ======================================================================================================================
def run_sequence(method, tasks, seed, budget, old_bs, record_every=100):
    """Task 0: Adam, 400 steps. Each later task: `budget` gradient-eval equivalents, 64 new + old_bs replayed examples per step.
    Returns final accuracy matrix and a trace of (evals, loss_all_seen, loss_current, loss_earlier, mean_test_acc) records.
    Loss is the mean cross-entropy on a fixed subset (<=400 train examples per task) -- measured outside the compute budget."""
    rng = np.random.default_rng(seed); T = len(tasks); w = init_params(rng); P = MixedBatch()
    ev_rng = np.random.default_rng(777 + seed)                                  # separate RNG: recording never changes the experiment
    subs = [ev_rng.choice(len(t["ytr"]), min(400, len(t["ytr"])), replace=False) for t in tasks]
    Xs = [t["Xtr"][s] for t, s in zip(tasks, subs)]; ys = [t["ytr"][s] for t, s in zip(tasks, subs)]
    accs = lambda w_, t_: [acc(w_, tasks[i]["Xte"], tasks[i]["yte"]) for i in range(t_ + 1)]
    trace = []

    def record(w_, t_, x):
        Ls = [ce(w_, Xs[i], ys[i]) for i in range(t_ + 1)]; As = accs(w_, t_)
        trace.append((float(x), float(np.mean(Ls)), Ls[t_], float(np.mean(Ls[:t_])) if t_ > 0 else float("nan"), float(np.mean(As))))
        return As

    opt = Adam(0.01); X0, y0 = tasks[0]["Xtr"], tasks[0]["ytr"]; record(w, 0, 0)          # task 0: Adam
    for s in range(400):
        i = rng.integers(0, len(y0), 128); P.set(X0[i], y0[i]); w, _c = opt.step(w, P, rng)
        if (s + 1) % record_every == 0: record(w, 0, s + 1)
    final_acc = np.full((T, T), np.nan); res = dict(dips=[], steps=[], barriers=[], task_stats=[])
    final_acc[0, :1] = accs(w, 0)
    if method == "joint":
        Xall = np.concatenate([t["Xtr"] for t in tasks]); yall = np.concatenate([t["ytr"] for t in tasks]); opt = Adam(0.01); ev = 0.0
        while ev < budget * (T - 1):
            i = rng.integers(0, len(yall), 128); P.set(Xall[i], yall[i]); w, c = opt.step(w, P, rng); ev += c
        final_acc[T - 1, :] = accs(w, T - 1); res.update(final_acc=final_acc.tolist(), trace=None); return res
    spec = METHODS[method]
    sched = CounterReplay(per_example_loss, **spec["sched"]) if spec["replay"] == "counter" else None
    bufX, bufy = [X0], [y0]
    if sched: sched.add(w, X0, y0)
    for t in range(1, T):
        Xn, yn = tasks[t]["Xtr"], tasks[t]["ytr"]; w_before = w.copy()
        if sched: sched.new_task()
        opt = make_opt(spec); bm = spec.get("bs_mult", 1); inner = spec.get("inner", 1)          # batch multiplier (Adam batch study): bm x the samples per step, bm x the cost
        ev, nxt, steps, dip = 0.0, 0.0, 0, 1.0
        x0 = 400 + (t - 1) * budget
        if spec["replay"] == "rand": OX, Oy = np.concatenate(bufX), np.concatenate(bufy)
        while ev < budget:
            if ev >= nxt:
                nxt += record_every
                As = record(w, t, x0 + ev); dip = min(dip, float(np.mean(As[:t])))
            if spec["replay"] is None:
                i = rng.integers(0, len(yn), 2 * NEW_BS * bm); P.set(Xn[i], yn[i]); P.audit = None
            else:
                i = rng.integers(0, len(yn), NEW_BS * bm)
                if spec["replay"] == "rand": j = rng.integers(0, len(Oy), old_bs * bm); Xo, yo = OX[j], Oy[j]
                else:
                    j = sched.select(rng, old_bs * bm); sched.update(j, per_example_loss(w, sched.X[j], sched.y[j])); ev += old_bs * bm / 128 / 3.0
                    Xo, yo = sched.X[j], sched.y[j]
                P.set(Xn[i], yn[i], Xo, yo)
                if spec.get("audit") and spec["replay"] == "rand":
                    ja = rng.integers(0, len(Oy), AUDIT_BS); ev += AUDIT_BS / 128 / 3.0; P.audit = (OX[ja], Oy[ja])
                elif spec.get("audit"):
                    ja = sched.select(rng, AUDIT_BS); sched.update(ja, per_example_loss(w, sched.X[ja], sched.y[ja])); ev += AUDIT_BS / 128 / 3.0
                    P.audit = (sched.X[ja], sched.y[ja])
                else: P.audit = None
            for _ in range(inner): w, c = opt.step(w, P, rng); ev += c * bm        # `inner` > 1: the SAME batch is reused for several steps (Adam data-reuse control)
            steps += 1                                                             # steps = batches drawn
        final_acc[t, :t + 1] = record(w, t, x0 + ev)
        Xold = np.concatenate([tasks[k]["Xtr"] for k in range(t)]); yold = np.concatenate([tasks[k]["ytr"] for k in range(t)])
        sub = rng.choice(len(yold), min(800, len(yold)), replace=False)             # (draws from the experiment RNG, as in the original runs)
        res["barriers"].append(line_barrier(w_before, w, Xold[sub], yold[sub])); res["dips"].append(dip); res["steps"].append(steps)
        if spec["opt"] == "fan": res["task_stats"].append(dict(opt.stats))
        if sched: sched.add(w, Xn, yn)
        else: bufX.append(Xn); bufy.append(yn)
    res.update(final_acc=final_acc.tolist(), trace=trace, evals_per_step=budget / max(float(np.mean(res["steps"])), 1.0))
    return res


# ======================================================================================================================
# 5. KNOWN LANDSCAPES (ground truth), embedded in D dimensions with a random rotation
#    corridor  narrow winding valley sloping to a goal           | ridge     two basins separated by a thin high wall
#    side_open deeper basin in a direction the gradient does not reveal (no wall)
#    barrier   two-task landscape: old well, joint well (slightly worse for the old task), broad hill between; Adam cannot cross
# ======================================================================================================================
H, WIDTH, ALPHA_R = 8.0, 0.1, 1.0
A_AMP, K_FREQ, P_GOAL, ALPHA_C = 0.8, 2.0, 6.0, 0.5
L_, H_, BETA, A2, ALPHA = 6.0, 2.0, 0.1, 0.3, 1.0


class Landscape:
    def __init__(self, kind, D, seed):
        rng = np.random.default_rng(10_000 + seed); Q, _ = np.linalg.qr(rng.standard_normal((D, D))); self.Q, self.D, self.kind = Q, D, kind

    def _fg(self, z):
        if self.kind == "ridge":
            s, perp = z[0], z[1:]; e = np.exp(-s * s / (2 * WIDTH ** 2))
            f = (s * s - 4) ** 2 / 32 - 0.15 * s + H * e + 0.5 * ALPHA_R * float(perp @ perp)
            gs = (s * s - 4) * s / 8 - 0.15 - H * s / WIDTH ** 2 * e
            return f, np.concatenate([[gs], ALPHA_R * perp])
        if self.kind == "side_open":                  # shallow bowl at the origin; compact-support well (radius 1.4, zero gradient outside) 3 away
            c = np.zeros_like(z); c[0] = 3.0; R2 = 1.4 ** 2
            d2 = float((z - c) @ (z - c)); u = max(0.0, 1.0 - d2 / R2)
            f = 0.05 * float(z @ z) - 1.5 * u * u
            g = 0.1 * z + (6.0 * u * (z - c) / R2 if u > 0 else 0.0)
            return f, g
        p, q, perp = z[0], z[1], z[2:]                # corridor
        r = q - A_AMP * np.sin(K_FREQ * p)
        f = 10 * r * r + 0.05 * (p - P_GOAL) ** 2 + 0.5 * ALPHA_C * float(perp @ perp)
        gp = 20 * r * (-A_AMP * K_FREQ * np.cos(K_FREQ * p)) + 0.1 * (p - P_GOAL)
        return f, np.concatenate([[gp, 20 * r], ALPHA_C * perp])

    def loss(self, x): return self._fg(self.Q.T @ x)[0]

    def lg(self, x): f, gz = self._fg(self.Q.T @ x); return f, self.Q @ gz

    def init(self, rng):
        D = self.D
        if self.kind == "side_open": z = 0.3 * rng.standard_normal(D)
        elif self.kind == "ridge": z = np.concatenate([[-2.4 + 0.1 * rng.standard_normal()], 0.5 * rng.standard_normal(D - 1)])
        else: z = np.concatenate([[0.2 + 0.1 * rng.standard_normal(), 0.3 * rng.standard_normal()], 0.3 * rng.standard_normal(D - 2)])
        return self.Q @ z

    def solved(self, x):
        z = self.Q.T @ x; f = self._fg(z)[0]
        if self.kind == "side_open": return f < -0.5
        return (z[0] > 0 and f < -0.15) if self.kind == "ridge" else f < 0.02


class Single:
    """Adapter so the fan optimisers (which expect new/old parts) can run on a single-objective landscape."""
    has_old, wo, audit = False, 0.0, None
    def __init__(self, land): self.land = land
    def lg_parts(self, x): l, g = self.land.lg(x); return (l, g), (0.0, np.zeros_like(g))
    def parts(self, x): return self.land.loss(x), 0.0
    def lg(self, x): return self.land.lg(x)
    def loss(self, x): return self.land.loss(x)
    def init(self, rng): return self.land.init(rng)
    def solved(self, x): return self.land.solved(x)
    def objective(self, x): return self.land.loss(x)


class TwoTask:
    """Mixed (replay) objective 0.5 f_A + 0.5 f_B has a LOCAL minimum at the old solution and a barrier to the joint optimum."""
    has_old = True; w_old = 0.5
    def __init__(self, D, seed):
        rng = np.random.default_rng(20_000 + seed); self.Q, _ = np.linalg.qr(rng.standard_normal((D, D))); self.D = D
        self.audit = None; self.audit_cost = 0.1                    # no audit (a tight audit blocks the crossing)
    @property
    def wo(self): return 0.5
    def _fa(self, z):
        s, perp = z[0], z[1:]; u = s / L_
        hill = H_ * (4 * u * (1 - u)) ** 2; dh = H_ * 2 * (4 * u * (1 - u)) * 4 * (1 - 2 * u) / L_
        ss = u * u * (3 - 2 * u); dss = 6 * u * (1 - u) / L_
        return hill + A2 * ss + 0.5 * ALPHA * float(perp @ perp), np.concatenate([[dh + A2 * dss], ALPHA * perp])
    def _fb(self, z):
        s, perp = z[0], z[1:]
        return 0.5 * BETA * ((s - L_) ** 2 + float(perp @ perp)), BETA * np.concatenate([[s - L_], perp])
    def lg_parts(self, w):
        z = self.Q.T @ w; fb, gb = self._fb(z); fa, ga = self._fa(z); return (fb, self.Q @ gb), (fa, self.Q @ ga)
    def parts(self, w): z = self.Q.T @ w; return self._fb(z)[0], self._fa(z)[0]
    def lg(self, w): (fb, gb), (fa, ga) = self.lg_parts(w); return 0.5 * (fb + fa), 0.5 * (gb + ga)
    def loss(self, w): fb, fa = self.parts(w); return 0.5 * (fb + fa)
    def audit_loss(self, w): return self._fa(self.Q.T @ w)[0]
    def init(self, rng): return self.Q @ np.concatenate([[0.0], 0.3 * rng.standard_normal(self.D - 1)])
    def solved(self, w):
        z = self.Q.T @ w; fb, fa = self._fb(z)[0], self._fa(z)[0]; return z[0] > L_ / 2 and fa + fb < 0.6
    def objective(self, w): fb, fa = self.parts(w); return fa + fb      # the joint objective f_A + f_B


LAND_KINDS = ["corridor", "ridge", "side_open", "barrier"]
LAND_METHODS = ["adam", "fan2", "curv_fixed", "curv_stop", "curv_fixed+as2", "fan2+stuck", "curv_fixed+stuck"]


def make_land(kind, D, seed): return TwoTask(D, seed) if kind == "barrier" else Single(Landscape(kind, D, seed))


def make_land_opt(name, lr=0.03):
    if name == "adam": return Adam(lr)
    if name == "fan2": return Fan(views=("grad", "mom"))
    if name == "fan3_s2": return Fan(views=("grad", "smooth", "mom"), scale=2.0, tmin=0.01, ratio=2.0)      # hand-tuned 3-view fan (scale chosen by hand)
    if "+" in name or name in ("curv_fixed", "curv_stop"):
        spec = METHODS[name]; return make_opt(spec)
    raise ValueError(name)


def run_landscape(job):
    kind, name, lr, D, seed, budget = job
    P = make_land(kind, D, seed); opt = make_land_opt(name, lr); rng = np.random.default_rng(seed); x = P.init(rng); ev = 0.0; steps = 0
    te, to = [0.0], [P.objective(x)]
    while ev < budget:
        if P.solved(x): break
        x, c = opt.step(x, P, rng); ev += c; steps += 1; te.append(ev); to.append(P.objective(x))
    ok = bool(P.solved(x)); return dict(te=te, to=to, ok=ok, evals=ev if ok else None, steps=steps if ok else None)


# ======================================================================================================================
# 6. PARALLEL HELPER, PLOTTING, REPORTING, CLI
# ======================================================================================================================
def _init_worker():
    try:
        from threadpoolctl import threadpool_limits; threadpool_limits(1)
    except Exception:
        os.environ["OMP_NUM_THREADS"] = "1"


def pmap(fn, items, workers=1):
    """Independent jobs in worker processes (fork). workers: 1 = serial, 0 = all cores. Results come back in job order."""
    items = list(items)
    if workers == 0: workers = os.cpu_count() or 1
    if workers <= 1 or len(items) <= 1 or not hasattr(os, "fork"): return [fn(x) for x in items]
    with mp.get_context("fork").Pool(min(workers, len(items)), initializer=_init_worker) as pool: return pool.map(fn, items, chunksize=1)


_CTX = {}


def _seq_job(job):
    m, s = job; return run_sequence(m, _CTX["tasks"], s, _CTX["budget"], _CTX["old_bs"])


def resample(xs, vs, grid):
    """Previous-value lookup of a step trace onto a grid (NaN before the first record)."""
    xs = np.asarray(xs, float); vs = np.asarray(vs, float); idx = np.searchsorted(xs, grid, side="right") - 1
    return np.where(idx >= 0, vs[np.clip(idx, 0, None)], np.nan)


COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#000000", "#aec7e8"]


def plot_sequence(name, results, budget, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    methods = [m for m in results if results[m][0]["trace"] is not None]
    T = len(results[methods[0]][0]["final_acc"]); xmax = 400 + (T - 1) * budget; grid = np.arange(0, xmax + 1, 25.0)
    panels = [(1, "loss on ALL tasks seen so far", True), (2, "loss on the CURRENT task", True), (3, "loss on EARLIER tasks", True), (4, "mean test accuracy (tasks seen so far)", False)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9)); axes = axes.ravel()
    for ci, m in enumerate(methods):
        for ax, (col, title, logy) in zip(axes, panels):
            Y = np.array([resample([r[0] for r in run["trace"]], [r[col] for r in run["trace"]], grid) for run in results[m]])
            with np.errstate(all="ignore"): mu = np.nanmean(Y, 0); sd = np.nanstd(Y, 0)
            ax.plot(grid, mu, color=COLORS[ci % len(COLORS)], lw=1.8, label=m)
            ax.fill_between(grid, np.maximum(mu - sd, 1e-6 if logy else -1), mu + sd, color=COLORS[ci % len(COLORS)], alpha=0.12)
    for ax, (col, title, logy) in zip(axes, panels):
        for k in range(T): ax.axvline(400 + (k) * budget if k > 0 else 400, color="k", ls=":", lw=0.8)
        ax.set_title(title); ax.set_xlabel("evals (gradient-evaluation equivalents)"); ax.grid(alpha=0.3)
        if logy: ax.set_yscale("log")
    axes[0].legend(fontsize=8, ncol=2); fig.suptitle(f"{name}: training loss and accuracy vs evals (dotted lines = task starts; first 400 evals = Adam on task 1)")
    plt.tight_layout(); path = os.path.join(out, f"loss_vs_evals_{name}.png"); plt.savefig(path, dpi=130); plt.close(); return path


def print_sequence_table(name, results, budget):
    print(f"\n=== {name}: final results (mean +- std over seeds) ===")
    print(f"{'method':22s} {'final acc':>14s} {'forgetting':>11s} {'loss all tasks':>15s} {'min old acc':>12s} {'steps/task':>11s} {'evals/step':>11s} {'stuck retries/step (won)':>25s}")
    for m, runs in results.items():
        FA = [np.array(r["final_acc"]) for r in runs]; acc_ = np.array([f[-1].mean() for f in FA]); T = FA[0].shape[0]
        forget = np.mean([np.mean([f[i, i] - f[-1, i] for i in range(T - 1)]) for f in FA])
        if runs[0]["trace"] is None: print(f"{m:22s} {acc_.mean():7.3f}±{acc_.std():.3f}   (capacity ceiling)"); continue
        L = np.array([r["trace"][-1][1] for r in runs]); dips = np.mean([np.mean(r["dips"]) for r in runs]); st = np.mean([np.mean(r["steps"]) for r in runs])
        S = [s for r in runs for s in r["task_stats"]]; rt = ""
        if S and sum(s.get("stuck_retries", 0) for s in S): rt = f"{sum(s['stuck_retries'] for s in S) / max(sum(s['steps'] for s in S), 1):.3f} ({sum(s['stuck_success'] for s in S) / max(sum(s['stuck_retries'] for s in S), 1):.0%})"
        print(f"{m:22s} {acc_.mean():7.3f}±{acc_.std():.3f} {forget:11.3f} {L.mean():9.3f}±{L.std():.3f} {dips:12.3f} {st:11.0f} {budget / st:11.1f} {rt:>25s}")


def to_jsonable(o):
    if isinstance(o, dict): return {k: to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, (np.floating, np.integer)): return o.item()
    return o


def run_sequences(exp, methods, seeds, budget, old_bs, workers, out):
    kind, in_dim, hidden = SEQUENCES[exp]
    try:
        set_net(in_dim, hidden); tasks = make_tasks(kind)
    except Exception as e:
        print(f"[{exp}] skipped: could not load data ({e})"); return
    _CTX.update(tasks=tasks, budget=budget, old_bs=old_bs); t0 = time.time()
    outs = pmap(_seq_job, [(m, s) for m in methods for s in range(seeds)], workers)
    results = {m: outs[i * seeds:(i + 1) * seeds] for i, m in enumerate(methods)}
    print(f"[{exp}] {len(methods)} methods x {seeds} seeds x {len(tasks)} tasks done in {time.time()-t0:.0f}s (workers={workers})")
    print_sequence_table(exp, results, budget)
    path = plot_sequence(exp, results, budget, out); print("wrote", path)
    json.dump(to_jsonable(results), open(os.path.join(out, f"results_{exp}.json"), "w"))


def run_landscapes(methods, dims, seeds, budget, plot_dim, workers, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    t0 = time.time(); res = {}; lr_best = {}
    print(f"\n=== landscapes: success rate (median evals | median steps), {seeds} seeds, budget {budget:.0f} ===")
    print(f"{'landscape / method':26s}" + "".join(f"{'D=' + str(D):>20s}" for D in dims))
    for kind in LAND_KINDS:
        if "adam" in methods and kind != "barrier":          # small learning-rate grid for Adam on this landscape (D = plot_dim, 3 seeds)
            lr_best[kind] = max((0.01, 0.03, 0.1, 0.3), key=lambda lr: sum(run_landscape((kind, "adam", lr, plot_dim, 500 + s, budget))["ok"] for s in range(3)))
        else: lr_best[kind] = 0.03
        for m in methods:
            line = ""
            for D in dims:
                rs = pmap(run_landscape, [(kind, m, lr_best[kind], D, s, budget) for s in range(seeds)], workers); res[(kind, m, D)] = rs
                ok = [r for r in rs if r["ok"]]
                line += f"{len(ok)/len(rs):4.2f} ({np.median([r['evals'] for r in ok]) if ok else float('nan'):6.0f}|{np.median([r['steps'] for r in ok]) if ok else float('nan'):4.0f})".rjust(20)
            print(f"{kind + ' / ' + m:26s}{line}", flush=True)
    print(f"({time.time()-t0:.0f}s)")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9)); axes = axes.ravel()
    for ax, kind in zip(axes, LAND_KINDS):
        grid = np.unique(np.round(np.geomspace(1, budget, 220)))
        for ci, m in enumerate(methods):
            rs = res[(kind, m, plot_dim)]
            Y = np.array([resample(r["te"], r["to"], grid) for r in rs]); q1, med, q3 = np.nanpercentile(Y, [25, 50, 75], axis=0)
            ax.plot(grid, med, color=COLORS[ci % len(COLORS)], lw=1.8, label=f"{m} ({sum(r['ok'] for r in rs)}/{len(rs)} solved)")
            ax.fill_between(grid, q1, q3, color=COLORS[ci % len(COLORS)], alpha=0.12)
        ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=0.05); ax.set_title(f"{kind} (D={plot_dim})" + ("  objective = f_A + f_B" if kind == "barrier" else ""))
        ax.set_xlabel("evals (gradient-evaluation equivalents)"); ax.set_ylabel("objective"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle("known landscapes: objective vs evals (median over seeds, band = inter-quartile range; runs stop when solved)")
    plt.tight_layout(); path = os.path.join(out, "landscape_loss_vs_evals.png"); plt.savefig(path, dpi=130); plt.close(); print("wrote", path)
    json.dump({f"{k}|{m}|{D}": [{kk: vv for kk, vv in r.items() if kk in ('ok', 'evals', 'steps')} for r in rs] for (k, m, D), rs in res.items()}, open(os.path.join(out, "results_landscapes.json"), "w"), default=float)


# ======================================================================================================================
# 7. BATCH / DATA-REUSE STUDY.  Same total compute per task for everyone. Question: what is the best way to spend a per-step budget of
#    ~15-30 evals on ONE drawn batch? (a) bigger batch for Adam (practical batch sizes: x4 = 320, x32 = 2560 samples/step), (b) Adam REUSING the
#    batch for several steps (equal training items AND equal evals as the fan), (c) the fan's probes (also on a x4 batch).
#    Learning rates are tuned per variant on separate seeds. 'items' = training examples drawn per task (new + replayed), 'batches' = steps.
# ======================================================================================================================
ADAM_VARIANTS = {          # label: (batch multiplier, inner steps per batch, learning-rate grid)
    "adam batch x1":  (1, 1, (0.003, 0.01, 0.03)), "adam batch x4": (4, 1, (0.01, 0.03, 0.1)),
    "adam batch x16": (16, 1, (0.01, 0.03, 0.1)), "adam batch x32": (32, 1, (0.03, 0.1, 0.3)),
    "adam reuse x4":  (1, 4, (0.001, 0.003, 0.01)), "adam reuse x16": (1, 16, (0.001, 0.003, 0.01)), "adam reuse x32": (1, 32, (0.001, 0.003, 0.01)),
}
FAN_VARIANTS = {"fan2": ("fan2", 1), "curv_fixed": ("curv_fixed", 1), "fan2 batch x4": ("fan2", 4), "curv_fixed batch x4": ("curv_fixed", 4)}


def run_batch_study(exp, seeds, budget, old_bs, workers, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    kind, in_dim, hidden = SEQUENCES[exp]
    try:
        set_net(in_dim, hidden); tasks = make_tasks(kind)
    except Exception as e:
        print(f"[{exp}] skipped: {e}"); return
    _CTX.update(tasks=tasks, budget=budget, old_bs=old_bs); t0 = time.time()
    cand = {}
    for lab, (bm, inn, lrs) in ADAM_VARIANTS.items():
        for lr in lrs: METHODS[f"{lab}|{lr}"] = dict(opt="adam", lr=lr, replay="rand", bs_mult=bm, inner=inn); cand.setdefault(lab, []).append(f"{lab}|{lr}")
    for lab, (base, bm) in FAN_VARIANTS.items(): METHODS[lab] = dict(METHODS[base], bs_mult=bm)
    names = [n for lab in cand for n in cand[lab]]; tune_seeds = [100, 101]
    outs = pmap(_seq_job, [(n, s) for n in names for s in tune_seeds], workers)
    score = {n: np.mean([np.mean(np.array(r["final_acc"])[-1]) for r in outs[k * 2:(k + 1) * 2]]) for k, n in enumerate(names)}
    best = {lab: max(cand[lab], key=lambda n: score[n]) for lab in cand}
    final = [(lab, best[lab]) for lab in ADAM_VARIANTS] + [(lab, lab) for lab in FAN_VARIANTS]
    outs = pmap(_seq_job, [(n, s) for _, n in final for s in range(seeds)], workers)
    results = {lab + (f" (lr {METHODS[n]['lr']})" if n in score else ""): outs[i * seeds:(i + 1) * seeds] for i, (lab, n) in enumerate(final)}
    print(f"\n=== batch / data-reuse study, {exp}: mean +- std over {seeds} seeds; {budget:.0f} evals per task for every method ===")
    print(f"{'method':34s} {'evals/batch':>11s} {'batches/task':>13s} {'items/task':>11s} {'final acc':>14s}")
    pts = {}
    for (lab, n), (name, runs) in zip(final, results.items()):
        spec = METHODS[n]; acc_ = np.array([np.mean(np.array(r["final_acc"])[-1]) for r in runs]); st = np.mean([np.mean(r["steps"]) for r in runs])
        items = st * (NEW_BS + old_bs) * spec.get("bs_mult", 1)
        print(f"{name:34s} {budget / st:11.1f} {st:13.0f} {items:11.0f} {acc_.mean():7.3f}±{acc_.std():.3f}")
        pts[name] = (items, st, acc_.mean(), acc_.std(), "fan" if spec["opt"] == "fan" else "adam")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, (idx, xl) in zip(axes, ((0, "training items drawn per task (new + replayed)"), (1, "batches (sequential steps) per task"))):
        for name, v in pts.items():
            ax.errorbar([v[idx]], [v[2]], [v[3]], marker="o" if v[4] == "adam" else "s", ms=8, ls="", color="#1f77b4" if v[4] == "adam" else "#d62728")
            ax.annotate(name.split(" (")[0], (v[idx], v[2]), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.set_xscale("log"); ax.set_xlabel(xl); ax.set_ylabel("final mean accuracy"); ax.grid(alpha=0.3)
    axes[0].set_title("blue = Adam variants, red = fan variants (same compute per task)"); fig.suptitle(f"{exp}: accuracy vs data drawn and vs sequential steps")
    plt.tight_layout(); path = os.path.join(out, f"batch_study_{exp}.png"); plt.savefig(path, dpi=130); plt.close(); print("wrote", path)
    keep = [k for k in results if k.startswith(("adam batch x1 ", "adam batch x32", "adam reuse x16", "adam reuse x32")) or k in ("fan2", "curv_fixed")]
    print("wrote", plot_sequence(f"batch_{exp}", {k: results[k] for k in keep}, budget, out), f"({time.time()-t0:.0f}s)")
    json.dump(to_jsonable({k: [{kk: vv for kk, vv in r.items() if kk != 'trace'} for r in v] for k, v in results.items()}), open(os.path.join(out, f"results_batch_{exp}.json"), "w"))


def main():
    ap = argparse.ArgumentParser(description="Replicate the fan experiments and draw loss-vs-evals graphs (see the docstring at the top of this file).")
    ap.add_argument("--exp", nargs="+", default=["split", "landscapes"], choices=list(SEQUENCES) + ["landscapes", "batch", "all"])
    ap.add_argument("--full", action="store_true", help="replication settings: 8 seeds, 3000 evals/task, all main methods, landscapes at D=2,8,32")
    ap.add_argument("--seeds", type=int); ap.add_argument("--budget", type=float, help="gradient-eval equivalents per task")
    ap.add_argument("--old-bs", type=int, default=16, help="replayed examples per step")
    ap.add_argument("--methods", help="comma list (see --list-methods)"); ap.add_argument("--workers", type=int, default=1, help="processes; 1 = serial, 0 = all cores")
    ap.add_argument("--out", default="fan_results"); ap.add_argument("--land-dim", type=int, default=8, help="dimension used for the landscape loss curves")
    ap.add_argument("--batch-seqs", nargs="+", default=["perm"], choices=list(SEQUENCES), help="sequences for --exp batch")
    ap.add_argument("--list-methods", action="store_true"); a = ap.parse_args()
    if a.list_methods:
        print("sequence methods:", ", ".join(METHODS)); print("landscape methods:", ", ".join(LAND_METHODS + ["fan3_s2"] + [m for m in METHODS if "+" in m])); return
    os.makedirs(a.out, exist_ok=True)
    exps = list(SEQUENCES) + ["landscapes"] if "all" in a.exp else a.exp
    full = a.full
    seeds = a.seeds or (8 if full else 2); budget = a.budget or (3000 if full else 1500)
    default_methods = ("ft,adam_rand,adam_cnt,fan2,curv_fixed,curv_stop,curv_fixed+as2,fan2+stuck,curv_fixed+stuck,curv_stop+stuck,joint" if full else "ft,adam_rand,adam_cnt,fan2,curv_fixed,curv_fixed+stuck")
    methods = (a.methods or default_methods).split(",")
    bad = [m for m in methods if m not in METHODS and m not in ("adam", "fan3_s2")]
    if bad: sys.exit(f"unknown methods {bad}; see --list-methods")
    print(f"settings: {'FULL' if full else 'QUICK'} | experiments {exps} | seeds {seeds} | budget {budget:.0f} evals/task | methods {methods}")
    for e in exps:
        if e in SEQUENCES: run_sequences(e, methods, seeds, budget, a.old_bs, a.workers, a.out)
    if "batch" in exps:
        for e in a.batch_seqs: run_batch_study(e, seeds, budget, a.old_bs, a.workers, a.out)
    if "landscapes" in exps:
        lm = [m for m in methods if m in ("adam", "fan2", "curv_fixed", "curv_stop", "fan3_s2") or "+" in m] if a.methods else LAND_METHODS
        if not a.methods or "adam" not in lm: lm = (["adam"] if "adam" not in lm else []) + lm
        dims = [2, 8, 32] if full else [a.land_dim]
        run_landscapes(lm, dims, a.seeds or (8 if full else 4), 15000 if full else 8000, a.land_dim, a.workers, a.out)


if __name__ == "__main__":
    main()
