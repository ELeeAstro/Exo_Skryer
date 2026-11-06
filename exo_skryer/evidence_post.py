"""
evidence_post.py
================

Evidence from posterior samples, independently of the sampler that drew them.

A proposal q is fitted to the pooled samples of one or more runs:

- ``importance`` draws fresh points from q and reweights them by p/q. It only
  needs q to cover the posterior, so pooling runs that reached different regions
  also corrects their mode weights.
- ``bridge`` and ``harmonic`` assume the samples are true posterior draws, so
  they run on each source alone. ``harmonic`` is the sensitive check: a source
  with wrong mode weights gives a value that disagrees with ``importance``.
  ``bridge`` leans on q, so it barely reacts to the same error.

Run after one or more retrievals of the same YAML setup::

    python -m exo_skryer.evidence_post --config retrieval_config.yaml \\
        --posterior nessai=run1/posterior.nc --posterior nautilus=run2/posterior.nc \\
        --split R_p=1.345

The ``harmonic`` method needs the pixi ``evidence`` environment.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import argparse
import json

import numpy as np
from scipy import stats as sps
from scipy.special import logsumexp

__all__ = [
    "MixtureProposal",
    "bridge_evidence",
    "evaluate_evidence",
    "harmonic_evidence",
    "importance_evidence",
    "load_posterior",
    "pareto_k",
]

_FLOORED = -1e250  # below this, the likelihood hit LOG_FLOOR
_PARETO_K_LIMIT = 0.7


@dataclass
class MixtureProposal:
    """Mixture of multivariate-t components, with a broad defensive component."""

    log_weights: np.ndarray
    components: List[Any]

    @classmethod
    def fit(
        cls,
        u: np.ndarray,
        n_components: int = 8,
        df: float = 5.0,
        inflate: float = 1.5,
        defensive: float = 0.1,
        seed: int = 0,
    ) -> "MixtureProposal":
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(n_components, covariance_type="full", reg_covar=1e-6, random_state=seed).fit(u)
        components = [
            sps.multivariate_t(loc=mean, shape=inflate * cov, df=df)
            for mean, cov in zip(gmm.means_, gmm.covariances_)
        ]
        components.append(sps.multivariate_t(loc=u.mean(axis=0), shape=4.0 * np.cov(u, rowvar=False), df=df))
        weights = np.append((1.0 - defensive) * gmm.weights_, defensive)
        return cls(np.log(weights), components)

    def logpdf(self, u: np.ndarray) -> np.ndarray:
        terms = [lw + np.atleast_1d(c.logpdf(u)) for lw, c in zip(self.log_weights, self.components)]
        return logsumexp(np.stack(terms), axis=0)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        counts = rng.multinomial(n, np.exp(self.log_weights - logsumexp(self.log_weights)))
        draws = [c.rvs(size=k, random_state=rng).reshape(k, -1) for c, k in zip(self.components, counts) if k]
        return rng.permutation(np.concatenate(draws))


def pareto_k(log_w: np.ndarray) -> float:
    """Generalised-Pareto tail index of the weights; above 0.7 the estimate is unreliable."""
    n_tail = int(min(0.2 * log_w.size, 3.0 * np.sqrt(log_w.size)))
    w = np.exp(np.sort(log_w) - np.max(log_w))
    threshold = w[-n_tail - 1]
    k, _, _ = sps.genpareto.fit(w[-n_tail:] - threshold, floc=0.0)
    return float(k)


def _ess(log_w: np.ndarray) -> float:
    return float(np.exp(2.0 * logsumexp(log_w) - logsumexp(2.0 * log_w)))


def importance_evidence(log_w: np.ndarray) -> Dict[str, float]:
    """Summarise importance weights log(p/q) of draws from q."""
    n = log_w.size
    ess = _ess(log_w)
    return {
        "logZ": float(logsumexp(log_w) - np.log(n)),
        "logZ_err": float(np.sqrt(max(1.0 / ess - 1.0 / n, 0.0))),
        "ESS": ess,
        "pareto_k": pareto_k(log_w),
    }


def bridge_evidence(
    log_ratio_post: np.ndarray,
    log_ratio_q: np.ndarray,
    n_eff_post: float,
    log_z0: float,
    max_iterations: int = 200,
    tol: float = 1e-8,
) -> float:
    """
    Meng & Wong optimal bridge, with log_ratio = log(p/q) at the posterior draws
    and at the draws from q. n_eff_post counts the independent posterior draws.
    """
    n_q = log_ratio_q.size
    log_s1 = np.log(n_eff_post / (n_eff_post + n_q))
    log_s2 = np.log(n_q / (n_eff_post + n_q))
    log_z = log_z0
    for _ in range(max_iterations):
        numerator = logsumexp(log_ratio_q - np.logaddexp(log_s1 + log_ratio_q, log_s2 + log_z)) - np.log(n_q)
        denominator = logsumexp(-np.logaddexp(log_s1 + log_ratio_post, log_s2 + log_z)) - np.log(log_ratio_post.size)
        previous, log_z = log_z, numerator - denominator
        if abs(log_z - previous) < tol:
            break
    return float(log_z)


def harmonic_evidence(
    u: np.ndarray,
    log_p: np.ndarray,
    n_chains: int = 20,
    temperature: float = 0.8,
    epochs: int = 20,
    seed: int = 0,
) -> Dict[str, float]:
    """Learned harmonic mean from posterior draws, split into pseudo-chains."""
    import harmonic as hm

    n_per_chain = u.shape[0] // n_chains
    order = np.random.default_rng(seed).permutation(u.shape[0])[: n_chains * n_per_chain]
    chains = hm.Chains(u.shape[1])
    chains.add_chains_3d(
        u[order].reshape(n_chains, n_per_chain, -1),
        log_p[order].reshape(n_chains, n_per_chain),
    )
    train, infer = hm.utils.split_data(chains, training_proportion=0.5)
    flow = hm.model.RQSplineModel(u.shape[1], standardize=True, temperature=temperature)
    flow.fit(train.samples, epochs=epochs, verbose=False)
    evidence = hm.Evidence(infer.nchains, flow)
    evidence.add_chains(infer)
    low, high = evidence.compute_ln_inv_evidence_errors()
    return {"logZ": float(-evidence.ln_evidence_inv), "logZ_err": float(0.5 * (abs(low) + abs(high)))}


def evaluate_evidence(
    logprob: Callable[[np.ndarray], np.ndarray],
    sources: Dict[str, np.ndarray],
    n_proposal: int = 20000,
    n_components: int = 8,
    use_harmonic: bool = False,
    seed: int = 0,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Estimate the evidence from sources {name: (n, d) u-space samples}.

    logprob returns log(L * prior) in u-space for an (n, d) array. Returns
    (report, u_q, log_w), with the proposal draws and their importance weights.
    """
    rng = np.random.default_rng(seed)

    # Each source gets an equal share of the fit, and keeps a held-out half.
    fit_rows, held_out = [], {}
    n_fit = min(u.shape[0] // 2 for u in sources.values())
    for name, u in sources.items():
        shuffled = rng.permutation(u)
        fit_rows.append(shuffled[:n_fit])
        held_out[name] = shuffled[n_fit:]
    proposal = MixtureProposal.fit(np.concatenate(fit_rows), n_components=n_components, seed=seed)

    u_q = proposal.sample(n_proposal, rng)
    log_p_q = logprob(u_q)
    log_w = log_p_q - proposal.logpdf(u_q)
    report: Dict[str, Any] = {
        "importance": importance_evidence(log_w),
        "n_floored": int(np.sum(log_p_q < _FLOORED)),
        "n_proposal": n_proposal,
        "sources": {},
    }
    log_z_is = report["importance"]["logZ"]

    for name, u in held_out.items():
        log_p = logprob(u)
        n_unique = np.unique(u, axis=0).shape[0]
        entry = {
            "bridge_logZ": bridge_evidence(log_p - proposal.logpdf(u), log_w, n_unique, log_z_is),
            "n_unique": int(n_unique),
            "n_floored": int(np.sum(log_p < _FLOORED)),
        }
        if use_harmonic:
            entry["harmonic"] = harmonic_evidence(u, log_p, seed=seed)
        report["sources"][name] = entry

    return report, u_q, log_w


def weighted_fraction(values: np.ndarray, threshold: float, log_w: np.ndarray) -> Dict[str, float]:
    """Importance-weighted posterior fraction below a threshold."""
    w = np.exp(log_w - logsumexp(log_w))
    fraction = float(np.sum(w[values < threshold]))
    return {
        "fraction_below": fraction,
        "err": float(np.sqrt(max(fraction * (1.0 - fraction), 0.0) / _ess(log_w))),
    }


def load_posterior(path: Path, names: Sequence[str]) -> np.ndarray:
    """Read equal-weight samples of the named parameters from a posterior.nc."""
    import arviz as az

    data = az.from_netcdf(str(path))
    posterior = data.posterior if hasattr(data, "posterior") else data["posterior"]
    return np.stack([np.asarray(posterior[name]).ravel() for name in names], axis=1)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evidence from the posteriors of existing runs.")
    parser.add_argument("--config", required=True, type=Path, help="YAML setup shared by the runs")
    parser.add_argument("--posterior", required=True, action="append", metavar="NAME=PATH",
                        help="posterior.nc of one run; repeat to pool runs")
    parser.add_argument("--split", action="append", default=[], metavar="PARAM=VALUE",
                        help="report the corrected posterior fraction below VALUE; repeatable")
    parser.add_argument("--n-proposal", type=int, default=20000)
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=256, help="likelihood batch size")
    parser.add_argument("--harmonic", action="store_true", help="add the learned harmonic mean per source")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="report path; default evidence_post.json beside the YAML")
    args = parser.parse_args(argv)

    from .help_runtime import apply_runtime_env
    from .read_yaml import read_yaml

    config = args.config.resolve()
    exp_dir = config.parent
    cfg = read_yaml(config)
    apply_runtime_env(cfg)  # before JAX starts

    from jax import config as jax_config
    jax_config.update("jax_enable_x64", True)

    from .retrieval_target import batched, build_target
    from .run_retrieval import build_retrieval_inputs

    obs, fm = build_retrieval_inputs(cfg, exp_dir)
    target = build_target(cfg, obs, fm)
    logprob = batched(target.logprob_u, args.chunk)

    sources = {}
    for item in args.posterior:
        name, path = item.split("=", 1)
        sources[name] = target.theta_to_u(load_posterior(Path(path), target.names))

    report, u_q, log_w = evaluate_evidence(
        logprob, sources, n_proposal=args.n_proposal, n_components=args.n_components,
        use_harmonic=args.harmonic, seed=args.seed,
    )

    theta_q = target.u_to_theta(u_q)
    report["splits"] = {}
    for item in args.split:
        name, value = item.split("=", 1)
        report["splits"][item] = weighted_fraction(theta_q[:, target.names.index(name)], float(value), log_w)

    importance = report["importance"]
    print(f"[evidence] importance logZ: {importance['logZ']:.3f} ± {importance['logZ_err']:.3f}")
    print(f"[evidence] ESS: {importance['ESS']:.0f} of {args.n_proposal}, pareto k: {importance['pareto_k']:.2f}")
    if importance["pareto_k"] > _PARETO_K_LIMIT:
        print(f"[evidence] Warning: pareto k is above {_PARETO_K_LIMIT}, so the importance estimate is unreliable.")
    for name, entry in report["sources"].items():
        print(f"[evidence] {name}: bridge logZ {entry['bridge_logZ']:.3f}, {entry['n_unique']} unique draws")
        if "harmonic" in entry:
            print(f"[evidence] {name}: harmonic logZ {entry['harmonic']['logZ']:.3f} ± {entry['harmonic']['logZ_err']:.3f}")
    for item, entry in report["splits"].items():
        print(f"[evidence] fraction below {item}: {entry['fraction_below']:.3f} ± {entry['err']:.3f}")

    output = args.output or exp_dir / "evidence_post.json"
    output.write_text(json.dumps(report, indent=2))
    print(f"[evidence] Report -> {output}")


if __name__ == "__main__":
    main()
