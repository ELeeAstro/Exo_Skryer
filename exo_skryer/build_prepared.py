"""
build_prepared.py
=================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import ndtr as _phi


__all__ = [
    "Prepared",
    "make_loglik_only",
    "make_unpack_u_to_theta",
    "make_loglik_u",
    "build_prepared"
]


@dataclass(frozen=True)
class Bijector:
    forward: Callable[[jnp.ndarray], jnp.ndarray]
    inverse: Callable[[jnp.ndarray], jnp.ndarray]
    log_abs_det_jac: Callable[[jnp.ndarray], jnp.ndarray]


def _identity_bijector() -> Bijector:
    return Bijector(
        forward=lambda u: u,
        inverse=lambda x: x,
        log_abs_det_jac=lambda u: jnp.zeros_like(u),
    )


def _log_bijector() -> Bijector:
    return Bijector(
        forward=lambda u: jnp.exp(u),
        inverse=lambda x: jnp.log(x),
        log_abs_det_jac=lambda u: u,
    )


def _logit_bijector(low: float, high: float) -> Bijector:
    lower = float(low)
    upper = float(high)
    width = upper - lower
    if not width > 0:
        raise ValueError("logit bounds must satisfy high > low")

    def forward(u):
        s = jax.nn.sigmoid(u)
        return lower + width * s

    def inverse(x):
        y = (x - lower) / width
        y = jnp.clip(y, 1e-12, 1 - 1e-12)
        return jnp.log(y) - jnp.log1p(-y)

    def log_det(u):
        s = jax.nn.sigmoid(u)
        s = jnp.clip(s, 1e-12, 1 - 1e-12)
        return jnp.log(width) + jnp.log(s) + jnp.log1p(-s)

    return Bijector(forward=forward, inverse=inverse, log_abs_det_jac=log_det)


def _whitened_log_bijector(low: float, high: float) -> Bijector:
    lower = float(low)
    upper = float(high)
    if not (upper > lower > 0):
        raise ValueError("log-whitened bounds must satisfy 0 < low < high")

    log_low = jnp.log(lower)
    log_high = jnp.log(upper)
    mean = 0.5 * (log_low + log_high)
    scale = 0.5 * (log_high - log_low)

    def forward(u):
        return jnp.exp(mean + scale * u)

    def inverse(theta):
        return (jnp.log(theta) - mean) / scale

    def log_det(u):
        theta = forward(u)
        return jnp.log(jnp.abs(scale)) + jnp.log(theta)

    return Bijector(forward=forward, inverse=inverse, log_abs_det_jac=log_det)


def _normal_logpdf(x, mu, sigma):
    z = (x - mu) / sigma
    return -0.5 * (z * z + jnp.log(2 * jnp.pi) + 2 * jnp.log(sigma))


def _lognormal_logpdf(x, mu, sigma):
    return jnp.where(
        x > 0,
        _normal_logpdf(jnp.log(x), mu, sigma) - jnp.log(x),
        -jnp.inf,
    )


def _uniform_logpdf(x, low, high):
    inside = (x >= low) & (x <= high)
    return jnp.where(inside, -jnp.log(high - low), -jnp.inf)


def _log_uniform_logpdf(x, low, high):
    inside = (x >= low) & (x <= high) & (x > 0)
    norm = jnp.log(high) - jnp.log(low)
    return jnp.where(inside, -jnp.log(x) - norm, -jnp.inf)

def _evaluate_prior_logpdf(dist: str, theta, params: dict) -> jnp.ndarray:
    selector = dist.lower()
    if selector in ("gaussian", "normal"):
        return _normal_logpdf(theta, params["mu"], params["sigma"])
    if selector == "lognormal":
        return _lognormal_logpdf(theta, params["mu"], params["sigma"])
    if selector == "uniform":
        return _uniform_logpdf(theta, params["low"], params["high"])
    if selector == "log_uniform":
        return _log_uniform_logpdf(theta, params["low"], params["high"])
    if selector == "delta":
        return jnp.array(0.0)
    return jnp.array(-jnp.inf)


@dataclass(frozen=True)
class Prepared:
    names: Tuple[str, ...]
    bijectors: Tuple[Bijector, ...]
    priors: Tuple[Tuple[str, dict], ...]
    init_u: jnp.ndarray
    logprob: Callable[[jnp.ndarray], jnp.ndarray]
    loglik: Callable[[jnp.ndarray], jnp.ndarray]
    logprior: Callable[[jnp.ndarray], jnp.ndarray]
    fixed: Dict[str, jnp.ndarray]
    lam: jnp.ndarray
    dlam: jnp.ndarray
    y: jnp.ndarray
    dy: jnp.ndarray
    dy_p: jnp.ndarray
    dy_m: jnp.ndarray
    fm: Callable[[Dict[str, jnp.ndarray]], jnp.ndarray]

    def unpack(self, u_vec: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        values = {self.names[i]: self.bijectors[i].forward(u_vec[i]) for i in range(len(self.names))}
        values.update(self.fixed)
        return values


def _infer_transform(dist: str, param) -> str:
    requested = getattr(param, "transform", None)
    if requested is not None:
        value = str(requested).lower()
        if value in {"identity", "log", "logit"}:
            return value
        raise ValueError(f"Unknown transform '{requested}' for param '{getattr(param,'name','?')}'")
    selector = dist.lower()
    if selector in {"uniform", "truncnormal"}:
        return "logit"
    if selector == "log_uniform":
        return "log"
    return "identity"


def _default_init(dist: str, param) -> Optional[float]:
    if getattr(param, "init", None) is not None:
        return float(param.init)
    selector = dist.lower()
    if selector in {"gaussian", "normal"}:
        return float(getattr(param, "mu"))
    if selector in {"uniform"}:
        low = float(getattr(param, "low"))
        high = float(getattr(param, "high"))
        return 0.5 * (low + high)
    if selector == "log_uniform":
        low = float(getattr(param, "low"))
        high = float(getattr(param, "high"))
        return float(jnp.sqrt(low * high))
    if selector == "delta":
        value = getattr(param, "value", getattr(param, "init", None))
        if value is None:
            raise ValueError(f"delta param '{getattr(param,'name','?')}' needs 'value' or 'init'")
        return float(value)
    if selector == "lognormal":
        return float(jnp.exp(getattr(param, "mu")))
    return None


def _canonical_prior_params(dist: str, param) -> dict:
    selector = dist.lower()
    if selector in {"gaussian", "normal"}:
        return {"mu": float(param.mu), "sigma": float(param.sigma)}
    if selector in {"uniform", "log_uniform"}:
        return {"low": float(param.low), "high": float(param.high)}
    if selector == "lognormal":
        return {"mu": float(param.mu), "sigma": float(param.sigma)}
    if selector == "delta":
        return {}
    raise ValueError(f"Unsupported dist '{dist}' for param '{getattr(param,'name','?')}'")


def make_loglik_only(prep: Prepared):
    lam = prep.lam
    dlam = prep.dlam
    y = prep.y
    dy = prep.dy
    dy_p = prep.dy_p
    dy_m = prep.dy_m
    fm = prep.fm

    def loglik_theta(theta_map: Dict[str, jnp.ndarray]):
        mu = fm(theta_map)
        res = y - mu
        sig = jnp.where(res >= 0.0, dy_p, dy_m)
        sig = jnp.clip(sig, 1e-300, jnp.inf)
        is_finite = jnp.all(jnp.isfinite(mu))

        def _ok(_):
            norm = jnp.clip(dy_p + dy_m, 1e-300, jnp.inf)
            r = res / sig
            r = jnp.where(jnp.isfinite(r), r, 0.0)
            logC = 0.5 * jnp.log(2.0 / jnp.pi) - jnp.log(norm)
            return jnp.sum(logC - 0.5 * (r * r))

        def _bad(_):
            return -jnp.inf

        return lax.cond(is_finite, _ok, _bad, operand=None)

    return loglik_theta


def make_unpack_u_to_theta(prep: Prepared):
    def unpack(u_vec):
        return prep.unpack(u_vec)

    return unpack


def make_loglik_u(prep: Prepared):
    unpack = make_unpack_u_to_theta(prep)
    loglik_theta = make_loglik_only(prep)

    def loglik_u(u_vec):
        theta_map = unpack(u_vec)
        return loglik_theta(theta_map)

    return loglik_u


def build_prepared(
    cfg,
    obs: dict,
    fm: Callable[[Dict[str, jnp.ndarray]], jnp.ndarray],
) -> Prepared:
    if fm is None:
        raise ValueError("build_prepared requires an explicit predict_fn.")
    params_cfg = getattr(cfg, "params", None)
    if params_cfg is None:
        raise ValueError("cfg.params must be defined in the YAML configuration.")

    names_free: List[str] = []
    bijectors: List[Bijector] = []
    prior_defs: List[Tuple[str, dict]] = []
    init_values: List[float] = []
    fixed: Dict[str, jnp.ndarray] = {}

    for param in params_cfg:
        name = getattr(param, "name", None)
        dist = getattr(param, "dist", "").lower()
        if not name or not dist:
            raise ValueError("Each parameter needs 'name' and 'dist'")

        transform = _infer_transform(dist, param)
        if transform == "identity":
            bij = _identity_bijector()
        elif transform == "log":
            low = getattr(param, "low", None)
            high = getattr(param, "high", None)
            if low is not None and high is not None:
                bij = _whitened_log_bijector(float(low), float(high))
            else:
                bij = _log_bijector()
        elif transform == "logit":
            low = float(getattr(param, "low"))
            high = float(getattr(param, "high"))
            bij = _logit_bijector(low, high)
        else:
            raise ValueError(f"Unknown transform '{transform}' for param '{name}'")

        prior_params = _canonical_prior_params(dist, param)
        init_theta = _default_init(dist, param)
        if init_theta is None:
            raise ValueError(f"Param '{name}' requires 'init' or an inferable default for dist '{dist}'")
        init_arr = jnp.asarray(init_theta)

        if dist == "delta":
            fixed[name] = init_arr
            continue

        names_free.append(name)
        bijectors.append(bij)
        prior_defs.append((dist, prior_params))
        init_values.append(float(bij.inverse(init_arr)))

    names_tuple = tuple(names_free)
    bijectors_tuple = tuple(bijectors)
    priors_tuple = tuple(prior_defs)
    init_u = jnp.asarray(init_values)

    lam = jnp.asarray(obs["wl"])
    dlam = jnp.asarray(obs["dwl"])
    y = jnp.asarray(obs["y"])
    dy = jnp.asarray(obs["dy"])
    dy_p = jnp.asarray(obs.get("dy_plus", obs["dy"]))
    dy_m = jnp.asarray(obs.get("dy_minus", obs["dy"]))

    def _unpack_all(u_vec: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        values = {names_tuple[i]: bijectors_tuple[i].forward(u_vec[i]) for i in range(len(names_tuple))}
        values.update(fixed)
        return values

    def loglik_u(u_vec: jnp.ndarray) -> jnp.ndarray:
        pars = _unpack_all(u_vec)
        mu = fm(pars)
        res = y - mu
        sig = jnp.where(res >= 0.0, dy_p, dy_m)
        sig = jnp.clip(sig, 1e-300, jnp.inf)
        is_finite = jnp.all(jnp.isfinite(mu))

        def _ok(_):
            norm = jnp.clip(dy_p + dy_m, 1e-300, jnp.inf)
            r = res / sig
            r = jnp.where(jnp.isfinite(r), r, 0.0)
            logC = 0.5 * jnp.log(2.0 / jnp.pi) - jnp.log(norm)
            return jnp.sum(logC - 0.5 * (r * r))

        def _bad(_):
            return -jnp.inf

        return lax.cond(is_finite, _ok, _bad, operand=None)

    def logprior_u(u_vec: jnp.ndarray) -> jnp.ndarray:
        logp = 0.0
        for i, (dist_i, params_i) in enumerate(priors_tuple):
            theta_i = bijectors_tuple[i].forward(u_vec[i])
            logp += _evaluate_prior_logpdf(dist_i, theta_i, params_i)
            logp += bijectors_tuple[i].log_abs_det_jac(u_vec[i])
        return logp

    def logprob(u_vec: jnp.ndarray) -> jnp.ndarray:
        return loglik_u(u_vec) + logprior_u(u_vec)

    return Prepared(
        names=names_tuple,
        bijectors=bijectors_tuple,
        priors=priors_tuple,
        init_u=init_u,
        logprob=logprob,
        loglik=loglik_u,
        logprior=logprior_u,
        fixed=fixed,
        lam=lam,
        dlam=dlam,
        y=y,
        dy=dy,
        dy_p=dy_p,
        dy_m=dy_m,
        fm=fm,
    )
