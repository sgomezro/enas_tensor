from __future__ import annotations

import math

import torch

from source.wann_engine import Population, compile_population, wann_forward


# ----------------------------------------------------------------------------
# Environment: CartPole swing-up (as used in the WANN paper), vectorized in torch
# ----------------------------------------------------------------------------
class SwingUp:
    g, m_c, m_p, l, force_mag, dt, b = 9.82, 0.5, 0.5, 0.6, 10.0, 0.01, 0.1
    x_threshold = 2.4
    n_obs, n_act = 5, 1

    @classmethod
    def reset(cls, R, device, gen, dtype=torch.float32):
        mean = torch.tensor([0.0, 0.0, math.pi, 0.0], device=device, dtype=dtype)
        return mean + 0.2 * torch.randn(R, 4, device=device, generator=gen, dtype=dtype)

    @staticmethod
    def observe(s):
        x, x_dot, th, th_dot = s.unbind(-1)
        return torch.stack([x, x_dot, torch.cos(th), torch.sin(th), th_dot], -1)

    @classmethod
    def step(cls, s, action):
        x, x_dot, th, th_dot = s.unbind(-1)
        a = action.clamp(-1.0, 1.0) * cls.force_mag
        m_tot, m_p_l = cls.m_c + cls.m_p, cls.m_p * cls.l
        sn, cs = torch.sin(th), torch.cos(th)
        xacc = ((-2 * m_p_l * th_dot**2 * sn + 3 * cls.m_p * cls.g * sn * cs
                 + 4 * a - 4 * cls.b * x_dot) / (4 * m_tot - 3 * cls.m_p * cs**2))
        thacc = ((-3 * m_p_l * th_dot**2 * sn * cs + 6 * m_tot * cls.g * sn
                  + 6 * (a - cls.b * x_dot) * cs) / (4 * cls.l * m_tot - 3 * m_p_l * cs**2))
        x = x + x_dot * cls.dt
        th = th + th_dot * cls.dt
        x_dot = x_dot + xacc * cls.dt
        th_dot = th_dot + thacc * cls.dt
        s = torch.stack([x, x_dot, th, th_dot], -1)
        out = x.abs() > cls.x_threshold
        reward = 0.5 * (torch.cos(th) + 1) * torch.cos(0.5 * math.pi * x / cls.x_threshold)
        return s, reward, out


def rollout_step(state, alive, ret, adj_f, act_masks, depth_masks, wcol, out_slice):
    """One policy+env step for the whole batch. Pure tensor code (compile-friendly)."""
    action = wann_forward(SwingUp.observe(state), adj_f, act_masks, depth_masks,
                          wcol, out_slice)[..., 0]
    state, reward, out = SwingUp.step(state, action)
    ret = ret + reward * alive
    alive = alive & ~out
    return state, alive, ret


def evaluate(pop: Population, weights: torch.Tensor, n_episodes: int, steps: int,
             seed: int, step_fn=rollout_step, dtype=torch.float32,
             theta: torch.Tensor | None = None) -> torch.Tensor:
    """Returns (P, W) mean episodic return for each genome at each shared weight.
    If theta (P,N,N) is given, per-edge weights are used (pass weights=[1.0])."""
    dev = pop.adj.device
    P, W, B = pop.size, weights.numel(), n_episodes
    R = W * B
    comp = compile_population(pop, dtype)
    if theta is not None:
        comp.adj_f = comp.adj_f * theta.to(dtype)
    gen = torch.Generator(device=dev).manual_seed(seed)
    s0 = SwingUp.reset(B, dev, gen, dtype)                      # common random numbers:
    state = s0.repeat(W, 1)[None].expand(P, R, 4).contiguous()   # same starts for all
    wcol = weights.to(dtype).repeat_interleave(B)[None, :, None]
    alive = torch.ones(P, R, dtype=torch.bool, device=dev)
    ret = torch.zeros(P, R, dtype=dtype, device=dev)
    for t in range(steps):
        state, alive, ret = step_fn(state, alive, ret, comp.adj_f, comp.act_masks,
                                    comp.depth_masks, wcol, comp.out_idx)
        if t % 100 == 99 and not bool(alive.any()):              # rare host sync
            break
    return ret.view(P, W, B).mean(-1)
