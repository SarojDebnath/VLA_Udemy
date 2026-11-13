# 42 · MDPs, Returns, and the Bellman Equations

**Section:** S05 Reinforcement Learning · **Type:** Math · **Duration:** ~30 min
**Companion notebooks:** `43_value_policy_iteration`, `44_q_learning_exploration`

## Learning objectives
- Formalize a control problem as a Markov decision process.
- Define returns, value functions and Q-functions, and derive the Bellman equations.
- Understand why the Bellman operator converges (contraction), and how a robot task maps onto an MDP.

---

## 1. The Markov decision process
$$\mathcal M = (\mathcal S, \mathcal A, P, r, \gamma, \rho_0)$$
- states $s\in\mathcal S$, actions $a\in\mathcal A$
- dynamics $P(s'\mid s,a)$ (MuJoCo is a deterministic $P$, plus noise and randomization)
- reward $r(s,a)$, discount $\gamma\in[0,1)$, initial distribution $\rho_0$
- **Markov property:** the future depends only on the current state, not the history.

A **policy** $\pi(a\mid s)$ maps states to action distributions. That's the same object as a VLA, $\pi(a\mid o,\ell)$, except the VLA sees observations rather than the full state.

## 2. Return and value
$$G_t = \sum_{k=0}^{\infty}\gamma^k r_{t+k}$$
$$V^\pi(s) = \mathbb E_\pi[G_t\mid s_t=s],\qquad Q^\pi(s,a) = \mathbb E_\pi[G_t\mid s_t=s,a_t=a]$$
$\gamma$ sets the effective horizon $\approx 1/(1-\gamma)$: 0.99 → ~100 steps.

## 3. Bellman equations
Expectation (for a fixed $\pi$):
$$V^\pi(s) = \sum_a \pi(a\mid s)\Big[r(s,a) + \gamma\sum_{s'}P(s'\mid s,a)V^\pi(s')\Big]$$
Optimality:
$$V^*(s) = \max_a\Big[r(s,a) + \gamma\sum_{s'}P(s'\mid s,a)V^*(s')\Big],\qquad
Q^*(s,a) = r + \gamma\,\mathbb E_{s'}\big[\max_{a'}Q^*(s',a')\big]$$
The optimal policy is greedy: $\pi^*(s) = \arg\max_a Q^*(s,a)$.

## 4. Why iterating works
The Bellman optimality operator $\mathcal T$ is a **γ-contraction** in the max norm:
$$\|\mathcal TV - \mathcal TU\|_\infty \le \gamma\|V-U\|_\infty$$
So value iteration $V_{k+1} = \mathcal TV_k$ converges geometrically to the unique fixed point $V^*$. Notebook 43: 7 sweeps on the deterministic 4×4 lake, 438 on the slippery one (rewards propagate slowly through stochastic transitions).

## 5. Dynamic programming vs learning
| | Needs the model $P$? | Method |
|---|---|---|
| Value iteration | yes | repeated Bellman optimality backups |
| Policy iteration | yes | evaluate π exactly, act greedily; few iterations (7 here) |
| Q-learning, SARSA | no (samples) | stochastic approximation of the Bellman equation |
| Policy gradient, PPO | no | optimize π directly from returns |

## 6. From MDP to robot (book §5.5)
Every robotics RL paper makes four design decisions:
1. **State/observation:** joint angles, object poses, images? Partial observability breaks the Markov property, so add history or memory.
2. **Action space:** torques, joint targets, end-effector deltas; and the control rate. These shape the exploration difficulty enormously.
3. **Reward:** dense (distance) vs sparse (success). Lecture 45 shows how dense rewards get hacked.
4. **Episode structure:** resets, time limits (truncation is *not* termination; bootstrap through truncations), safety constraints.

Our `PandaReachEnv`: 20-D state (joints, velocities, target offset, ee position), 7-D joint-delta actions at 20 Hz, reward = −distance − action penalty + success bonus, 50-step episodes.

## Key takeaways
- An MDP is (S, A, P, r, γ); a policy maps states to action distributions.
- The Bellman equations define $V^*$ and $Q^*$; their contraction makes iteration converge.
- Robot RL is mostly about choosing state, action, reward and episode structure well.

## References
- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.), Ch. 3–4
- Kandapagari, *Action Models for Robot Learning*, §5.1–5.2, §5.5
