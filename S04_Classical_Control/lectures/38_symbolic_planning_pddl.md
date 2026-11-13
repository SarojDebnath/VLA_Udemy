# 38 · Symbolic Actions: STRIPS, PDDL, and Action Schemas

**Section:** S04 · **Type:** Theory · **Duration:** ~25 min
**Companion notebook:** `38_pddl_symbolic_planning.ipynb`

## Learning objectives
- Model a manipulation domain with predicates and action schemas.
- Run a classical planner and interpret search vs heuristic trade-offs.
- Explain what symbolic planning gives VLAs today (task decomposition, SayCan-style grounding).

---

## 1. The symbolic abstraction
The world is a set of true facts: `(on a b)`, `(clear a)`, `(handempty)`. Geometry, physics and perception are abstracted away. An **action schema** has:
- parameters: `?x ?y`
- **preconditions**: facts that must hold
- **effects**: an add list and a delete list (`not`)

STRIPS (Fikes & Nilsson 1971) defined this form; PDDL (1998) standardized the syntax.

## 2. Blocksworld
```lisp
(:action stack
  :parameters (?x ?y)
  :precondition (and (holding ?x) (clear ?y))
  :effect (and (on ?x ?y) (clear ?x) (handempty) (not (holding ?x)) (not (clear ?y))))
```
Four schemas (`pick-up`, `put-down`, `stack`, `unstack`) fully describe stacking. The problem file gives the objects, initial facts and goal. From towers a/b/c and d/e to a single column a-b-c-d-e, the optimal plan has 10 actions.

## 3. Planning as search
The state space is exponential in the number of facts. Planners search it with heuristics derived automatically from the domain:
| Search | Optimal? | Notebook (5 blocks) |
|---|---|---|
| BFS (blind) | yes (unit cost) | 10 steps, 14 ms |
| A* + h_FF | no (h_FF inadmissible) | 10 steps |
| A* + LM-cut | yes | 10 steps |
| Greedy best-first + h_FF | no | 14 steps, fastest |

Scaling to building a tower of 7 blocks: BFS takes 0.94 s, greedy 21 ms. **Heuristics turn exponential search into practical search**, at the cost of optimality.

## 4. Where symbolic planning breaks
- Someone must write the domain, and they'll miss things (a block slightly rotated, a fragile object).
- **Grounding:** symbols don't say *where* or *how*. `stack c d` needs geometry (IK) and control, which is project 39's job.
- Symbols ignore geometry: the planner doesn't know if putting `a` down blocks the path to `b` (the problem PDDLStream solves).

## 5. Symbolic planning in the VLA era
- **SayCan (2022):** an LLM proposes skills ("pick up the sponge"), and value functions say which are feasible. It's PDDL-like decomposition with learned grounding.
- **LLM-as-planner / Code-as-Policies:** the language model writes the plan; VLAs or skills execute the steps.
- **Dual-system VLAs (S11):** the slow System 2 VLM plays the role of the symbolic planner, emitting subgoals for a fast System 1 policy.
- PDDL remains useful for verification and long-horizon tasks with hard constraints.

## Key takeaways
- PDDL = predicates + action schemas (preconditions, add/delete effects).
- Heuristic search makes planning fast; optimality costs time.
- Symbols must be grounded in geometry and control, which is the "three-layer cake" of project 39.

## References
- Fikes & Nilsson 1971, *STRIPS*; McDermott et al. 1998, *PDDL*
- Helmert 2006, *The Fast Downward Planning System*
- Ahn et al. 2022, *SayCan*; Garrett et al. 2020, *PDDLStream*
- Kandapagari, *Action Models for Robot Learning*, §4.1, Exercise 4.x.1
