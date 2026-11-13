# %% [markdown]
# # 38 · Symbolic planning: Blocksworld in PDDL, solved with pyperplan

# %%
import os, time, logging
from pyperplan.planner import search_plan, SEARCHES, HEURISTICS

logging.disable(logging.CRITICAL)

DOMAIN = """(define (domain blocksworld)
  (:requirements :strips)
  (:predicates (on ?x ?y) (ontable ?x) (clear ?x) (holding ?x) (handempty))
  (:action pick-up
    :parameters (?x)
    :precondition (and (clear ?x) (ontable ?x) (handempty))
    :effect (and (holding ?x) (not (ontable ?x)) (not (clear ?x)) (not (handempty))))
  (:action put-down
    :parameters (?x)
    :precondition (holding ?x)
    :effect (and (ontable ?x) (clear ?x) (handempty) (not (holding ?x))))
  (:action stack
    :parameters (?x ?y)
    :precondition (and (holding ?x) (clear ?y))
    :effect (and (on ?x ?y) (clear ?x) (handempty) (not (holding ?x)) (not (clear ?y))))
  (:action unstack
    :parameters (?x ?y)
    :precondition (and (on ?x ?y) (clear ?x) (handempty))
    :effect (and (holding ?x) (clear ?y) (not (on ?x ?y)) (not (clear ?x)) (not (handempty)))))
"""

def problem(init, goal, objects="a b c d e"):
    return f"""(define (problem p) (:domain blocksworld)
  (:objects {objects})
  (:init {' '.join(init)})
  (:goal (and {' '.join(goal)})))"""

def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(text)
    return path

def solve(domain_file, problem_file, search="astar", heuristic="hff"):
    t0 = time.perf_counter()
    plan = search_plan(domain_file, problem_file, SEARCHES[search], HEURISTICS[heuristic] if heuristic else None)
    return plan, time.perf_counter() - t0

# %%
INIT = ["(on a b)", "(on b c)", "(ontable c)", "(on d e)", "(ontable e)", "(clear a)", "(clear d)", "(handempty)"]
GOAL = ["(on a b)", "(on b c)", "(on c d)", "(on d e)", "(ontable e)"]
dom = write("generated/blocksworld.pddl", DOMAIN)
prob = write("generated/problem_5.pddl", problem(INIT, GOAL))
plan, t = solve(dom, prob)
print(f"plan found in {t * 1000:.1f} ms, length {len(plan)}:")
for i, op in enumerate(plan):
    print(f"  {i + 1:2d}. {op.name}")

# %%
for search, h in [("bfs", None), ("astar", "hff"), ("gbf", "hff"), ("astar", "lmcut")]:
    p, t = solve(dom, prob, search, h)
    print(f"{search:6s} {str(h):6s} plan length {len(p):2d}   {t * 1000:7.1f} ms")

# %%
for n in [4, 5, 6, 7]:
    objs = " ".join("abcdefgh"[:n])
    init = [f"(ontable {o})" for o in objs.split()] + [f"(clear {o})" for o in objs.split()] + ["(handempty)"]
    goal = [f"(on {x} {y})" for x, y in zip(objs.split()[:-1], objs.split()[1:])]
    pf = write(f"generated/tower_{n}.pddl", problem(init, goal, objs))
    for search, h in [("bfs", None), ("gbf", "hff")]:
        p, t = solve(dom, pf, search, h)
        print(f"{n} blocks  {search:4s}: length {len(p):2d}  {t * 1000:8.1f} ms")
