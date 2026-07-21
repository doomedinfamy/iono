from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from models import sgp, sigma

TERMINALS = ["P_ini_A", "P_ini_B", "dP_A", "dP_B", "P_aft_A", "P_aft_B",
              "EV_A", "EV_B", "const"]
BINARY_OPS = ["add", "sub", "mul", "sgp"]

OP_ARITY = {op: 2 for op in BINARY_OPS}


# expression trees are nested tuples: ("add", left, right) | ("var", name)
# | ("const", value)

def random_tree(rng, depth=3, p_leaf=0.35):
    if depth <= 0 or rng.random() < p_leaf:
        t = rng.choice(TERMINALS)
        if t == "const":
            return ("const", float(np.round(rng.uniform(-2, 2), 3)))
        return ("var", t)
    op = rng.choice(BINARY_OPS)
    return (op, random_tree(rng, depth - 1, p_leaf), random_tree(rng, depth - 1, p_leaf))


def evaluate(tree, F):
    kind = tree[0]
    if kind == "var":
        return F[tree[1]]
    if kind == "const":
        return np.full_like(F["P_ini_A"], tree[1])
    a, b = evaluate(tree[1], F), evaluate(tree[2], F)
    with np.errstate(all="ignore"):
        if kind == "add":
            return a + b
        if kind == "sub":
            return a - b
        if kind == "mul":
            return a * b
        if kind == "sgp":
            return sgp(a, np.clip(b, -4, 4))
    raise ValueError(kind)


def complexity(tree):
    if tree[0] in ("var", "const"):
        return 2                      # paper-style: leaves cost 2
    return 1 + complexity(tree[1]) + complexity(tree[2])


def to_str(tree):
    kind = tree[0]
    if kind == "var":
        return tree[1]
    if kind == "const":
        return f"{tree[1]:.3g}"
    sym = {"add": "+", "sub": "-", "mul": "*"}
    if kind in sym:
        return f"({to_str(tree[1])} {sym[kind]} {to_str(tree[2])})"
    return f"sgp({to_str(tree[1])}, {to_str(tree[2])})"


def fitness(tree, F, p_emp):
    """MSE of sigma(k * expr) with the scalar k optimized (Brent)."""
    with np.errstate(all="ignore"):
        x = evaluate(tree, F)
    if not np.all(np.isfinite(x)) or np.std(x) < 1e-12:
        return 1e6, 0.0
    x = np.clip(x, -1e6, 1e6)

    def loss(k):
        return float(np.mean((sigma(k * x) - p_emp) ** 2))

    res = minimize_scalar(loss, bounds=(-80, 80), method="bounded",
                          options={"xatol": 1e-4})
    return float(res.fun), float(res.x)


def all_subtrees(tree, path=()):
    yield path, tree
    if tree[0] not in ("var", "const"):
        yield from all_subtrees(tree[1], path + (1,))
        yield from all_subtrees(tree[2], path + (2,))


def replace_at(tree, path, sub):
    if not path:
        return sub
    lst = list(tree)
    lst[path[0]] = replace_at(tree[path[0]], path[1:], sub)
    return tuple(lst)


def crossover(rng, t1, t2):
    p1 = list(all_subtrees(t1))
    p2 = list(all_subtrees(t2))
    path1, _ = p1[rng.integers(len(p1))]
    _, sub2 = p2[rng.integers(len(p2))]
    return replace_at(t1, path1, sub2)


def mutate(rng, tree):
    r = rng.random()
    nodes = list(all_subtrees(tree))
    path, sub = nodes[rng.integers(len(nodes))]
    if r < 0.4:                                   # subtree mutation
        return replace_at(tree, path, random_tree(rng, 2))
    if r < 0.7 and sub[0] in BINARY_OPS:          # node (operator) mutation
        return replace_at(tree, path, (rng.choice(BINARY_OPS), sub[1], sub[2]))
    # constant perturbation
    def perturb(t):
        if t[0] == "const":
            return ("const", float(t[1] + rng.normal(0, 0.3)))
        if t[0] == "var":
            return t
        return (t[0], perturb(t[1]), perturb(t[2]))
    return perturb(tree)


def pareto_front(items):
    """items: list of (complexity, mse, tree, scale); returns nondominated."""
    front = []
    for it in sorted(items, key=lambda t: (t[0], t[1])):
        if all(it[1] < f[1] for f in front):
            front.append(it)
    return front


def run_sr(F, p_emp, seed=0, pop_size=250, generations=35, max_complexity=20):
    rng = np.random.default_rng(seed)
    pop = [random_tree(rng) for _ in range(pop_size)]
    scored = {}

    def score(tree):
        key = to_str(tree)
        if key not in scored:
            c = complexity(tree)
            if c > max_complexity:
                scored[key] = (c, 1e6, tree, 0.0)
            else:
                mse, k = fitness(tree, F, p_emp)
                scored[key] = (c, mse, tree, k)
        return scored[key]

    evaluated = [score(t) for t in pop]
    for gen in range(generations):
        children = []
        for _ in range(pop_size):
            # binary tournament on (rank by mse, tie: complexity)
            cand = [evaluated[rng.integers(len(evaluated))] for _ in range(2)]
            p1 = min(cand, key=lambda t: (t[1], t[0]))[2]
            cand = [evaluated[rng.integers(len(evaluated))] for _ in range(2)]
            p2 = min(cand, key=lambda t: (t[1], t[0]))[2]
            child = crossover(rng, p1, p2)
            child = mutate(rng, child)
            children.append(child)
        evaluated.extend(score(t) for t in children)
        # environmental selection: keep Pareto front + best by mse
        front = pareto_front(evaluated)
        rest = sorted(evaluated, key=lambda t: (t[1], t[0]))[:pop_size - len(front)]
        evaluated = front + rest
    return pareto_front(evaluated)
