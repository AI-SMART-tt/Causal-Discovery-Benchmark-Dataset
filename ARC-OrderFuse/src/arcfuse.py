from __future__ import annotations
import itertools
import math
from collections import Counter
DEFAULTS = {'k': 4.0, 'data_a0': 0.75, 'llm_a0': 0.8, 'llm_a0_bounds': (0.6, 0.9), 'abstain': 0.05, 'iters': 8, 'a_max': 0.98}

def skeleton_pairs(directed, undirected=()):
    return sorted({tuple(sorted(e)) for e in directed} | {tuple(sorted(e)) for e in undirected})

def has_path(directed, source, target):
    children = {}
    for a, b in directed:
        children.setdefault(a, []).append(b)
    stack, seen = ([source], set())
    while stack:
        x = stack.pop()
        if x == target:
            return True
        if x in seen:
            continue
        seen.add(x)
        stack.extend(children.get(x, []))
    return False

def v_structures(directed, adjacency):
    parents = {}
    for a, b in directed:
        parents.setdefault(b, set()).add(a)
    out = set()
    for z, ps in parents.items():
        for x, y in itertools.combinations(sorted(ps), 2):
            if frozenset((x, y)) not in adjacency:
                out.add((x, z, y))
    return out

def dag_to_cpdag(nodes, dag):
    adj = {frozenset(e) for e in dag}
    directed = set()
    for x, z, y in v_structures(dag, adj):
        directed.add((x, z))
        directed.add((y, z))
    und = {p for p in adj if not any(((a, b) in directed for a, b in itertools.permutations(tuple(p))))}
    directed, und = meek(nodes, directed, und, adj)
    return (directed, und)

def meek(nodes, directed, und, adj=None):
    directed, und = (set(directed), set(und))
    if adj is None:
        adj = {frozenset(e) for e in directed} | set(und)
    changed = True
    while changed:
        changed = False
        for p in sorted(und, key=lambda q: tuple(sorted(q))):
            a, b = tuple(sorted(p))
            for x, y in ((a, b), (b, a)):
                r1 = any(((w, x) in directed and frozenset((w, y)) not in adj for w in nodes if w not in (x, y)))
                r2 = any(((x, w) in directed and (w, y) in directed for w in nodes))
                ws = [w for w in nodes if frozenset((x, w)) in und and (w, y) in directed]
                r3 = any((frozenset((w1, w2)) not in adj for w1, w2 in itertools.combinations(ws, 2)))
                if (r1 or r2 or r3) and (not has_path(directed, y, x)):
                    und.discard(p)
                    directed.add((x, y))
                    changed = True
                    break
    return (directed, und)

def greedy_acyclic(items):
    directed, cycle_avoids = (set(), 0)
    for it in sorted(items, key=lambda r: (-r['margin'], r['a'], r['b'])):
        first, second = (it['a'], it['b']) if it['p'] >= 0.5 else (it['b'], it['a'])
        if not has_path(directed, second, first):
            directed.add((first, second))
        elif not has_path(directed, first, second):
            directed.add((second, first))
            cycle_avoids += 1
    return (directed, cycle_avoids)

def boot_vote(boot, pair):
    if not boot or not boot.get('reps'):
        return 0.0
    nab, nba, _ = boot['counts'].get('||'.join(pair), [0, 0, 0])
    return (nab - nba) / (boot['reps'] + 1.0)

def llm_vote(prior, pair, use_support=True):
    sup = prior.get('pair_support') if use_support else None
    if sup is not None and pair in sup:
        return 2.0 * sup[pair] - 1.0
    rank = prior['_rank'] if '_rank' in prior else {n: i for i, n in enumerate(prior['ordered_nodes'])}
    return 1.0 if rank[pair[0]] < rank[pair[1]] else -1.0

def llm_default_reliability(prior, bounds=DEFAULTS['llm_a0_bounds']):
    try:
        c = float(prior.get('confidence'))
    except (TypeError, ValueError):
        c = 0.75
    return max(bounds[0], min(bounds[1], c))

def logit(p):
    return math.log(p / (1.0 - p))

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))
SIGNED_ORDER = True

def contribution(source, x, a, calibrated=('hc',)):
    if x == 0:
        return 0.0
    if source in calibrated:
        return math.copysign(logit(min(DEFAULTS['a_max'], 0.5 + abs(x) / 2.0)), x)
    return x * logit(a[source])

def estimate_reliabilities(votes, a0, k=DEFAULTS['k'], iters=DEFAULTS['iters'], a_max=DEFAULTS['a_max'], update=True, calibrated=('hc',)):
    a = dict(a0)
    n_eff = {s: 0.0 for s in a0}
    if not update:
        return (a, n_eff)
    for _ in range(iters):
        new = dict(a)
        for s in a0:
            if s in calibrated:
                continue
            num = den = 0.0
            for v in votes.values():
                x = v.get(s, 0.0)
                if x == 0:
                    continue
                L = sum((contribution(o, xo, a, calibrated) for o, xo in v.items() if o != s))
                q = sigmoid(L if x > 0 else -L)
                c = max(q, 1.0 - q)
                w = abs(x) * (2.0 * c - 1.0)
                if w <= 0:
                    continue
                num += w * ((1.0 if q > 0.5 else 0.0) - (1.0 - c))
                den += w * (2.0 * c - 1.0)
            lower = 1.0 - a_max if s == 'llm' and SIGNED_ORDER else 0.5
            new[s] = max(lower, min(a_max, (num + k * a0[s]) / (den + k)))
            n_eff[s] = den
        a = new
    return (a, n_eff)

def arc_fuse(nodes, pairs, votes, a0, k=DEFAULTS['k'], abstain=DEFAULTS['abstain'], update=True, calibrated=('hc',)):
    a, n_eff = estimate_reliabilities(votes, a0, k=k, update=update, calibrated=calibrated)
    items, und, stats = ([], set(), Counter())
    for pair in pairs:
        v = votes[pair]
        data_L = sum((contribution(s, x, a, calibrated) for s, x in v.items() if s != 'llm'))
        llm_L = contribution('llm', v.get('llm', 0.0), a, calibrated)
        L = data_L + llm_L
        p = sigmoid(L)
        lv = v.get('llm', 0.0)
        if data_L * lv < 0:
            stats['conflicts'] += 1
            stats['llm_wins' if (L > 0) == (lv > 0) else 'data_wins'] += 1
        if abs(p - 0.5) < abstain:
            und.add(frozenset(pair))
            stats['abstain'] += 1
            continue
        items.append({'a': pair[0], 'b': pair[1], 'p': p, 'margin': abs(p - 0.5)})
    directed, cyc = greedy_acyclic(items)
    audit = {'rel_' + s: round(val, 4) for s, val in a.items() if s not in calibrated}
    audit.update({'neff_' + s: round(val, 3) for s, val in n_eff.items() if s not in calibrated})
    audit.update({'conflicts': stats['conflicts'], 'llm_wins': stats['llm_wins'], 'data_wins': stats['data_wins'], 'abstain': stats['abstain'], 'cycle_avoids': cyc})
    return (directed, und, audit)

def build_votes(pairs, prior, boot_pc=None, boot_hc=None, use_support=True, fixed_base=None, eta=0.62):
    prior = dict(prior, _rank={n: i for i, n in enumerate(prior['ordered_nodes'])})
    votes = {}
    for pair in pairs:
        v = {}
        if fixed_base is not None:
            s = 2.0 * (eta - 0.5)
            v['base'] = s if pair in fixed_base else -s if (pair[1], pair[0]) in fixed_base else 0.0
        if boot_pc is not None:
            v['pc'] = boot_vote(boot_pc, pair)
        if boot_hc is not None:
            v['hc'] = boot_vote(boot_hc, pair)
        v['llm'] = llm_vote(prior, pair, use_support)
        votes[pair] = v
    return votes

def arc_orient(nodes, base_directed, base_undirected, prior, boot_pc, boot_hc, k=DEFAULTS['k'], data_a0=DEFAULTS['data_a0'], abstain=DEFAULTS['abstain'], variant='full', llm_a0=None):
    pairs = skeleton_pairs(base_directed, base_undirected)
    use_support = variant != 'no_support'
    if variant == 'fixed_eta':
        votes = build_votes(pairs, prior, fixed_base={tuple(e) for e in base_directed}, use_support=use_support)
    elif variant == 'pc_only':
        votes = build_votes(pairs, prior, boot_pc=boot_pc, use_support=use_support)
    elif variant == 'hc_only':
        votes = build_votes(pairs, prior, boot_hc=boot_hc, use_support=use_support)
    else:
        votes = build_votes(pairs, prior, boot_pc=boot_pc, boot_hc=boot_hc, use_support=use_support)
    if llm_a0 == 'self_reported':
        la0 = llm_default_reliability(prior)
    else:
        la0 = DEFAULTS['llm_a0'] if llm_a0 is None else llm_a0
    a0 = {'llm': la0}
    for s in ('pc', 'hc', 'base'):
        if any((s in v for v in votes.values())):
            a0[s] = data_a0
    calibrated = () if variant == 'estimate_all' else ('hc',)
    global SIGNED_ORDER
    if variant == 'data_only':
        votes = {p: {s: x for s, x in v.items() if s != 'llm'} for p, v in votes.items()}
        a0 = {s: v for s, v in a0.items() if s != 'llm'}
        return arc_fuse(nodes, pairs, votes, a0, k=k, abstain=abstain, update=False, calibrated=calibrated)
    if variant == 'unsigned':
        SIGNED_ORDER = False
        try:
            return arc_fuse(nodes, pairs, votes, a0, k=k, abstain=abstain, update=True, calibrated=calibrated)
        finally:
            SIGNED_ORDER = True
    return arc_fuse(nodes, pairs, votes, a0, k=k, abstain=abstain, update=variant != 'no_update', calibrated=calibrated)

def orient_by_order(base_directed, base_undirected, order):
    rank = {n: i for i, n in enumerate(order)}
    return {(a, b) if rank[a] < rank[b] else (b, a) for a, b in skeleton_pairs(base_directed, base_undirected)}

def order_on_cpdag(nodes, base_directed, base_undirected, order):
    rank = {n: i for i, n in enumerate(order)}
    out = set((tuple(e) for e in base_directed))
    for p in sorted(base_undirected, key=lambda q: tuple(sorted(q))):
        a, b = tuple(sorted(p))
        f, s = (a, b) if rank[a] < rank[b] else (b, a)
        if not has_path(out, s, f):
            out.add((f, s))
        elif not has_path(out, f, s):
            out.add((s, f))
    return out

def mec_consistent_order(nodes, base_directed, base_undirected, prior):
    directed = set((tuple(e) for e in base_directed))
    und = {frozenset(p) for p in base_undirected}
    adj = {frozenset(e) for e in directed} | und
    pr = dict(prior, _rank={n: i for i, n in enumerate(prior['ordered_nodes'])})
    queue = sorted((tuple(sorted(p)) for p in und), key=lambda k: (-abs(llm_vote(pr, k)), k))
    for pair in queue:
        p = frozenset(pair)
        if p not in und:
            continue
        x = llm_vote(pr, pair)
        if x == 0:
            continue
        src, dst = pair if x > 0 else (pair[1], pair[0])
        if has_path(directed, dst, src):
            continue
        new_collider = any(((w, dst) in directed and frozenset((w, src)) not in adj for w in nodes if w not in (src, dst)))
        if new_collider:
            continue
        und.discard(p)
        directed.add((src, dst))
        directed, und = meek(nodes, directed, und, adj)
    return (directed, und)

def evaluate(nodes, true_edges, directed, undirected=()):
    true_edges = set(true_edges)
    directed = set((tuple(e) for e in directed))
    undirected = {frozenset(p) for p in undirected}
    true_skel = {frozenset(e) for e in true_edges}
    pred_skel = {frozenset(e) for e in directed} | undirected
    tp = len(directed & true_edges)
    prec = tp / len(directed) if directed else 0.0
    rec = tp / len(true_edges) if true_edges else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    stp = len(true_skel & pred_skel)
    sp = stp / len(pred_skel) if pred_skel else 0.0
    sr = stp / len(true_skel) if true_skel else 0.0
    sf1 = 2 * sp * sr / (sp + sr) if sp + sr else 0.0
    reversed_edges = sum((1 for a, b in directed if (b, a) in true_edges))
    undirected_true = sum((1 for p in undirected if p in true_skel))
    p_ori = tp / stp if stp else 0.0
    of1 = 2 * p_ori * rec / (p_ori + rec) if p_ori + rec else 0.0
    committed = tp + reversed_edges
    ori_acc = tp / committed if committed else float('nan')
    shd = len(true_skel - pred_skel) + len(pred_skel - true_skel) + reversed_edges + 0.5 * undirected_true
    dtop, dtop_rate = topological_divergence(nodes, true_edges, directed, undirected)
    return {'dtop': dtop, 'dtop_rate': dtop_rate, 'f1': f1, 'precision': prec, 'recall': rec, 'shd': shd, 'skeleton_f1': sf1, 'orient_f1': of1, 'orient_acc': ori_acc, 'reversed': reversed_edges, 'n_directed': len(directed), 'n_undirected': len(undirected), 'missing_adj': len(true_skel - pred_skel), 'extra_adj': len(pred_skel - true_skel)}

def ancestor_pairs(nodes, edges):
    children = {n: set() for n in nodes}
    for a, b in edges:
        children.setdefault(a, set()).add(b)
    out = set()
    for s in nodes:
        stack, seen = (list(children.get(s, ())), set())
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(children.get(x, ()))
        out.update(((s, t) for t in seen))
    return out

def topological_divergence(nodes, true_edges, directed, undirected=()):
    from collections import deque
    rank0 = {n: i for i, n in enumerate(nodes)}
    g = set(directed)
    for p in sorted(undirected, key=lambda q: tuple(sorted(q))):
        a, b = tuple(p)
        first, second = (a, b) if rank0.get(a, 0) <= rank0.get(b, 0) else (b, a)
        if not has_path(g, second, first):
            g.add((first, second))
        elif not has_path(g, first, second):
            g.add((second, first))
    indeg = {n: 0 for n in nodes}
    ch = {n: [] for n in nodes}
    for a, b in sorted(g, key=lambda e: (rank0.get(e[0], len(nodes)), rank0.get(e[1], len(nodes)), e[0], e[1])):
        ch[a].append(b)
        indeg[b] += 1
    q = deque((n for n in nodes if indeg[n] == 0))
    order = []
    while q:
        n = q.popleft()
        order.append(n)
        for c in sorted(ch[n], key=lambda x: (rank0.get(x, len(nodes)), x)):
            indeg[c] -= 1
            if indeg[c] == 0:
                q.append(c)
    if len(order) < len(nodes):
        return (float('nan'), float('nan'))
    rank = {n: i for i, n in enumerate(order)}
    pairs = ancestor_pairs(nodes, true_edges)
    v = sum((1 for s, t in pairs if rank[s] >= rank[t]))
    return (v, v / len(pairs) if pairs else float('nan'))

def mec_diagnostics(nodes, base_directed, base_undirected, out_directed, out_undirected=()):
    base_directed = set((tuple(e) for e in base_directed))
    out_directed = set((tuple(e) for e in out_directed))
    adj = {frozenset(e) for e in base_directed} | {frozenset(p) for p in base_undirected}
    base_v = v_structures(base_directed, adj)
    out_v = v_structures(out_directed, adj)
    reversed_compelled = sum((1 for a, b in base_directed if (b, a) in out_directed))
    kept_v = len(base_v & out_v)
    return {'base_directed': len(base_directed), 'base_reversed': reversed_compelled, 'base_v': len(base_v), 'base_v_kept': kept_v, 'new_v': len(out_v - base_v), 'same_v_structures': int(base_v == out_v), 'in_base_class': int(base_v == out_v and reversed_compelled == 0)}
