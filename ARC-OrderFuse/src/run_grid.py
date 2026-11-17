from __future__ import annotations
import argparse
import csv
import itertools
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent
CODE = HERE.parent
PRIOR_DIR = HERE.parent / 'priors' / 'blind'
OUT = HERE.parent / 'runs'
LLM_MODEL = 'gpt-5.4-mini'
LLM_MODE = 'blind_full'
BN_SIZES = [250, 500, 1000, 5000, 10000]
LG_SIZES = [250, 500, 1000, 5000]
SEEDS = [0, 1, 2, 3, 4]
VARIANTS = ['main', 'lin_laplace', 'tanh_laplace']

def sem_rows(nodes, edges, n, seed, variant):
    import baseline as B
    if variant == 'main':
        return B.synthetic_sem_rows(nodes, edges, n, seed)
    rng = random.Random(seed)
    order = B.topological_sort(nodes, edges)
    parents = defaultdict(list)
    rank = {node: i for i, node in enumerate(nodes)}
    edge_order = sorted(edges, key=lambda e: (rank.get(e[0], len(nodes)), rank.get(e[1], len(nodes)), e[0], e[1]))
    for src, dst in edge_order:
        parents[dst].append(src)
    weights = {(s, d): rng.uniform(0.35, 1.15) * (-1 if rng.random() < 0.35 else 1) for s, d in edge_order}

    def laplace(sd):
        return sd / math.sqrt(2.0) * (rng.expovariate(1.0) - rng.expovariate(1.0))
    rows = []
    for _ in range(n):
        row = {}
        for node in order:
            value = laplace(1.0)
            if parents[node]:
                value += sum((weights[p, node] * row[p] for p in parents[node]))
                if variant == 'tanh_laplace':
                    value = math.tanh(value / (1 + 0.2 * len(parents[node]))) + laplace(0.35)
            row[node] = value
        rows.append({node: f'{row[node]:.8f}' for node in nodes})
    return (nodes, rows)

def load_data(ds, n, seed, variant):
    import baseline as B
    ds_dir = CODE / 'data' / ds['path']
    if ds['collection'] == 'bnlearn':
        headers, rows = B.sample_bnlearn_rows(B.load_bnlearn_parsed(ds_dir), n, seed)
    else:
        nodes = [r['node'] for r in B.read_csv_dicts(ds_dir / 'nodes.csv')]
        headers, rows = sem_rows(nodes, B.read_edges(ds_dir / 'edges.csv'), n, seed, variant)
    return (headers, rows, B.infer_rows_data_type(ds['collection'], rows))

def dag_to_cpdag(nodes, dag):
    adj = {frozenset(e) for e in dag}
    pa = {n: set() for n in nodes}
    for a, b in dag:
        pa[b].add(a)
    directed = set()
    for z in nodes:
        for x, y in itertools.combinations(sorted(pa[z]), 2):
            if frozenset((x, y)) not in adj:
                directed.add((x, z))
                directed.add((y, z))
    und = {p for p in adj if not any(((a, b) in directed for a, b in itertools.permutations(tuple(p))))}
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
                if r1 or r2 or r3:
                    und.discard(p)
                    directed.add((x, y))
                    changed = True
                    break
    return (directed, und)

def run_base(base, headers, rows, dtype, args, scorer=None, order=None):
    import baseline as B
    if base == 'pc_cl':
        r = B.run_causallearn_pc(headers, rows, dtype, args['alpha'], args['max_cond'])
    elif base == 'ges_cl':
        r = B.run_causallearn_ges(headers, rows, dtype, args['max_parents'])
    else:
        r = B.run_hc_bic(headers, rows, dtype, args['max_parents'], args['max_iter'], oracle_order=order, scorer=scorer)
    return (r['directed_edges'], r['undirected_edges'])

def pack(directed, undirected=()):
    return {'directed': sorted(([a, b] for a, b in directed)), 'undirected': sorted((sorted(p) for p in undirected))}

def run_task(task):
    import baseline as B
    import prior as P
    ds, n, seed, variant, reps, args = (task['ds'], task['n'], task['seed'], task['variant'], task['reps'], task['args'])
    out = OUT / variant / f"{ds['dataset']}__n{n}__seed{seed}.json"
    t0 = time.time()
    rec = {'collection': ds['collection'], 'dataset': ds['dataset'], 'n': n, 'seed': seed, 'variant': variant, 'reps_requested': reps, 'base': {}, 'boot': {}, 'hc_order': {}, 'errors': {}, 'time': {}}
    try:
        headers, rows, dtype = load_data(ds, n, seed, variant)
    except Exception as exc:
        rec['errors']['data'] = repr(exc)[:300]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec), encoding='utf-8')
        return (ds['dataset'], n, seed, variant, time.time() - t0, rec['errors'])
    rec['nodes'], rec['data_type'] = (headers, dtype)
    true_edges = B.read_edges(CODE / 'data' / ds['path'] / 'edges.csv')
    scorer = B.LocalBICScorer(headers, rows, dtype)
    for base in ['pc_cl', 'ges_cl', 'hc_bic_lite']:
        t = time.time()
        try:
            d, u = run_base(base, headers, rows, dtype, args, scorer=scorer if base == 'hc_bic_lite' else None)
            rec['base'][base] = pack(d, u)
        except Exception as exc:
            rec['errors'][base] = repr(exc)[:300]
        rec['time'][base] = round(time.time() - t, 3)
    for base in ['pc_cl', 'hc_bic_lite']:
        t = time.time()
        counts, done = ({}, 0)
        for rep in range(reps):
            _, brows = B.bootstrap_rows(headers, rows, len(rows), B.stable_int('arc_boot', variant, ds['dataset'], n, seed, rep))
            try:
                d, u = run_base(base, headers, brows, dtype, args)
            except Exception as exc:
                rec['errors'][f'boot_{base}'] = repr(exc)[:300]
                continue
            if base == 'hc_bic_lite':
                d, u = dag_to_cpdag(headers, d)
            done += 1
            for a, b in d:
                k = tuple(sorted((a, b)))
                counts.setdefault('||'.join(k), [0, 0, 0])[0 if (a, b) == k else 1] += 1
            for p in u:
                counts.setdefault('||'.join(sorted(p)), [0, 0, 0])[2] += 1
        rec['boot'][base] = {'reps': done, 'counts': counts}
        rec['time'][f'boot_{base}'] = round(time.time() - t, 3)
    oracle = B.oracle_order_from_edges(headers, {(a, b) for a, b in true_edges if a in headers and b in headers})
    orders = {'oracle': oracle, 'random': B.make_random_order(headers, ds['collection'], ds['dataset'], n, seed, 777), 'wrong': B.make_wrong_order(headers, oracle)}
    try:
        recs = P.load_orders(PRIOR_DIR, ds['collection'], ds['dataset'], LLM_MODEL, LLM_MODE)
        if recs:
            orders['llm'] = P.consensus_prior(headers, recs)['ordered_nodes']
    except Exception as exc:
        rec['errors']['llm_prior'] = repr(exc)[:300]
    rec['orders'] = {k: v for k, v in orders.items() if k in ('random', 'llm')}
    t = time.time()
    for name, order in orders.items():
        if not order:
            continue
        try:
            d, u = run_base('hc_bic_lite', headers, rows, dtype, args, scorer=scorer, order=order)
            rec['hc_order'][name] = pack(d, u)
        except Exception as exc:
            rec['errors'][f'hc_order_{name}'] = repr(exc)[:300]
    rec['time']['hc_order'] = round(time.time() - t, 3)
    rec['time']['total'] = round(time.time() - t0, 3)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(rec), encoding='utf-8')
    os.replace(tmp, out)
    return (ds['dataset'], n, seed, variant, time.time() - t0, rec['errors'])

def build_tasks(opts):
    with (CODE / 'data' / 'dataset_index.csv').open(newline='', encoding='utf-8-sig') as f:
        index = [r for r in csv.DictReader(f) if r['collection'] in {'bnlearn', 'llm_causal_order_graphs'}]
    if opts.datasets:
        wanted = {x.strip() for x in opts.datasets.split(',')}
        index = [r for r in index if r['dataset'] in wanted]
    alg_args = {'alpha': 0.05, 'max_cond': 2, 'max_parents': 3, 'max_iter': 100}
    tasks = []
    for vi, variant in enumerate(opts.variants.split(',')):
        for ds in index:
            bn = ds['collection'] == 'bnlearn'
            if variant != 'main' and bn:
                continue
            sizes = [250] if opts.quick else BN_SIZES if bn else LG_SIZES
            seeds = [0] if opts.quick else SEEDS
            if opts.sizes:
                sizes = [s for s in sizes if s in {int(x) for x in opts.sizes.split(',')}]
            if opts.seeds:
                seeds = [s for s in seeds if s in {int(x) for x in opts.seeds.split(',')}]
            for n in sizes:
                for seed in seeds:
                    out = OUT / variant / f"{ds['dataset']}__n{n}__seed{seed}.json"
                    if out.exists() and (not opts.redo):
                        if not opts.retry_errors:
                            continue
                        try:
                            if not json.loads(out.read_text(encoding='utf-8')).get('errors'):
                                continue
                        except Exception:
                            pass
                    cost = int(ds['nodes']) ** 2 * n * (6 if bn else 1)
                    tasks.append({'ds': ds, 'n': n, 'seed': seed, 'variant': variant, 'reps': opts.reps, 'args': alg_args, 'key': (vi, cost)})
    tasks.sort(key=lambda t: t['key'])
    return tasks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--variants', default=','.join(VARIANTS))
    ap.add_argument('--datasets', default='')
    ap.add_argument('--sizes', default='', help='optional subset of sample sizes, e.g. 250,1000')
    ap.add_argument('--seeds', default='', help='optional subset of seeds, e.g. 0,1')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--redo', action='store_true')
    ap.add_argument('--retry-errors', action='store_true')
    opts = ap.parse_args()
    if opts.quick:
        opts.datasets = opts.datasets or 'asia,cancer_paper_repo'
        opts.reps = min(opts.reps, 3)
    import multiprocessing as mp
    tasks = build_tasks(opts)
    OUT.mkdir(parents=True, exist_ok=True)
    versions = {'python': sys.version.split()[0], 'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED', '')}
    for mod in ['numpy', 'scipy', 'causallearn']:
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, '__version__', 'unknown')
        except Exception as exc:
            versions[mod] = f'missing: {exc}'
    try:
        from importlib.metadata import version
        versions['causal-learn'] = version('causal-learn')
    except Exception:
        pass
    (OUT / 'run_config.json').write_text(json.dumps({'versions': versions, 'reps': opts.reps, 'llm_model': LLM_MODEL, 'llm_mode': LLM_MODE, 'started': time.strftime('%Y-%m-%d %H:%M:%S')}, indent=1), encoding='utf-8')
    print(f'versions: {versions}')
    if 'missing' in str(versions.get('causallearn', '')):
        print('causal-learn is not installed in this Python. Run:  pip install causal-learn')
        return
    if not PRIOR_DIR.exists():
        print(f'WARNING: {PRIOR_DIR} not found - order-constrained HC for the LLM order will be skipped.')
    print(f'{len(tasks)} tasks to run with {opts.workers} workers (finished tasks are skipped). Output: {OUT}', flush=True)
    t0, done, failed = (time.time(), 0, 0)
    with mp.Pool(opts.workers) as pool:
        for name, n, seed, variant, sec, errors in pool.imap_unordered(run_task, tasks, chunksize=1):
            done += 1
            failed += bool(errors)
            el = time.time() - t0
            flag = f'  ERRORS: {errors}' if errors else ''
            print(f'[{done}/{len(tasks)}] {variant:12s} {name:28s} n={n:<6d} seed={seed} {sec:7.1f}s  elapsed {el / 60:6.1f} min{flag}', flush=True)
    print(f'\nFinished {done} tasks in {(time.time() - t0) / 60:.1f} min; {failed} with errors.' + ('  Rerun with --retry-errors to repeat those.' if failed else ''))
if __name__ == '__main__':
    if os.environ.get('PYTHONHASHSEED') != '0':
        env = dict(os.environ, PYTHONHASHSEED='0')
        sys.exit(subprocess.call([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:], env=env))
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    main()
