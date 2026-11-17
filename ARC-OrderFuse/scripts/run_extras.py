import os as _os, sys as _sys
from pathlib import Path as _Path
_ROOT = next((p for p in _Path(__file__).resolve().parents if (p / 'src').is_dir()))
_os.chdir(_ROOT)
_sys.path.insert(0, str(_ROOT / 'src'))
for _d in ('tables', 'figures', 'latex', 'tmp'):
    (_ROOT / 'output' / _d).mkdir(parents=True, exist_ok=True)
import argparse, csv, json, math, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--code', default='.')
ap.add_argument('--grid', default='src')
ap.add_argument('--priors', default='priors/blind')
ap.add_argument('--out', default='output/extras')
ap.add_argument('--job', required=True, choices=['lingam', 'softprior'])
ap.add_argument('--sizes', default='')
ap.add_argument('--seeds', default='')
o = ap.parse_args()
import baseline as B, run_grid as G, arcfuse as A, prior as P
G.CODE = Path(o.code)
index = [r for r in csv.DictReader(open(Path(o.code) / 'data/dataset_index.csv', encoding='utf-8-sig'))]
index.sort(key=lambda r: int(r['nodes']))
out = Path(o.out)
out.mkdir(parents=True, exist_ok=True)
path = out / f'{o.job}.csv'
done = set()
if path.exists():
    for r in csv.DictReader(open(path)):
        done.add((r['variant'], r['dataset'], r['n'], r['seed'], r['method'], r.get('prior', '')))
fh = open(path, 'a', newline='')
writer = None

def emit(row):
    global writer
    if writer is None:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if path.stat().st_size == 0:
            writer.writeheader()
    writer.writerow({k: round(v, 6) if isinstance(v, float) else v for k, v in row.items()})
    fh.flush()
sel_n = {int(x) for x in o.sizes.split(',')} if o.sizes else None
sel_s = {int(x) for x in o.seeds.split(',')} if o.seeds else None

class PriorScorer:

    def __init__(self, base, prior, n, beta=4.0):
        self.base, self.beta, self.unit = (base, beta, math.log(n) / 2.0)
        self.rank = {x: i for i, x in enumerate(prior['ordered_nodes'])}
        self.sup = prior.get('pair_support')

    def violation(self, p, x):
        key = (p, x) if p < x else (x, p)
        if self.sup and key in self.sup:
            s = self.sup[key] if key == (p, x) else 1.0 - self.sup[key]
        else:
            s = 1.0 if self.rank[p] < self.rank[x] else 0.0
        return max(0.0, 1.0 - 2.0 * s)

    def score(self, node, parents):
        return self.base.score(node, parents) - self.beta * self.unit * sum((self.violation(p, node) for p in parents))
for variant in ['main', 'lin_laplace', 'tanh_laplace'] if o.job == 'lingam' else ['main']:
    for ds in index:
        bn = ds['collection'] == 'bnlearn'
        if o.job == 'lingam' and bn:
            continue
        true_all = B.read_edges(Path(o.code) / 'data' / ds['path'] / 'edges.csv')
        for n in G.BN_SIZES if bn else G.LG_SIZES:
            if sel_n and n not in sel_n:
                continue
            for seed in G.SEEDS:
                if sel_s and seed not in sel_s:
                    continue
                meta = {'variant': variant, 'collection': ds['collection'], 'dataset': ds['dataset'], 'n': n, 'seed': seed}
                headers = rows = None

                def data():
                    global headers, rows, dtype
                    if headers is None:
                        headers, rows, dtype = G.load_data(ds, n, seed, variant)
                    return (headers, rows, dtype)
                if o.job == 'lingam':
                    for m in ['direct_lingam', 'ica_lingam']:
                        if (variant, ds['dataset'], str(n), str(seed), m, '') in done:
                            continue
                        h, r, dt = data()
                        t = time.time()
                        try:
                            g = B.run_lingam(h, r, dt, m, seed, 0.05)['directed_edges']
                        except Exception as exc:
                            print('ERR', meta, m, repr(exc)[:120], flush=True)
                            continue
                        true = {(a, b) for a, b in true_all if a in h and b in h}
                        emit(dict(meta, method=m, prior='', **A.evaluate(h, true, g), runtime=time.time() - t))
                else:
                    h, r, dt = data()
                    true = {(a, b) for a, b in true_all if a in h and b in h}
                    oracle = B.oracle_order_from_edges(h, true)
                    base = B.LocalBICScorer(h, r, dt)
                    recs = P.load_orders(Path(o.priors), ds['collection'], ds['dataset'], 'gpt-5.4-mini', 'blind_full')
                    priors = {'gpt_blind_consensus': P.consensus_prior(h, recs)}
                    import random
                    for rho in [0.5, 1.0, 'rev']:
                        rng = random.Random(B.stable_int('corrupt', ds['dataset'], n, seed, rho, 0))
                        od = list(reversed(oracle)) if rho == 'rev' else sorted(h, key=lambda x: (1 - rho) * oracle.index(x) / len(h) + rho * rng.random())
                        priors[f'corrupt_{rho}'] = {'ordered_nodes': od}
                    for pn, pr in priors.items():
                        for beta in [2.0, 4.0, 8.0] if pn == 'gpt_blind_consensus' else [4.0]:
                            m = f'hc_softprior_b{beta:g}'
                            if (variant, ds['dataset'], str(n), str(seed), m, pn) in done:
                                continue
                            t = time.time()
                            g = B.run_hc_bic(h, r, dt, 3, 100, scorer=PriorScorer(base, pr, len(r), beta))['directed_edges']
                            rank = {x: i for i, x in enumerate(pr['ordered_nodes'])}
                            acc = sum((rank[a] < rank[b] for a, b in true)) / max(len(true), 1)
                            emit(dict(meta, method=m, prior=pn, prior_edge_acc=acc, **A.evaluate(h, true, g), runtime=time.time() - t))
                print(o.job, variant, ds['dataset'], n, seed, flush=True)
