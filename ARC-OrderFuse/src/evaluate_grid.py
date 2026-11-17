from __future__ import annotations
import argparse
import csv
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
import arcfuse as A
import prior as P
BASES = ['pc_cl', 'ges_cl', 'hc_bic_lite']
PRIMARY = ('gpt-5.4-mini', 'blind_full')
RHOS = [0.0, 0.15, 0.3, 0.5, 0.75, 1.0, 'rev']
CFG = {}

def init(code_dir, prior_dir):
    pass
    import baseline as B
    CFG['B'] = B
    CFG['args'] = B.build_arg_parser().parse_args([])
    CFG['code'] = Path(code_dir)
    CFG['priors'] = Path(prior_dir)
    CFG['true'] = {}
    CFG['legacy'] = Path(code_dir) / 'priors' / 'submitted'

def true_edges(collection, dataset):
    key = (collection, dataset)
    if key not in CFG['true']:
        CFG['true'][key] = CFG['B'].read_edges(CFG['code'] / 'data' / collection / dataset / 'edges.csv')
    return CFG['true'][key]

def legacy_view(prior):
    return dict(prior, source='api', prompt_mode='full')

def legacy_rules(nodes, d0, u0, prior, base):
    B, args = (CFG['B'], CFG['args'])
    pred = {'directed_edges': set(d0), 'undirected_edges': set(u0), 'completed_edges': set(d0), 'extra': {}}
    lp = legacy_view(prior)
    order = lp['ordered_nodes']
    out = {}
    for name, fn in [('soft', lambda: B.apply_soft_llm_prior_to_prediction(nodes, pred, order, base, lp, args, False)), ('conflict', lambda: B.apply_soft_llm_prior_to_prediction(nodes, pred, order, base, lp, args, True)), ('arc_orig', lambda: B.apply_adaptive_llm_prior_to_prediction(nodes, pred, order, base, lp, args))]:
        r = fn()
        ex = r['extra']
        out[name] = (r['directed_edges'], set(), {'conflicts': ex.get('llm_data_conflicts', ''), 'llm_wins': ex.get('llm_conflict_llm_wins', ''), 'data_wins': ex.get('llm_conflict_data_wins', ''), 'cycle_avoids': ex.get('cycle_avoids', '')})
    return out

def base_cpdag(nodes, base, d0, u0):
    if base == 'hc_bic_lite':
        return A.dag_to_cpdag(nodes, d0)
    return (set(d0), set(u0))

def corrupted_order(nodes, oracle, rho, rng):
    if rho == 'rev':
        return list(reversed(oracle))
    pos = {x: i / len(nodes) for i, x in enumerate(oracle)}
    return sorted(nodes, key=lambda x: (1 - rho) * pos[x] + rho * rng.random())

def order_edge_accuracy(order, edges):
    rank = {n: i for i, n in enumerate(order)}
    e = [(a, b) for a, b in edges if a in rank and b in rank]
    return sum((rank[a] < rank[b] for a, b in e)) / len(e) if e else float('nan')

def process(path):
    B = CFG['B']
    rec = json.loads(Path(path).read_text(encoding='utf-8'))
    if 'nodes' not in rec:
        return ([], [], [])
    nodes, coll, ds = (rec['nodes'], rec['collection'], rec['dataset'])
    true = {(a, b) for a, b in true_edges(coll, ds) if a in nodes and b in nodes}
    oracle = B.oracle_order_from_edges(nodes, true)
    meta = {'variant': rec['variant'], 'collection': coll, 'dataset': ds, 'n': rec['n'], 'seed': rec['seed']}
    boot_pc, boot_hc = (rec['boot'].get('pc_cl'), rec['boot'].get('hc_bic_lite'))

    def load(model, mode):
        recs = P.load_orders(CFG['priors'], coll, ds, model, mode)
        return P.consensus_prior(nodes, recs) if recs else None
    primary = load(*PRIMARY)
    main_rows, prior_rows, cor_rows = ([], [], [])

    def row(base, method, prior_name, directed, undirected=(), audit=None, cp=None, order=None, extra=None):
        r = dict(meta, base=base, method=method, prior=prior_name)
        r.update(A.evaluate(nodes, true, directed, undirected))
        if order is not None:
            r['prior_edge_acc'] = order_edge_accuracy(order, true)
        if cp is not None:
            r.update(A.mec_diagnostics(nodes, cp[0], cp[1], directed, undirected))
        r.update(audit or {})
        r.update(extra or {})
        return r
    for base in BASES:
        if base not in rec['base']:
            continue
        d0 = {tuple(e) for e in rec['base'][base]['directed']}
        u0 = {frozenset(e) for e in rec['base'][base]['undirected']}
        cp = base_cpdag(nodes, base, d0, u0)
        hc = base == 'hc_bic_lite'
        main_rows.append(row(base, 'data', 'none', d0, u0, cp=cp))
        controls = {'oracle': oracle, 'random': rec.get('orders', {}).get('random'), 'wrong': list(reversed(oracle)) if oracle else None}
        for name, order in controls.items():
            if not order:
                continue
            if hc and name in rec['hc_order']:
                g = {tuple(e) for e in rec['hc_order'][name]['directed']}
            else:
                g = A.orient_by_order(d0, u0, order)
            main_rows.append(row(base, name, name, g, order=order))
            if hc:
                main_rows.append(row(base, name + '_skeleton', name, A.orient_by_order(d0, u0, order), order=order))
        if primary is not None:
            order = primary['ordered_nodes']
            pn = 'gpt_blind_consensus'
            if hc and 'llm' in rec['hc_order']:
                main_rows.append(row(base, 'hard', pn, {tuple(e) for e in rec['hc_order']['llm']['directed']}, order=order))
                main_rows.append(row(base, 'hard_skeleton', pn, A.orient_by_order(d0, u0, order), cp=cp, order=order))
            else:
                main_rows.append(row(base, 'hard', pn, A.orient_by_order(d0, u0, order), cp=cp, order=order))
            for name, (g, u, audit) in legacy_rules(nodes, d0, u0, primary, base).items():
                main_rows.append(row(base, name, pn, g, u, audit=audit, cp=cp, order=order))
            for variant in ['full', 'fixed_eta', 'no_update', 'pc_only', 'hc_only', 'no_support', 'estimate_all', 'unsigned', 'data_only']:
                g, u, audit = A.arc_orient(nodes, d0, u0, primary, boot_pc, boot_hc, variant=variant)
                main_rows.append(row(base, 'arc' if variant == 'full' else 'arc_' + variant, pn, g, u, audit=audit, cp=cp, order=order))
            for k in [1.0, 2.0, 8.0, 16.0]:
                g, u, audit = A.arc_orient(nodes, d0, u0, primary, boot_pc, boot_hc, k=k)
                main_rows.append(row(base, f'arc_k{k:g}', pn, g, u, audit=audit, order=order))
            for a0 in [0.65, 0.7, 0.8, 0.85]:
                g, u, audit = A.arc_orient(nodes, d0, u0, primary, boot_pc, boot_hc, data_a0=a0)
                main_rows.append(row(base, f'arc_a{a0:g}', pn, g, u, audit=audit, order=order))
            g, u, audit = A.arc_orient(nodes, d0, u0, primary, boot_pc, boot_hc, llm_a0='self_reported')
            main_rows.append(row(base, 'arc_selfconf', pn, g, u, audit=audit, order=order))
            for la in [0.65, 0.7, 0.75, 0.85, 0.9]:
                g, u, audit = A.arc_orient(nodes, d0, u0, primary, boot_pc, boot_hc, llm_a0=la)
                main_rows.append(row(base, f'arc_l{la:g}', pn, g, u, audit=audit, order=order))
            for ab in [0.0, 0.02, 0.1]:
                g, u, audit = A.arc_orient(nodes, d0, u0, primary, boot_pc, boot_hc, abstain=ab)
                main_rows.append(row(base, f'arc_abst{ab:g}', pn, g, u, audit=audit, order=order))
            main_rows.append(row(base, 'order_on_cpdag', pn, A.order_on_cpdag(nodes, cp[0], cp[1], order), cp=cp, order=order))
            g, u = A.mec_consistent_order(nodes, cp[0], cp[1], primary)
            main_rows.append(row(base, 'mec_consistent', pn, g, u, cp=cp, order=order))
        alts = {'gpt_blind_single': None, 'gpt_anon_consensus': ('gpt-5.4-mini', 'anon_full'), 'gpt_names_consensus': ('gpt-5.4-mini', 'blind_names'), 'gpt_named_single': ('gpt-5.4-mini', 'named_full'), 'deepseek_blind_consensus': ('deepseek-v4-flash', 'blind_full'), 'gemini_blind_consensus': ('gemini-2.5-flash-lite', 'blind_full'), 'gpt_blind_consensus': PRIMARY}
        for pn, spec in alts.items():
            if spec is None:
                recs = P.load_orders(CFG['priors'], coll, ds, *PRIMARY)
                pr = P.consensus_prior(nodes, recs[:1]) if recs else None
            else:
                pr = load(*spec)
            if pr is None:
                continue
            order = pr['ordered_nodes']
            prior_rows.append(row(base, 'hard_skeleton', pn, A.orient_by_order(d0, u0, order), order=order))
            g, u, audit = legacy_rules(nodes, d0, u0, pr, base)['soft']
            prior_rows.append(row(base, 'soft', pn, g, u, order=order))
            g, u, audit = A.arc_orient(nodes, d0, u0, pr, boot_pc, boot_hc)
            prior_rows.append(row(base, 'arc', pn, g, u, audit=audit, order=order))
            g, u, audit = A.arc_orient(nodes, d0, u0, pr, boot_pc, boot_hc, llm_a0='self_reported')
            prior_rows.append(row(base, 'arc_selfconf', pn, g, u, audit=audit, order=order))
        lf = CFG['legacy'] / coll / ds / 'gpt-5.4-mini__full__order.json'
        if lf.exists():
            lp = json.loads(lf.read_text(encoding='utf-8'))
            lp['ordered_nodes'] = B.restrict_order_to_nodes(lp['ordered_nodes'], nodes)
            order = lp['ordered_nodes']
            prior_rows.append(row(base, 'hard_skeleton', 'gpt_original_prompt', A.orient_by_order(d0, u0, order), order=order))
            g, u, audit = legacy_rules(nodes, d0, u0, lp, base)['soft']
            prior_rows.append(row(base, 'soft', 'gpt_original_prompt', g, u, order=order))
            g, u, audit = A.arc_orient(nodes, d0, u0, lp, boot_pc, boot_hc)
            prior_rows.append(row(base, 'arc', 'gpt_original_prompt', g, u, audit=audit, order=order))
        if oracle and rec['variant'] == 'main':
            for rho in RHOS:
                for rep in range(1 if rho in (0.0, 'rev') else 2):
                    rng = random.Random(B.stable_int('corrupt', ds, rec['n'], rec['seed'], rho, rep))
                    order = corrupted_order(nodes, oracle, rho, rng)
                    pr = {'ordered_nodes': order, 'confidence': 0.75, 'high_confidence_pairs': [], 'source': 'api', 'prompt_mode': 'full'}
                    ex = {'rho': rho, 'rep': rep}
                    cor_rows.append(row(base, 'data', 'corrupt', d0, u0, order=order, extra=ex))
                    cor_rows.append(row(base, 'hard_skeleton', 'corrupt', A.orient_by_order(d0, u0, order), order=order, extra=ex))
                    for name, (g, u, audit) in legacy_rules(nodes, d0, u0, pr, base).items():
                        cor_rows.append(row(base, name, 'corrupt', g, u, order=order, extra=ex))
                    for variant in ['full', 'fixed_eta', 'no_update', 'hc_only', 'pc_only', 'estimate_all', 'unsigned']:
                        g, u, audit = A.arc_orient(nodes, d0, u0, pr, boot_pc, boot_hc, variant=variant)
                        cor_rows.append(row(base, 'arc' if variant == 'full' else 'arc_' + variant, 'corrupt', g, u, audit=audit, order=order, extra=ex))
                    cor_rows.append(row(base, 'order_on_cpdag', 'corrupt', A.order_on_cpdag(nodes, cp[0], cp[1], order), order=order, extra=ex))
                    g, u = A.mec_consistent_order(nodes, cp[0], cp[1], pr)
                    cor_rows.append(row(base, 'mec_consistent', 'corrupt', g, u, order=order, extra=ex))
    return (main_rows, prior_rows, cor_rows)

def write(path, rows):
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: round(v, 6) if isinstance(v, float) else v for k, v in r.items()})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='runs')
    ap.add_argument('--priors', default='priors/blind')
    ap.add_argument('--code', default='.')
    ap.add_argument('--out', default='results')
    ap.add_argument('--workers', type=int, default=2)
    o = ap.parse_args()
    files = sorted((str(p) for p in Path(o.runs).glob('*/*.json') if p.name != 'run_config.json'))
    out = Path(o.out)
    out.mkdir(parents=True, exist_ok=True)
    M, R, C = ([], [], [])
    with mp.Pool(o.workers, initializer=init, initargs=(o.code, o.priors)) as pool:
        for i, (m, r, c) in enumerate(pool.imap_unordered(process, files, chunksize=4), 1):
            M += m
            R += r
            C += c
            if i % 50 == 0:
                print(f'{i}/{len(files)} run files', flush=True)
    key = lambda r: (r['variant'], r['collection'], r['dataset'], r['n'], r['seed'], r['base'], r['method'], r['prior'], str(r.get('rho', '')), r.get('rep', 0))
    write(out / 'results_main.csv', sorted(M, key=key))
    write(out / 'results_priors.csv', sorted(R, key=key))
    write(out / 'results_corruption.csv', sorted(C, key=key))
    print(f'wrote {len(M)} main, {len(R)} prior, {len(C)} corruption rows from {len(files)} run files')
if __name__ == '__main__':
    main()
