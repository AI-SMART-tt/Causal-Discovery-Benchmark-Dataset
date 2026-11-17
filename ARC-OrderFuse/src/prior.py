from __future__ import annotations
import json
from pathlib import Path

def load_orders(prior_dir: Path, collection: str, dataset: str, model: str, mode: str) -> list[dict]:
    files = sorted((Path(prior_dir) / collection / dataset).glob(f'{model}__{mode}__perm*.json'))
    return [json.loads(f.read_text(encoding='utf-8')) for f in files]

def consensus_prior(nodes: list[str], records: list[dict]) -> dict:
    orders = []
    for rec in records:
        seen = [n for n in rec['ordered_nodes'] if n in nodes]
        orders.append(seen + [n for n in nodes if n not in seen])
    K = len(orders)
    ranks = [{n: i for i, n in enumerate(o)} for o in orders]
    mean_rank = {n: sum((r[n] for r in ranks)) / K for n in nodes}
    first = ranks[0]
    order = sorted(nodes, key=lambda n: (mean_rank[n], first[n]))
    support = {}
    for i, a in enumerate(sorted(nodes)):
        for b in sorted(nodes)[i + 1:]:
            support[a, b] = sum((1 for r in ranks if r[a] < r[b])) / K
    confs = []
    for rec in records:
        try:
            confs.append(float(rec.get('confidence')))
        except (TypeError, ValueError):
            pass
    votes: dict[tuple[str, str], list[float]] = {}
    for rec in records:
        for p in rec.get('high_confidence_pairs', []) or []:
            try:
                votes.setdefault((p['before'], p['after']), []).append(float(p.get('confidence') or 0.7))
            except (TypeError, ValueError, KeyError):
                continue
    pairs = [{'before': a, 'after': b, 'confidence': sum(v) / len(v)} for (a, b), v in votes.items() if len(v) > K / 2 and len(votes.get((b, a), [])) == 0]
    return {'ordered_nodes': order, 'pair_support': support, 'K': K, 'confidence': sum(confs) / len(confs) if confs else 0.7, 'high_confidence_pairs': pairs, 'source': 'api_consensus_permutations', 'model': records[0].get('model', ''), 'prompt_mode': records[0].get('prompt_mode', '')}

def llm_votes(prior: dict, pairs) -> dict:
    sup = prior.get('pair_support')
    rank = {n: i for i, n in enumerate(prior['ordered_nodes'])}
    out = {}
    for a, b in pairs:
        if sup is not None and (a, b) in sup:
            out[a, b] = 2.0 * sup[a, b] - 1.0
        else:
            out[a, b] = 1.0 if rank[a] < rank[b] else -1.0
    return out
