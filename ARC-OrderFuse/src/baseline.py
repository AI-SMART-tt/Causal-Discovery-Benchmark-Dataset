from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
import os
import random
import re
import time
from collections import defaultdict, deque
from pathlib import Path
import urllib.error
import urllib.request
ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / 'data'
DEFAULT_RESULTS_DIR = ROOT / 'results'
DEFAULT_LLM_PRIOR_DIR = ROOT / 'llm_priors'
EPS = 1e-12
try:
    import numpy as np
except Exception:
    np = None
try:
    import pandas as pd
except Exception:
    pd = None

def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def read_data_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return (list(reader.fieldnames or []), rows)

def write_csv_dicts(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def read_edges(path: Path) -> set[tuple[str, str]]:
    edges = set()
    for row in read_csv_dicts(path):
        source, target = (row['source'], row['target'])
        if source != target:
            edges.add((source, target))
    return edges

def pick_sample_file(ds_dir: Path, dataset: str, sample_size: int) -> Path | None:
    samples = ds_dir / 'samples'
    exact = samples / f'{dataset}_n{sample_size}.csv'
    if exact.exists():
        return exact
    matches = sorted(samples.glob(f'*n{sample_size}.csv'))
    return matches[0] if matches else None

def parse_csv_list(value: str) -> set[str]:
    return {v.strip() for v in value.split(',') if v.strip()}

def parse_ordered_csv_list(value: str) -> list[str]:
    seen = set()
    items = []
    for raw in value.split(','):
        item = raw.strip()
        if item and item not in seen:
            seen.add(item)
            items.append(item)
    return items

def parse_int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(',') if v.strip()]

def skeleton(edges: set[tuple[str, str]], undirected: set[frozenset[str]] | None=None) -> set[frozenset[str]]:
    skel = {frozenset((a, b)) for a, b in edges if a != b}
    if undirected:
        skel |= set(undirected)
    return skel

def children_map(nodes: list[str], edges: set[tuple[str, str]]) -> dict[str, set[str]]:
    children = {n: set() for n in nodes}
    for src, dst in edges:
        children.setdefault(src, set()).add(dst)
        children.setdefault(dst, set())
    return children

def has_path(nodes: list[str], edges: set[tuple[str, str]], source: str, target: str) -> bool:
    if source == target:
        return True
    children = children_map(nodes, edges)
    seen = {source}
    stack = [source]
    while stack:
        node = stack.pop()
        for child in children.get(node, ()):
            if child == target:
                return True
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return False

def is_acyclic(nodes: list[str], edges: set[tuple[str, str]]) -> bool:
    return topological_sort(nodes, edges) is not None

def topological_sort(nodes: list[str], edges: set[tuple[str, str]]) -> list[str] | None:
    indeg = {n: 0 for n in nodes}
    children = {n: [] for n in nodes}
    node_rank = {node: i for i, node in enumerate(nodes)}
    for src, dst in sorted(edges, key=lambda e: (node_rank.get(e[0], len(nodes)), node_rank.get(e[1], len(nodes)), e[0], e[1])):
        if src not in indeg:
            indeg[src] = 0
            children[src] = []
        if dst not in indeg:
            indeg[dst] = 0
            children[dst] = []
        children[src].append(dst)
        indeg[dst] += 1
    q = deque([n for n in nodes if indeg.get(n, 0) == 0])
    order = []
    while q:
        node = q.popleft()
        order.append(node)
        for child in sorted(children.get(node, []), key=lambda n: (node_rank.get(n, len(nodes)), n)):
            indeg[child] -= 1
            if indeg[child] == 0:
                q.append(child)
    return order if len(order) == len(indeg) else None

def transitive_ancestor_pairs(nodes: list[str], edges: set[tuple[str, str]]) -> set[tuple[str, str]]:
    children = children_map(nodes, edges)
    pairs = set()
    for src in nodes:
        stack = list(children.get(src, ()))
        seen = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            pairs.add((src, node))
            stack.extend(children.get(node, ()))
    return pairs

def complete_cpdag_to_dag(nodes: list[str], directed_edges: set[tuple[str, str]], undirected_edges: set[frozenset[str]]) -> set[tuple[str, str]]:
    node_rank = {node: i for i, node in enumerate(nodes)}
    completed = set(directed_edges)
    for pair in sorted(undirected_edges, key=lambda p: tuple(sorted(p))):
        a, b = tuple(pair)
        first, second = (a, b) if node_rank.get(a, 0) <= node_rank.get(b, 0) else (b, a)
        if not has_path(nodes, completed, second, first):
            completed.add((first, second))
        elif not has_path(nodes, completed, first, second):
            completed.add((second, first))
    return completed

def oracle_order_from_edges(nodes: list[str], true_edges: set[tuple[str, str]]) -> list[str] | None:
    filtered_edges = {(src, dst) for src, dst in true_edges if src in nodes and dst in nodes}
    return topological_sort(nodes, filtered_edges)

def orient_skeleton_by_order(nodes: list[str], directed_edges: set[tuple[str, str]], undirected_edges: set[frozenset[str]], oracle_order: list[str]) -> set[tuple[str, str]]:
    rank = {node: i for i, node in enumerate(oracle_order)}
    fallback_rank = {node: i for i, node in enumerate(nodes)}
    oriented = set()
    for pair in skeleton(directed_edges, undirected_edges):
        a, b = tuple(pair)
        rank_a = rank.get(a, fallback_rank.get(a, 0))
        rank_b = rank.get(b, fallback_rank.get(b, 0))
        if rank_a <= rank_b:
            oriented.add((a, b))
        else:
            oriented.add((b, a))
    return oriented

def apply_oracle_order_to_prediction(nodes: list[str], prediction: dict, oracle_order: list[str] | None, base_name: str) -> dict:
    if oracle_order is None:
        raise AlgorithmSkip('oracle order is unavailable because the true graph is not a DAG')
    oriented = orient_skeleton_by_order(nodes, prediction['directed_edges'], prediction['undirected_edges'], oracle_order)
    extra = dict(prediction.get('extra', {}))
    extra.update({'oracle_order': 'true_topological_order', 'oracle_mode': 'orient_predicted_skeleton', 'base_algorithm': base_name})
    return {'directed_edges': oriented, 'undirected_edges': set(), 'completed_edges': oriented, 'extra': extra}

def canonical_algorithm_name(algorithm: str) -> str:
    return {'pc': 'pc_lite', 'hc_bic': 'hc_bic_lite'}.get(algorithm, algorithm)
LLM_GUIDANCE_SUFFIXES = ('_llm_consensus_conflict_order', '_llm_consensus_soft_order', '_llm_consensus_order', '_llm_adaptive_order', '_llm_order', '_llm_soft_order', '_llm_conflict_order')
CONTROL_ORDER_SUFFIXES = ('_random_order', '_wrong_order')

def strip_algorithm_suffix(algorithm: str, suffixes: tuple[str, ...]) -> tuple[str, str]:
    canonical = canonical_algorithm_name(algorithm)
    for suffix in suffixes:
        if canonical.endswith(suffix):
            return (canonical[:-len(suffix)], suffix)
    return (canonical, '')

def algorithm_uses_llm_order(algorithm: str) -> bool:
    _, suffix = strip_algorithm_suffix(algorithm, LLM_GUIDANCE_SUFFIXES)
    return bool(suffix)

def algorithm_uses_control_order(algorithm: str) -> bool:
    _, suffix = strip_algorithm_suffix(algorithm, CONTROL_ORDER_SUFFIXES)
    return bool(suffix)

def algorithm_uses_random_order(algorithm: str) -> bool:
    return canonical_algorithm_name(algorithm).endswith('_random_order')

def algorithm_uses_wrong_order(algorithm: str) -> bool:
    return canonical_algorithm_name(algorithm).endswith('_wrong_order')

def stable_int(*parts: object) -> int:
    text = '::'.join((str(part) for part in parts))
    return int(hashlib.sha256(text.encode('utf-8')).hexdigest()[:16], 16)

def model_slug(model: str) -> str:
    slug = re.sub('[^A-Za-z0-9_.-]+', '_', model.strip())
    return slug.strip('._-') or 'model'

def load_node_metadata(ds_dir: Path, nodes: list[str] | None=None) -> list[dict[str, str]]:
    path = ds_dir / 'nodes.csv'
    rows = read_csv_dicts(path) if path.exists() else []
    by_node = {row.get('node', ''): row for row in rows}
    ordered_nodes = nodes or [row.get('node', '') for row in rows if row.get('node')]
    metadata = []
    for node in ordered_nodes:
        row = dict(by_node.get(node, {}))
        row.setdefault('node', node)
        row.setdefault('description', '')
        row.setdefault('states', '')
        row.setdefault('state_count', '')
        metadata.append(row)
    return metadata

def perturb_node_metadata(node_rows: list[dict[str, str]], mode: str, seed_key: str) -> list[dict[str, str]]:
    rows = [dict(row) for row in node_rows]
    if mode == 'full':
        return rows
    if mode == 'names_only':
        for row in rows:
            row['description'] = ''
            row['states'] = ''
            row['state_count'] = ''
        return rows
    if mode == 'state_only':
        for row in rows:
            row['description'] = ''
        return rows
    if mode == 'shuffled_descriptions':
        rng = random.Random(stable_int(seed_key, mode))
        payloads = [(row.get('description', ''), row.get('states', ''), row.get('state_count', '')) for row in rows]
        rng.shuffle(payloads)
        for row, payload in zip(rows, payloads):
            row['description'], row['states'], row['state_count'] = payload
        return rows
    if mode == 'masked_descriptions':
        sensitive_tokens = ['cancer', 'tuberculosis', 'bronchitis', 'smoking', 'smoke', 'x-ray', 'xray', 'dyspnoea', 'dyspnea', 'earthquake', 'alarm', 'disease', 'infection', 'virus', 'immune', 'inflammatory', 'cough', 'pain', 'asia', 'travel']
        pattern = re.compile('|'.join((re.escape(token) for token in sensitive_tokens)), re.IGNORECASE)
        for row in rows:
            row['description'] = pattern.sub('[MASK]', row.get('description', ''))
        return rows
    if mode == 'generic_descriptions':
        for row in rows:
            row['description'] = 'Observed variable in this causal discovery dataset.'
            row['states'] = ''
            row['state_count'] = ''
        return rows
    raise ValueError(f'Unknown LLM prompt perturbation mode: {mode}')

def build_llm_order_prompt(collection: str, dataset: str, node_rows: list[dict[str, str]], prompt_mode: str) -> tuple[str, str]:
    variables = []
    for row in node_rows:
        variables.append({'id': row.get('node', ''), 'description': row.get('description', ''), 'states': row.get('states', ''), 'state_count': row.get('state_count', '')})
    variable_json = json.dumps(variables, ensure_ascii=False, indent=2)
    system = 'You are a careful causal discovery expert. Your task is to convert variable semantics into a plausible causal order prior. Do not claim that the order is ground truth. Return strict JSON only.'
    user = f'Dataset collection: {collection}\nDataset name: {dataset}\n\nPrompt perturbation mode: {prompt_mode}\n\nVariables are listed below. Use the exact id strings.\n{variable_json}\n\nTask: propose a causal order prior. Earlier variables are potential causes or upstream conditions; later variables are downstream effects, measurements, symptoms, decisions, or aggregate outcomes.\n\nReturn one JSON object with this schema:\n{{\n  "ordered_nodes": ["exact id 1", "exact id 2", "..."],\n  "confidence": 0.0,\n  "high_confidence_pairs": [\n    {{"before": "exact id", "after": "exact id", "confidence": 0.0}}\n  ],\n  "rationale": "brief, no more than 80 words"\n}}\n\nConstraints: ordered_nodes must contain every provided id exactly once; do not invent nodes; if evidence is weak, still provide the best semantic prior and lower the confidence.'
    return (system, user)

def read_llm_api_key(args: argparse.Namespace) -> tuple[str, str]:
    for env_name in parse_ordered_csv_list(args.llm_api_key_env):
        value = os.getenv(env_name)
        if value:
            return (value, env_name)
    return ('', '')

def response_message_content(response: object) -> str:
    if isinstance(response, dict):
        choices = response.get('choices')
    else:
        choices = getattr(response, 'choices', None)
        if choices is None and hasattr(response, 'model_dump'):
            try:
                dumped = response.model_dump()
                choices = dumped.get('choices') if isinstance(dumped, dict) else None
            except Exception:
                choices = None
    if not choices:
        return ''
    first_choice = choices[0]
    if isinstance(first_choice, dict):
        message = first_choice.get('message')
    else:
        message = getattr(first_choice, 'message', None)
    if message is None:
        return ''
    if isinstance(message, dict):
        content = message.get('content', '')
    else:
        content = getattr(message, 'content', '')
    if isinstance(content, list):
        return '\n'.join((str(item) for item in content))
    return str(content)

def require_nonempty_llm_content(response: object, client_name: str) -> str:
    content = response_message_content(response)
    if content.strip():
        return content
    raise RuntimeError(f'{client_name} returned an empty message content')

def call_zhipu_chat(api_key: str, model: str, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int, timeout: int) -> tuple[str, str]:
    messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
    try:
        from zai import ZhipuAiClient
        client = ZhipuAiClient(api_key=api_key)
        response = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        return (require_nonempty_llm_content(response, 'zai-sdk'), 'zai-sdk')
    except Exception as zai_exc:
        first_error = zai_exc
    try:
        from zhipuai import ZhipuAI
        client = ZhipuAI(api_key=api_key)
        response = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        return (require_nonempty_llm_content(response, 'zhipuai'), 'zhipuai')
    except Exception as zhipu_exc:
        second_error = zhipu_exc
    payload = json.dumps({'model': model, 'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens}, ensure_ascii=False).encode('utf-8')
    request = urllib.request.Request('https://open.bigmodel.cn/api/paas/v4/chat/completions', data=payload, headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode('utf-8')
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'Zhipu API call failed by SDK and HTTP fallback. zai-sdk={type(first_error).__name__}: {first_error}; zhipuai={type(second_error).__name__}: {second_error}; http={exc.code}: {detail[:500]}') from exc
    data = json.loads(body)
    content = response_message_content(data)
    if not content.strip():
        raise RuntimeError(f'Zhipu API call returned empty content through all clients. zai-sdk={type(first_error).__name__}: {first_error}; zhipuai={type(second_error).__name__}: {second_error}; http_body={body[:500]}')
    return (content, 'http')

def extract_json_object(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith('```'):
        stripped = re.sub('^```(?:json)?', '', stripped.strip(), flags=re.IGNORECASE).strip()
        stripped = re.sub('```$', '', stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    start = stripped.find('{')
    if start < 0:
        raise ValueError('LLM response does not contain a JSON object')
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(stripped)):
        ch = stripped[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return json.loads(stripped[start:idx + 1])
    raise ValueError('LLM response JSON object is incomplete')

def canonicalize_node(raw: object, nodes: list[str]) -> str | None:
    if isinstance(raw, dict):
        raw = raw.get('id') or raw.get('node') or raw.get('name') or raw.get('variable')
    if raw is None:
        return None
    text = str(raw).strip()
    if text in nodes:
        return text
    lower_map = {node.lower().strip(): node for node in nodes}
    if text.lower().strip() in lower_map:
        return lower_map[text.lower().strip()]
    compact_map = {re.sub('\\s+', ' ', node.lower().strip()): node for node in nodes}
    compact = re.sub('\\s+', ' ', text.lower().strip())
    return compact_map.get(compact)

def repair_order(raw_order: object, nodes: list[str]) -> tuple[list[str], list[str], list[str]]:
    if not isinstance(raw_order, list):
        raw_order = []
    ordered = []
    seen = set()
    unknown = []
    for item in raw_order:
        node = canonicalize_node(item, nodes)
        if node is None:
            unknown.append(str(item))
            continue
        if node not in seen:
            seen.add(node)
            ordered.append(node)
    missing = [node for node in nodes if node not in seen]
    ordered.extend(missing)
    return (ordered, missing, unknown)

def semantic_heuristic_order(node_rows: list[dict[str, str]], nodes: list[str]) -> list[str]:
    early = {'age': -35, 'gender': -35, 'sex': -35, 'visit': -30, 'travel': -30, 'smok': -30, 'pollution': -25, 'exposure': -25, 'virus enters': -25, 'infection': -15, 'history': -15}
    middle = {'cancer': 5, 'disease': 5, 'tuberculosis': 5, 'bronchitis': 5, 'immune': 10, 'inflammatory': 10, 'viremia': 10, 'leakage': 15, 'either': 20}
    late = {'symptom': 30, 'pain': 30, 'cough': 35, 'dysp': 35, 'x-ray': 40, 'xray': 40, 'test': 40, 'diagnosis': 45, 'positive': 45, 'result': 45, 'death': 55, 'outcome': 55}
    row_by_node = {row.get('node', ''): row for row in node_rows}
    scored = []
    for idx, node in enumerate(nodes):
        row = row_by_node.get(node, {})
        text = f"{node} {row.get('description', '')}".lower()
        score = 50.0 + idx * 0.001
        for token, delta in early.items():
            if token in text:
                score += delta
        for token, delta in middle.items():
            if token in text:
                score += delta
        for token, delta in late.items():
            if token in text:
                score += delta
        scored.append((score, idx, node))
    return [node for _, _, node in sorted(scored)]

def parse_llm_order_payload(payload: dict, nodes: list[str]) -> dict:
    raw_order = payload.get('ordered_nodes') or payload.get('causal_order') or payload.get('order') or payload.get('nodes') or []
    ordered, missing, unknown = repair_order(raw_order, nodes)
    confidence = payload.get('confidence', '')
    try:
        confidence = round(max(0.0, min(1.0, float(confidence))), 6)
    except (TypeError, ValueError):
        confidence = ''
    return {'ordered_nodes': ordered, 'confidence': confidence, 'missing_nodes_appended': missing, 'unknown_nodes_ignored': unknown, 'high_confidence_pairs': payload.get('high_confidence_pairs', []), 'rationale': str(payload.get('rationale', ''))[:1000]}

def cache_llm_prior(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')

def write_llm_failure(path: Path, text: str, error: Exception) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'error': f'{type(error).__name__}: {error}', 'raw_response': text, 'created_at': dt.datetime.now(dt.timezone.utc).isoformat()}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')

def display_cache_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)

def load_or_create_llm_order_prior(collection: str, dataset: str, ds_dir: Path, nodes: list[str], args: argparse.Namespace) -> dict | None:
    raw_node_rows = load_node_metadata(ds_dir, nodes)
    prompt_mode = args.llm_prompt_mode
    node_rows = perturb_node_metadata(raw_node_rows, prompt_mode, f'{collection}/{dataset}')
    system_prompt, user_prompt = build_llm_order_prompt(collection, dataset, node_rows, prompt_mode)
    prompt_hash = hashlib.sha256((system_prompt + '\n' + user_prompt).encode('utf-8')).hexdigest()[:16]
    prior_dir = Path(args.llm_prior_dir) / collection / dataset
    cache_path = prior_dir / f'{model_slug(args.llm_model)}__{model_slug(prompt_mode)}__order.json'
    legacy_cache_path = prior_dir / f'{model_slug(args.llm_model)}__order.json'
    if cache_path.exists() and (not args.llm_refresh):
        record = json.loads(cache_path.read_text(encoding='utf-8'))
        parsed = parse_llm_order_payload({'ordered_nodes': record.get('ordered_nodes', [])}, nodes)
        record['ordered_nodes'] = parsed['ordered_nodes']
        record.setdefault('source', 'cache')
        record['cache_path'] = display_cache_path(cache_path)
        record.setdefault('prompt_mode', prompt_mode)
        return record
    if prompt_mode == 'full' and legacy_cache_path.exists() and (not args.llm_refresh):
        record = json.loads(legacy_cache_path.read_text(encoding='utf-8'))
        parsed = parse_llm_order_payload({'ordered_nodes': record.get('ordered_nodes', [])}, nodes)
        record['ordered_nodes'] = parsed['ordered_nodes']
        record.setdefault('source', 'cache')
        record['cache_path'] = display_cache_path(legacy_cache_path)
        record.setdefault('prompt_mode', prompt_mode)
        return record
    if args.llm_cache_only:
        if args.llm_fallback == 'heuristic':
            order = semantic_heuristic_order(node_rows, nodes)
            record = {'collection': collection, 'dataset': dataset, 'model': args.llm_model, 'provider': 'heuristic', 'source': 'heuristic_fallback', 'prompt_mode': prompt_mode, 'created_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'prompt_hash': prompt_hash, 'ordered_nodes': order, 'confidence': 0.35, 'rationale': 'Heuristic fallback used because --llm-cache-only was set.', 'cache_path': display_cache_path(cache_path)}
            cache_llm_prior(cache_path, record)
            return record
        return None
    api_key, api_key_env = read_llm_api_key(args)
    if not api_key:
        if args.llm_fallback == 'heuristic':
            order = semantic_heuristic_order(node_rows, nodes)
            record = {'collection': collection, 'dataset': dataset, 'model': args.llm_model, 'provider': 'heuristic', 'source': 'heuristic_fallback', 'prompt_mode': prompt_mode, 'created_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'prompt_hash': prompt_hash, 'ordered_nodes': order, 'confidence': 0.35, 'rationale': 'Heuristic fallback used because no LLM API key was available.', 'cache_path': display_cache_path(cache_path)}
            cache_llm_prior(cache_path, record)
            return record
        return None
    try:
        text, client_name = call_zhipu_chat(api_key=api_key, model=args.llm_model, system_prompt=system_prompt, user_prompt=user_prompt, temperature=args.llm_temperature, max_tokens=args.llm_max_tokens, timeout=args.llm_timeout)
        repair_response = ''
        try:
            payload = extract_json_object(text)
        except Exception as parse_exc:
            repair_system = 'You repair malformed model output into strict JSON. Return exactly one JSON object and no markdown.'
            repair_user = f'The previous response was not valid JSON for the causal-order schema. Convert it into exactly this schema using only the provided node ids:\n{{\n  "ordered_nodes": ["exact id"],\n  "confidence": 0.0,\n  "high_confidence_pairs": [\n    {{"before": "exact id", "after": "exact id", "confidence": 0.0}}\n  ],\n  "rationale": "brief"\n}}\n\nAllowed node ids: {json.dumps(nodes, ensure_ascii=False)}\n\nPrevious response:\n{text[:6000]}'
            try:
                repair_response, repair_client = call_zhipu_chat(api_key=api_key, model=args.llm_model, system_prompt=repair_system, user_prompt=repair_user, temperature=0.0, max_tokens=args.llm_max_tokens, timeout=args.llm_timeout)
                client_name = f'{client_name}+repair:{repair_client}'
                payload = extract_json_object(repair_response)
            except Exception as repair_exc:
                write_llm_failure(cache_path.with_suffix('.failed.json'), text, parse_exc)
                raise repair_exc
        parsed = parse_llm_order_payload(payload, nodes)
        record = {'collection': collection, 'dataset': dataset, 'model': args.llm_model, 'provider': 'zhipu', 'client': client_name, 'source': 'zhipu_glm_api', 'prompt_mode': prompt_mode, 'api_key_env': api_key_env, 'temperature': args.llm_temperature, 'max_tokens': args.llm_max_tokens, 'created_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'prompt_hash': prompt_hash, 'ordered_nodes': parsed['ordered_nodes'], 'confidence': parsed['confidence'], 'missing_nodes_appended': parsed['missing_nodes_appended'], 'unknown_nodes_ignored': parsed['unknown_nodes_ignored'], 'high_confidence_pairs': parsed['high_confidence_pairs'], 'rationale': parsed['rationale'], 'raw_response': text, 'repair_response': repair_response, 'cache_path': display_cache_path(cache_path)}
        cache_llm_prior(cache_path, record)
        return record
    except Exception as exc:
        failure_path = cache_path.with_suffix('.failed.json')
        if not failure_path.exists():
            write_llm_failure(failure_path, '', exc)
        if args.llm_fallback == 'heuristic':
            order = semantic_heuristic_order(node_rows, nodes)
            record = {'collection': collection, 'dataset': dataset, 'model': args.llm_model, 'provider': 'heuristic', 'source': 'heuristic_fallback', 'prompt_mode': prompt_mode, 'created_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'prompt_hash': prompt_hash, 'ordered_nodes': order, 'confidence': 0.25, 'rationale': f'Heuristic fallback used after LLM call failed: {type(exc).__name__}', 'llm_error': f'{type(exc).__name__}: {str(exc)[:500]}', 'cache_path': display_cache_path(cache_path)}
            cache_llm_prior(cache_path, record)
            print(f'[llm-fallback] {collection}/{dataset}: {type(exc).__name__}: {exc}')
            return record
        raise

def restrict_order_to_nodes(order: list[str] | None, nodes: list[str]) -> list[str] | None:
    if not order:
        return None
    fixed, _, _ = repair_order(order, nodes)
    return fixed

def apply_llm_order_to_prediction(nodes: list[str], prediction: dict, llm_order: list[str] | None, base_name: str, llm_prior: dict | None) -> dict:
    if llm_order is None:
        raise AlgorithmSkip('LLM order prior is unavailable')
    oriented = orient_skeleton_by_order(nodes, prediction['directed_edges'], prediction['undirected_edges'], llm_order)
    extra = dict(prediction.get('extra', {}))
    extra.update({'llm_order': 'causal_order_prior', 'llm_mode': 'orient_predicted_skeleton', 'llm_order_source': (llm_prior or {}).get('source', ''), 'llm_model': (llm_prior or {}).get('model', ''), 'llm_prompt_mode': (llm_prior or {}).get('prompt_mode', ''), 'llm_confidence': (llm_prior or {}).get('confidence', ''), 'base_algorithm': base_name})
    return {'directed_edges': oriented, 'undirected_edges': set(), 'completed_edges': oriented, 'extra': extra}

def make_random_order(nodes: list[str], collection: str, dataset: str, sample_size: int, seed: int, order_seed: int) -> list[str]:
    order = list(nodes)
    rng = random.Random(stable_int('random_order', collection, dataset, sample_size, seed, order_seed))
    rng.shuffle(order)
    return order

def make_wrong_order(nodes: list[str], oracle_order: list[str] | None) -> list[str] | None:
    if oracle_order is None:
        return None
    return list(reversed(oracle_order))

def apply_named_order_to_prediction(nodes: list[str], prediction: dict, order: list[str] | None, base_name: str, order_type: str) -> dict:
    if order is None:
        raise AlgorithmSkip(f'{order_type} order prior is unavailable')
    oriented = orient_skeleton_by_order(nodes, prediction['directed_edges'], prediction['undirected_edges'], order)
    extra = dict(prediction.get('extra', {}))
    extra.update({'order_prior': order_type, 'order_mode': 'orient_predicted_skeleton', 'base_algorithm': base_name})
    return {'directed_edges': oriented, 'undirected_edges': set(), 'completed_edges': oriented, 'extra': extra}

def llm_prior_confidence(llm_prior: dict | None, default_confidence: float) -> float:
    value = (llm_prior or {}).get('confidence', '')
    try:
        return max(0.5, min(0.99, float(value)))
    except (TypeError, ValueError):
        return default_confidence

def llm_prior_raw_confidence(llm_prior: dict | None, default_confidence: float) -> float:
    value = (llm_prior or {}).get('confidence', '')
    try:
        return max(0.0, min(0.99, float(value)))
    except (TypeError, ValueError):
        return max(0.0, min(0.99, default_confidence))

def llm_source_reliability(llm_prior: dict | None, args: argparse.Namespace) -> float:
    source = str((llm_prior or {}).get('source', '')).lower()
    provider = str((llm_prior or {}).get('provider', '')).lower()
    prompt_mode = str((llm_prior or {}).get('prompt_mode', '')).lower()
    if 'self_consistency' in source or 'consensus' in source:
        base = args.adaptive_consensus_source_weight
    elif 'api' in source or 'zhipu' in source or 'glm' in provider:
        base = args.adaptive_api_source_weight
    elif 'heuristic' in source:
        base = args.adaptive_heuristic_source_weight
    else:
        base = 1.0
    if prompt_mode and prompt_mode not in {'full', 'self_consistency', 'consensus'}:
        base *= args.adaptive_perturbed_prompt_source_weight
    return max(0.0, min(1.5, base))

def rank_separation_before_prob(before: str, after: str, llm_order: list[str] | None, raw_confidence: float, gain: float) -> float:
    if not llm_order:
        return 0.5
    rank = {node: i for i, node in enumerate(llm_order)}
    if before not in rank or after not in rank or len(rank) <= 1:
        return 0.5
    sep = abs(rank[before] - rank[after]) / max(len(rank) - 1, 1)
    calibrated_conf = max(0.0, raw_confidence - 0.5) / 0.49
    margin = min(0.49, max(0.0, gain) * min(1.0, calibrated_conf) * sep)
    return 0.5 + margin if rank[before] < rank[after] else 0.5 - margin

def confidence_power(value: float, power: float) -> float:
    value = max(0.0, value)
    if power <= 0:
        return value
    return value ** power

def llm_pairwise_before_prob(before: str, after: str, llm_order: list[str] | None, llm_prior: dict | None, default_confidence: float) -> float:
    if not llm_order:
        return 0.5
    rank = {node: i for i, node in enumerate(llm_order)}
    if before not in rank or after not in rank:
        return 0.5
    base_conf = llm_prior_confidence(llm_prior, default_confidence)
    prob = base_conf if rank[before] < rank[after] else 1.0 - base_conf
    for pair in (llm_prior or {}).get('high_confidence_pairs', []) or []:
        p_before = canonicalize_node(pair.get('before'), list(rank))
        p_after = canonicalize_node(pair.get('after'), list(rank))
        try:
            pair_conf = max(0.5, min(0.99, float(pair.get('confidence', base_conf))))
        except (TypeError, ValueError):
            pair_conf = base_conf
        if p_before == before and p_after == after:
            prob = pair_conf
        elif p_before == after and p_after == before:
            prob = 1.0 - pair_conf
    return max(0.01, min(0.99, prob))

def data_before_prob_for_pair(a: str, b: str, directed_edges: set[tuple[str, str]], data_confidence: float) -> tuple[float, str]:
    if (a, b) in directed_edges:
        return (data_confidence, 'a_before_b')
    if (b, a) in directed_edges:
        return (1.0 - data_confidence, 'b_before_a')
    return (0.5, 'undirected')

def orient_pairs_by_scores(nodes: list[str], pair_scores: list[dict]) -> tuple[set[tuple[str, str]], int]:
    directed: set[tuple[str, str]] = set()
    cycle_avoids = 0
    for item in sorted(pair_scores, key=lambda r: (-r['margin'], r['a'], r['b'])):
        a, b = (item['a'], item['b'])
        first, second = (a, b) if item['prob_a_before_b'] >= 0.5 else (b, a)
        if not has_path(nodes, directed, second, first):
            directed.add((first, second))
            continue
        if not has_path(nodes, directed, first, second):
            directed.add((second, first))
            cycle_avoids += 1
    return (directed, cycle_avoids)

def apply_soft_llm_prior_to_prediction(nodes: list[str], prediction: dict, llm_order: list[str] | None, base_name: str, llm_prior: dict | None, args: argparse.Namespace, conflict_aware: bool) -> dict:
    if llm_order is None:
        raise AlgorithmSkip('LLM order prior is unavailable')
    pair_scores = []
    conflicts = 0
    conflict_llm_wins = 0
    conflict_data_wins = 0
    data_conf = max(0.5, min(0.99, args.data_directed_confidence))
    for pair in skeleton(prediction['directed_edges'], prediction['undirected_edges']):
        a, b = tuple(sorted(pair))
        data_prob, data_state = data_before_prob_for_pair(a, b, prediction['directed_edges'], data_conf)
        llm_prob = llm_pairwise_before_prob(a, b, llm_order, llm_prior, args.llm_order_default_confidence)
        llm_strength = abs(llm_prob - 0.5) * 2
        data_strength = abs(data_prob - 0.5) * 2
        data_direction = 1 if data_prob > 0.5 else -1 if data_prob < 0.5 else 0
        llm_direction = 1 if llm_prob > 0.5 else -1 if llm_prob < 0.5 else 0
        is_conflict = data_direction and llm_direction and (data_direction != llm_direction)
        weight = args.llm_soft_weight
        if conflict_aware and is_conflict:
            conflicts += 1
            if llm_strength < args.llm_conflict_trust_threshold:
                weight = args.llm_conflict_weight
        fused_prob = (1.0 - weight) * data_prob + weight * llm_prob
        if is_conflict:
            fused_direction = 1 if fused_prob >= 0.5 else -1
            if fused_direction == llm_direction:
                conflict_llm_wins += 1
            else:
                conflict_data_wins += 1
        pair_scores.append({'a': a, 'b': b, 'prob_a_before_b': fused_prob, 'margin': abs(fused_prob - 0.5), 'data_state': data_state})
    oriented, cycle_avoids = orient_pairs_by_scores(nodes, pair_scores)
    extra = dict(prediction.get('extra', {}))
    extra.update({'llm_order': 'causal_order_prior', 'llm_mode': 'conflict_aware_soft_fusion' if conflict_aware else 'soft_order_fusion', 'llm_order_source': (llm_prior or {}).get('source', ''), 'llm_model': (llm_prior or {}).get('model', ''), 'llm_prompt_mode': (llm_prior or {}).get('prompt_mode', ''), 'llm_confidence': (llm_prior or {}).get('confidence', ''), 'llm_soft_weight': args.llm_soft_weight, 'llm_conflict_weight': args.llm_conflict_weight if conflict_aware else '', 'llm_data_conflicts': conflicts, 'llm_conflict_llm_wins': conflict_llm_wins, 'llm_conflict_data_wins': conflict_data_wins, 'cycle_avoids': cycle_avoids, 'base_algorithm': base_name})
    return {'directed_edges': oriented, 'undirected_edges': set(), 'completed_edges': oriented, 'extra': extra}

def apply_adaptive_llm_prior_to_prediction(nodes: list[str], prediction: dict, llm_order: list[str] | None, base_name: str, llm_prior: dict | None, args: argparse.Namespace) -> dict:
    if llm_order is None:
        raise AlgorithmSkip('LLM order prior is unavailable')
    pair_scores = []
    conflicts = 0
    conflict_llm_wins = 0
    conflict_data_wins = 0
    data_conf = max(0.5, min(0.99, args.data_directed_confidence))
    raw_conf = llm_prior_raw_confidence(llm_prior, args.llm_order_default_confidence)
    source_weight = llm_source_reliability(llm_prior, args)
    llm_weight_values = []
    data_weight_values = []
    llm_strength_values = []
    data_strength_values = []
    conflict_weight_values = []
    for pair in skeleton(prediction['directed_edges'], prediction['undirected_edges']):
        a, b = tuple(sorted(pair))
        data_prob, data_state = data_before_prob_for_pair(a, b, prediction['directed_edges'], data_conf)
        base_llm_prob = llm_pairwise_before_prob(a, b, llm_order, llm_prior, args.llm_order_default_confidence)
        rank_prob = rank_separation_before_prob(a, b, llm_order, raw_conf, args.adaptive_rank_separation_gain)
        if abs(rank_prob - 0.5) > abs(base_llm_prob - 0.5):
            llm_prob = rank_prob
        else:
            llm_prob = base_llm_prob
        llm_strength = abs(llm_prob - 0.5) * 2.0
        data_strength = abs(data_prob - 0.5) * 2.0
        data_direction = 1 if data_prob > 0.5 else -1 if data_prob < 0.5 else 0
        llm_direction = 1 if llm_prob > 0.5 else -1 if llm_prob < 0.5 else 0
        is_conflict = data_direction and llm_direction and (data_direction != llm_direction)
        llm_reliability = source_weight * confidence_power(llm_strength, args.adaptive_reliability_power) * (args.adaptive_confidence_floor + (1.0 - args.adaptive_confidence_floor) * raw_conf)
        data_reliability = args.adaptive_data_reliability * confidence_power(data_strength, args.adaptive_reliability_power)
        if data_state == 'undirected':
            data_reliability *= args.adaptive_undirected_data_discount
        if is_conflict:
            conflicts += 1
            data_reliability *= args.adaptive_conflict_data_boost
            if raw_conf < args.llm_conflict_trust_threshold:
                llm_reliability *= args.adaptive_low_conflict_llm_discount
        denom = llm_reliability + data_reliability + 1e-12
        weight = llm_reliability / denom
        weight = max(args.adaptive_min_llm_weight, min(args.adaptive_max_llm_weight, weight))
        fused_prob = (1.0 - weight) * data_prob + weight * llm_prob
        if is_conflict:
            conflict_weight_values.append(weight)
            fused_direction = 1 if fused_prob >= 0.5 else -1
            if fused_direction == llm_direction:
                conflict_llm_wins += 1
            else:
                conflict_data_wins += 1
        llm_weight_values.append(weight)
        data_weight_values.append(1.0 - weight)
        llm_strength_values.append(llm_strength)
        data_strength_values.append(data_strength)
        pair_scores.append({'a': a, 'b': b, 'prob_a_before_b': fused_prob, 'margin': abs(fused_prob - 0.5), 'data_state': data_state})
    oriented, cycle_avoids = orient_pairs_by_scores(nodes, pair_scores)
    extra = dict(prediction.get('extra', {}))
    extra.update({'llm_order': 'causal_order_prior', 'llm_mode': 'adaptive_reliability_calibrated_fusion', 'llm_order_source': (llm_prior or {}).get('source', ''), 'llm_model': (llm_prior or {}).get('model', ''), 'llm_prompt_mode': (llm_prior or {}).get('prompt_mode', ''), 'llm_confidence': (llm_prior or {}).get('confidence', ''), 'adaptive_llm_weight': round(mean(llm_weight_values), 6) if llm_weight_values else '', 'adaptive_data_weight': round(mean(data_weight_values), 6) if data_weight_values else '', 'adaptive_conflict_llm_weight': round(mean(conflict_weight_values), 6) if conflict_weight_values else '', 'adaptive_llm_strength': round(mean(llm_strength_values), 6) if llm_strength_values else '', 'adaptive_data_strength': round(mean(data_strength_values), 6) if data_strength_values else '', 'adaptive_source_weight': round(source_weight, 6), 'adaptive_raw_confidence': round(raw_conf, 6), 'llm_soft_weight': '', 'llm_conflict_weight': '', 'llm_data_conflicts': conflicts, 'llm_conflict_llm_wins': conflict_llm_wins, 'llm_conflict_data_wins': conflict_data_wins, 'cycle_avoids': cycle_avoids, 'base_algorithm': base_name})
    return {'directed_edges': oriented, 'undirected_edges': set(), 'completed_edges': oriented, 'extra': extra}

def evaluate_order_prior(nodes: list[str], true_edges: set[tuple[str, str]], order: list[str] | None) -> dict[str, float | int | str]:
    if not order:
        return {'llm_order_direct_edge_accuracy': '', 'llm_order_pair_accuracy': '', 'llm_order_dtop': '', 'llm_order_dtop_rate': ''}
    fixed = restrict_order_to_nodes(order, nodes) or nodes
    rank = {node: i for i, node in enumerate(fixed)}
    true_edges_in_nodes = {(src, dst) for src, dst in true_edges if src in rank and dst in rank}
    correct_edges = sum((1 for src, dst in true_edges_in_nodes if rank[src] < rank[dst]))
    ancestor_pairs = transitive_ancestor_pairs(nodes, true_edges_in_nodes)
    violations = sum((1 for src, dst in ancestor_pairs if rank.get(src, 0) >= rank.get(dst, 0)))
    dtop_rate = safe_div(violations, len(ancestor_pairs))
    return {'llm_order_direct_edge_accuracy': round(safe_div(correct_edges, len(true_edges_in_nodes)), 6), 'llm_order_pair_accuracy': round(1.0 - dtop_rate, 6) if ancestor_pairs else '', 'llm_order_dtop': violations if ancestor_pairs else '', 'llm_order_dtop_rate': round(dtop_rate, 6) if ancestor_pairs else ''}

def gammaincc(a: float, x: float) -> float:
    if x <= 0:
        return 1.0
    if a <= 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        ap = a
        summ = 1.0 / a
        delta = summ
        for _ in range(200):
            ap += 1.0
            delta *= x / ap
            summ += delta
            if abs(delta) < abs(summ) * 3e-14:
                break
        p = summ * math.exp(-x + a * math.log(x) - gln)
        return max(0.0, min(1.0, 1.0 - p))
    b = x + 1.0 - a
    c = 1.0 / 1e-300
    d = 1.0 / b
    h = d
    for i in range(1, 200):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    q = math.exp(-x + a * math.log(x) - gln) * h
    return max(0.0, min(1.0, q))

def chi_square_sf(stat: float, df: int) -> float:
    if df <= 0:
        return 1.0
    return gammaincc(df / 2.0, stat / 2.0)

def invert_matrix(matrix: list[list[float]]) -> list[list[float]] | None:
    n = len(matrix)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        aug[col], aug[pivot] = (aug[pivot], aug[col])
        div = aug[col][col]
        aug[col] = [v / div for v in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor:
                aug[row] = [v - factor * aug[col][i] for i, v in enumerate(aug[row])]
    return [row[n:] for row in aug]

def covariance_matrix(columns: list[list[float]]) -> list[list[float]]:
    n = len(columns[0])
    means = [sum(col) / n for col in columns]
    matrix = []
    for i, col_i in enumerate(columns):
        row = []
        mean_i = means[i]
        for j, col_j in enumerate(columns):
            mean_j = means[j]
            cov = sum(((col_i[k] - mean_i) * (col_j[k] - mean_j) for k in range(n))) / max(n - 1, 1)
            row.append(cov)
        matrix.append(row)
    return matrix

def pearson_corr(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx, my = (sum(x) / n, sum(y) / n)
    sx = sum(((v - mx) ** 2 for v in x))
    sy = sum(((v - my) ** 2 for v in y))
    if sx <= EPS or sy <= EPS:
        return 0.0
    cov = sum(((x[i] - mx) * (y[i] - my) for i in range(n)))
    return max(-0.999999, min(0.999999, cov / math.sqrt(sx * sy)))

class CITester:

    def __init__(self, headers: list[str], rows: list[dict[str, str]], data_type: str):
        self.headers = headers
        self.rows = rows
        self.data_type = data_type
        self.cache: dict[tuple[str, str, tuple[str, ...]], tuple[float, float, int]] = {}
        self.categories = {col: sorted({row[col] for row in rows}) for col in headers}
        self.numeric = {col: [float(row[col]) for row in rows] for col in headers} if data_type == 'continuous' else {}

    def test(self, x: str, y: str, cond: tuple[str, ...]) -> tuple[float, float, int]:
        key = (min(x, y), max(x, y), tuple(sorted(cond)))
        if key not in self.cache:
            if self.data_type == 'continuous':
                self.cache[key] = self._fisher_z(x, y, cond)
            else:
                self.cache[key] = self._chi_square(x, y, cond)
        return self.cache[key]

    def _chi_square(self, x: str, y: str, cond: tuple[str, ...]) -> tuple[float, float, int]:
        grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
        if cond:
            for row in self.rows:
                grouped[tuple((row[c] for c in cond))].append(row)
        else:
            grouped[()].extend(self.rows)
        stat = 0.0
        df = 0
        for group_rows in grouped.values():
            x_vals = sorted({row[x] for row in group_rows})
            y_vals = sorted({row[y] for row in group_rows})
            if len(x_vals) < 2 or len(y_vals) < 2:
                continue
            x_index = {v: i for i, v in enumerate(x_vals)}
            y_index = {v: i for i, v in enumerate(y_vals)}
            table = [[0 for _ in y_vals] for _ in x_vals]
            for row in group_rows:
                table[x_index[row[x]]][y_index[row[y]]] += 1
            row_sum = [sum(r) for r in table]
            col_sum = [sum((table[i][j] for i in range(len(x_vals)))) for j in range(len(y_vals))]
            total = sum(row_sum)
            if total == 0:
                continue
            for i in range(len(x_vals)):
                for j in range(len(y_vals)):
                    expected = row_sum[i] * col_sum[j] / total
                    if expected > 0:
                        stat += (table[i][j] - expected) ** 2 / expected
            df += (len(x_vals) - 1) * (len(y_vals) - 1)
        return (chi_square_sf(stat, df), stat, df)

    def _fisher_z(self, x: str, y: str, cond: tuple[str, ...]) -> tuple[float, float, int]:
        n = len(self.rows)
        if not cond:
            rho = pearson_corr(self.numeric[x], self.numeric[y])
        else:
            vars_ = [x, y, *cond]
            cov = covariance_matrix([self.numeric[v] for v in vars_])
            inv = invert_matrix(cov)
            if inv is None:
                for i in range(len(cov)):
                    cov[i][i] += 1e-08
                inv = invert_matrix(cov)
            if inv is None or inv[0][0] <= 0 or inv[1][1] <= 0:
                rho = 0.0
            else:
                rho = -inv[0][1] / math.sqrt(inv[0][0] * inv[1][1])
                rho = max(-0.999999, min(0.999999, rho))
        dof = n - len(cond) - 3
        if dof <= 0:
            return (1.0, 0.0, dof)
        z = 0.5 * math.log((1 + rho) / (1 - rho)) * math.sqrt(dof)
        p_value = math.erfc(abs(z) / math.sqrt(2.0))
        return (p_value, abs(z), dof)

def run_pc(headers: list[str], rows: list[dict[str, str]], data_type: str, alpha: float, max_cond: int) -> dict:
    nodes = headers
    tester = CITester(headers, rows, data_type)
    adj = {n: set(nodes) - {n} for n in nodes}
    sep_sets: dict[frozenset[str], set[str]] = {}
    ci_tests = 0
    for level in range(max_cond + 1):
        level_adj = {n: set(adj[n]) for n in nodes}
        pairs = [(x, y) for i, x in enumerate(nodes) for y in nodes[i + 1:] if y in adj[x]]
        for x, y in pairs:
            if y not in adj[x]:
                continue
            candidate_sets = [level_adj[x] - {y}, level_adj[y] - {x}]
            removed = False
            for candidates in candidate_sets:
                if len(candidates) < level:
                    continue
                for cond in itertools.combinations(sorted(candidates), level):
                    p_value, _, _ = tester.test(x, y, tuple(cond))
                    ci_tests += 1
                    if p_value > alpha:
                        adj[x].discard(y)
                        adj[y].discard(x)
                        sep_sets[frozenset((x, y))] = set(cond)
                        removed = True
                        break
                if removed:
                    break
    directed: set[tuple[str, str]] = set()
    undirected: set[frozenset[str]] = {frozenset((x, y)) for i, x in enumerate(nodes) for y in nodes[i + 1:] if y in adj[x]}

    def is_adjacent(a: str, b: str) -> bool:
        return frozenset((a, b)) in undirected or (a, b) in directed or (b, a) in directed

    def orient(a: str, b: str) -> bool:
        pair = frozenset((a, b))
        if (a, b) in directed:
            return False
        if (b, a) in directed:
            return False
        if has_path(nodes, directed, b, a):
            return False
        if pair not in undirected:
            return False
        undirected.discard(pair)
        directed.add((a, b))
        return True
    for z in nodes:
        neighbors = [n for n in nodes if frozenset((n, z)) in undirected]
        for x, y in itertools.combinations(neighbors, 2):
            if not is_adjacent(x, y) and z not in sep_sets.get(frozenset((x, y)), set()):
                orient(x, z)
                orient(y, z)
    changed = True
    while changed:
        changed = False
        for a, b in list(directed):
            for pair in list(undirected):
                if b not in pair:
                    continue
                c = next(iter(pair - {b}))
                if not is_adjacent(a, c):
                    changed |= orient(b, c)
        for pair in list(undirected):
            a, b = tuple(pair)
            for c in nodes:
                if (a, c) in directed and (c, b) in directed:
                    changed |= orient(a, b)
                    break
                if (b, c) in directed and (c, a) in directed:
                    changed |= orient(b, a)
                    break
        for pair in list(undirected):
            a, b = tuple(pair)
            for head, tail in [(a, b), (b, a)]:
                candidates = [c for c in nodes if frozenset((head, c)) in undirected and (c, tail) in directed]
                for c, d in itertools.combinations(candidates, 2):
                    if not is_adjacent(c, d):
                        changed |= orient(head, tail)
                        break
    completed = complete_cpdag_to_dag(nodes, directed, undirected)
    return {'directed_edges': directed, 'undirected_edges': undirected, 'completed_edges': completed, 'extra': {'ci_tests': ci_tests, 'undirected_edges': len(undirected), 'alpha': alpha, 'max_cond': max_cond}}

class LocalBICScorer:

    def __init__(self, headers: list[str], rows: list[dict[str, str]], data_type: str):
        self.headers = headers
        self.rows = rows
        self.data_type = data_type
        self.n = len(rows)
        self.cache: dict[tuple[str, tuple[str, ...]], float] = {}
        if data_type == 'continuous':
            self.numeric = {col: [float(row[col]) for row in rows] for col in headers}
            self.cov_cache: dict[tuple[str, ...], list[list[float]]] = {}
        else:
            self.categories = {col: sorted({row[col] for row in rows}) for col in headers}
            self.cat_codes = {}
            self.cat_cardinality = {}
            for col in headers:
                mapping = {value: idx for idx, value in enumerate(self.categories[col])}
                self.cat_codes[col] = [mapping[row[col]] for row in rows]
                self.cat_cardinality[col] = len(mapping)

    def score(self, child: str, parents: set[str] | tuple[str, ...]) -> float:
        parents_tuple = tuple(sorted(parents, key=self.headers.index))
        key = (child, parents_tuple)
        if key not in self.cache:
            if self.data_type == 'continuous':
                self.cache[key] = self._continuous_score(child, parents_tuple)
            else:
                self.cache[key] = self._categorical_score(child, parents_tuple)
        return self.cache[key]

    def _categorical_score(self, child: str, parents: tuple[str, ...]) -> float:
        grouped: dict[tuple[int, ...], dict[int, int]] = defaultdict(lambda: defaultdict(int))
        child_codes = self.cat_codes[child]
        parent_codes = [self.cat_codes[parent] for parent in parents]
        if parent_codes:
            for idx, child_value in enumerate(child_codes):
                key = tuple((codes[idx] for codes in parent_codes))
                grouped[key][child_value] += 1
        else:
            counts = grouped[()]
            for child_value in child_codes:
                counts[child_value] += 1
        ll = 0.0
        for counts in grouped.values():
            total = sum(counts.values())
            for count in counts.values():
                if count:
                    ll += count * math.log(count / total)
        param_count = max(self.cat_cardinality[child] - 1, 1)
        for parent in parents:
            param_count *= max(self.cat_cardinality[parent], 1)
        return ll - 0.5 * param_count * math.log(max(self.n, 2))

    def _continuous_score(self, child: str, parents: tuple[str, ...]) -> float:
        y = self.numeric[child]
        var_y = covariance_matrix([y])[0][0]
        if not parents:
            residual_var = max(var_y, EPS)
        else:
            parent_cols = [self.numeric[p] for p in parents]
            cov_pp = covariance_matrix(parent_cols)
            cov_py = covariance_matrix(parent_cols + [y])
            s_py = [cov_py[i][-1] for i in range(len(parents))]
            inv = invert_matrix(cov_pp)
            if inv is None:
                for i in range(len(cov_pp)):
                    cov_pp[i][i] += 1e-08
                inv = invert_matrix(cov_pp)
            if inv is None:
                residual_var = var_y
            else:
                explained = 0.0
                for i in range(len(parents)):
                    for j in range(len(parents)):
                        explained += s_py[i] * inv[i][j] * s_py[j]
                residual_var = max(var_y - explained, EPS)
        params = len(parents) + 2
        return -0.5 * self.n * math.log(residual_var) - 0.5 * params * math.log(max(self.n, 2))

def run_hc_bic(headers: list[str], rows: list[dict[str, str]], data_type: str, max_parents: int, max_iter: int, oracle_order: list[str] | None=None, order_meta: dict | None=None, scorer: LocalBICScorer | None=None) -> dict:
    nodes = headers
    parents = {n: set() for n in nodes}
    edges: set[tuple[str, str]] = set()
    if scorer is None:
        scorer = LocalBICScorer(headers, rows, data_type)
    total_score = sum((scorer.score(node, parents[node]) for node in nodes))
    iterations = 0
    order_rank = {node: i for i, node in enumerate(oracle_order or [])}

    def order_allows(src: str, dst: str) -> bool:
        if oracle_order is None:
            return True
        return order_rank.get(src, 0) < order_rank.get(dst, 0)
    for iteration in range(max_iter):
        best_delta = 1e-09
        best_op = None
        best_new_score = None
        for src in nodes:
            for dst in nodes:
                if src == dst or (src, dst) in edges or (dst, src) in edges:
                    continue
                if not order_allows(src, dst):
                    continue
                if len(parents[dst]) >= max_parents:
                    continue
                if has_path(nodes, edges, dst, src):
                    continue
                old = scorer.score(dst, parents[dst])
                new_parents = set(parents[dst]) | {src}
                new = scorer.score(dst, new_parents)
                delta = new - old
                if delta > best_delta:
                    best_delta = delta
                    best_op = ('add', src, dst)
                    best_new_score = total_score + delta
        for src, dst in sorted(edges):
            old_dst = scorer.score(dst, parents[dst])
            del_parents = set(parents[dst]) - {src}
            new_dst = scorer.score(dst, del_parents)
            delta = new_dst - old_dst
            if delta > best_delta:
                best_delta = delta
                best_op = ('delete', src, dst)
                best_new_score = total_score + delta
            if len(parents[src]) < max_parents and (dst, src) not in edges:
                temp_edges = set(edges)
                temp_edges.remove((src, dst))
                if not has_path(nodes, temp_edges, src, dst):
                    if not order_allows(dst, src):
                        continue
                    old_src = scorer.score(src, parents[src])
                    rev_dst_parents = set(parents[dst]) - {src}
                    rev_src_parents = set(parents[src]) | {dst}
                    new_score = scorer.score(dst, rev_dst_parents) + scorer.score(src, rev_src_parents)
                    old_score = old_dst + old_src
                    delta = new_score - old_score
                    if delta > best_delta:
                        best_delta = delta
                        best_op = ('reverse', src, dst)
                        best_new_score = total_score + delta
        if best_op is None:
            break
        op, src, dst = best_op
        if op == 'add':
            edges.add((src, dst))
            parents[dst].add(src)
        elif op == 'delete':
            edges.remove((src, dst))
            parents[dst].remove(src)
        elif op == 'reverse':
            edges.remove((src, dst))
            parents[dst].remove(src)
            edges.add((dst, src))
            parents[src].add(dst)
        total_score = best_new_score if best_new_score is not None else total_score + best_delta
        iterations = iteration + 1
    if oracle_order and order_meta is None:
        order_extra = {'oracle_order': 'true_topological_order', 'oracle_mode': 'order_constrained_search'}
    else:
        order_extra = dict(order_meta or {})
    return {'directed_edges': edges, 'undirected_edges': set(), 'completed_edges': edges, 'extra': {'score': round(total_score, 6), 'iterations': iterations, 'max_parents': max_parents, **order_extra}}

class AlgorithmSkip(Exception):
    pass

def clone_prediction(prediction: dict) -> dict:
    cloned = {'directed_edges': set(prediction.get('directed_edges', set())), 'undirected_edges': set(prediction.get('undirected_edges', set())), 'completed_edges': set(prediction.get('completed_edges', set())), 'extra': dict(prediction.get('extra', {}))}
    if 'cached_runtime_sec' in prediction:
        cloned['cached_runtime_sec'] = prediction['cached_runtime_sec']
    return cloned

def encode_rows_for_numpy(headers: list[str], rows: list[dict[str, str]], data_type: str, standardize: bool=False):
    if np is None:
        raise AlgorithmSkip('numpy is not available')
    if data_type == 'continuous':
        arr = np.asarray([[float(row[h]) for h in headers] for row in rows], dtype=float)
    else:
        mappings = {h: {v: i for i, v in enumerate(sorted({row[h] for row in rows}))} for h in headers}
        arr = np.asarray([[mappings[h][row[h]] for h in headers] for row in rows], dtype=float)
    if standardize:
        means = arr.mean(axis=0)
        stds = arr.std(axis=0)
        stds[stds == 0] = 1.0
        arr = (arr - means) / stds
    return arr

def parse_causallearn_general_graph(graph, nodes: list[str]) -> tuple[set[tuple[str, str]], set[frozenset[str]]]:
    directed: set[tuple[str, str]] = set()
    undirected: set[frozenset[str]] = set()
    node_set = set(nodes)
    for edge in graph.get_graph_edges():
        n1 = str(edge.get_node1())
        n2 = str(edge.get_node2())
        if n1 not in node_set or n2 not in node_set or n1 == n2:
            continue
        ep1 = str(edge.get_endpoint1())
        ep2 = str(edge.get_endpoint2())
        if ep1 == 'TAIL' and ep2 == 'ARROW':
            directed.add((n1, n2))
        elif ep1 == 'ARROW' and ep2 == 'TAIL':
            directed.add((n2, n1))
        else:
            undirected.add(frozenset((n1, n2)))
    return (directed, undirected)

def run_causallearn_pc(headers: list[str], rows: list[dict[str, str]], data_type: str, alpha: float, max_cond: int) -> dict:
    try:
        from causallearn.search.ConstraintBased.PC import pc as cl_pc
    except Exception as exc:
        raise AlgorithmSkip(f'causal-learn PC unavailable: {exc}') from exc
    arr = encode_rows_for_numpy(headers, rows, data_type, standardize=False)
    indep_test = 'chisq' if data_type == 'categorical' else 'fisherz'
    result = cl_pc(arr, alpha=alpha, indep_test=indep_test, stable=True, verbose=False, show_progress=False, node_names=headers, max_k=max_cond)
    directed, undirected = parse_causallearn_general_graph(result.G, headers)
    completed = complete_cpdag_to_dag(headers, directed, undirected)
    return {'directed_edges': directed, 'undirected_edges': undirected, 'completed_edges': completed, 'extra': {'library': 'causal-learn', 'variant': 'PC-stable', 'indep_test': indep_test, 'alpha': alpha, 'max_cond': max_cond}}

def run_causallearn_ges(headers: list[str], rows: list[dict[str, str]], data_type: str, max_parents: int) -> dict:
    try:
        from causallearn.search.ScoreBased.GES import ges
    except Exception as exc:
        raise AlgorithmSkip(f'causal-learn GES unavailable: {exc}') from exc
    arr = encode_rows_for_numpy(headers, rows, data_type, standardize=False)
    score_func = 'local_score_BDeu' if data_type == 'categorical' else 'local_score_BIC'
    result = ges(arr, score_func=score_func, maxP=max_parents, node_names=headers)
    directed, undirected = parse_causallearn_general_graph(result['G'], headers)
    completed = complete_cpdag_to_dag(headers, directed, undirected)
    return {'directed_edges': directed, 'undirected_edges': undirected, 'completed_edges': completed, 'extra': {'library': 'causal-learn', 'variant': 'GES', 'score_func': score_func, 'score': round(float(result.get('score', 0.0)), 6), 'max_parents': max_parents}}

def run_lingam(headers: list[str], rows: list[dict[str, str]], data_type: str, variant: str, seed: int, threshold: float) -> dict:
    if data_type != 'continuous':
        raise AlgorithmSkip('LiNGAM baselines require continuous data')
    try:
        import lingam
    except Exception as exc:
        raise AlgorithmSkip(f'lingam unavailable: {exc}') from exc
    arr = encode_rows_for_numpy(headers, rows, data_type, standardize=True)
    if variant == 'direct_lingam':
        model = lingam.DirectLiNGAM(random_state=seed)
    elif variant == 'ica_lingam':
        model = lingam.ICALiNGAM(random_state=seed, max_iter=1000)
    else:
        raise ValueError(f'Unknown LiNGAM variant: {variant}')
    model.fit(arr)
    adjacency = model.adjacency_matrix_
    directed = set()
    for target_idx, source_idx in zip(*np.where(np.abs(adjacency) > threshold)):
        source = headers[int(source_idx)]
        target = headers[int(target_idx)]
        if source != target:
            directed.add((source, target))
    return {'directed_edges': directed, 'undirected_edges': set(), 'completed_edges': directed, 'extra': {'library': 'lingam', 'variant': variant, 'lingam_threshold': threshold}}

def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0

def f1_score(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

def evaluate_graph(nodes: list[str], true_edges: set[tuple[str, str]], directed_edges: set[tuple[str, str]], undirected_edges: set[frozenset[str]], completed_edges: set[tuple[str, str]]) -> dict[str, float | int | str]:
    true_skel = skeleton(true_edges)
    pred_skel = skeleton(directed_edges, undirected_edges)
    skel_tp = len(true_skel & pred_skel)
    skel_fp = len(pred_skel - true_skel)
    skel_fn = len(true_skel - pred_skel)
    skel_precision = safe_div(skel_tp, skel_tp + skel_fp)
    skel_recall = safe_div(skel_tp, skel_tp + skel_fn)
    arrow_tp = len(directed_edges & true_edges)
    arrow_fp = len(directed_edges - true_edges)
    arrow_fn = len(true_edges - directed_edges)
    arrow_precision = safe_div(arrow_tp, arrow_tp + arrow_fp)
    arrow_recall = safe_div(arrow_tp, arrow_tp + arrow_fn)
    arrow_f1 = f1_score(arrow_precision, arrow_recall)
    reversed_edges = sum((1 for src, dst in directed_edges if (dst, src) in true_edges))
    shd = 0.0
    node_pairs = [frozenset((a, b)) for i, a in enumerate(nodes) for b in nodes[i + 1:]]
    for pair in node_pairs:
        a, b = tuple(pair)
        true_ab = (a, b) in true_edges
        true_ba = (b, a) in true_edges
        pred_ab = (a, b) in directed_edges
        pred_ba = (b, a) in directed_edges
        pred_und = pair in undirected_edges
        true_present = true_ab or true_ba
        pred_present = pred_ab or pred_ba or pred_und
        if true_present and (not pred_present):
            shd += 1
        elif not true_present and pred_present:
            shd += 1
        elif true_present and pred_present:
            if pred_und:
                shd += 0.5
            elif true_ab and pred_ba or (true_ba and pred_ab):
                shd += 1
    order = topological_sort(nodes, completed_edges)
    cyclic = order is None
    dtop = ''
    dtop_rate = ''
    if order is not None:
        rank = {node: i for i, node in enumerate(order)}
        ancestor_pairs = transitive_ancestor_pairs(nodes, true_edges)
        violations = sum((1 for src, dst in ancestor_pairs if rank.get(src, 0) >= rank.get(dst, 0)))
        dtop = violations
        dtop_rate = round(safe_div(violations, len(ancestor_pairs)), 6)
    return {'true_edges': len(true_edges), 'pred_directed_edges': len(directed_edges), 'pred_undirected_edges': len(undirected_edges), 'pred_skeleton_edges': len(pred_skel), 'shd': round(shd, 6), 'tpr': round(arrow_recall, 6), 'precision': round(arrow_precision, 6), 'f1': round(arrow_f1, 6), 'skeleton_precision': round(skel_precision, 6), 'skeleton_recall': round(skel_recall, 6), 'skeleton_f1': round(f1_score(skel_precision, skel_recall), 6), 'reversed_edges': reversed_edges, 'extra_skeleton_edges': skel_fp, 'missing_skeleton_edges': skel_fn, 'cyclic': str(cyclic).lower(), 'dtop': dtop, 'dtop_rate': dtop_rate}

def save_predicted_edges(out_dir: Path, dataset: str, algorithm: str, sample_size: int, seed: int, directed_edges: set[tuple[str, str]], undirected_edges: set[frozenset[str]]) -> None:
    rows = [{'source': src, 'target': dst, 'edge_type': 'directed'} for src, dst in sorted(directed_edges)]
    for pair in sorted(undirected_edges, key=lambda p: tuple(sorted(p))):
        src, dst = tuple(sorted(pair))
        rows.append({'source': src, 'target': dst, 'edge_type': 'undirected'})
    write_csv_dicts(out_dir / f'{dataset}__n{sample_size}__seed{seed}__{algorithm}__edges.csv', rows, ['source', 'target', 'edge_type'])
METRIC_COLS = ['shd', 'tpr', 'precision', 'f1', 'skeleton_f1', 'dtop_rate', 'runtime_sec', 'bootstrap_edge_confidence', 'bootstrap_direction_confidence', 'bootstrap_skeleton_stability', 'bootstrap_stable_edge_ratio', 'edge_confidence', 'data_direction_confidence', 'llm_pair_confidence', 'adaptive_llm_weight', 'adaptive_data_weight']

def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float('nan')

def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum(((v - m) ** 2 for v in values)) / (len(values) - 1))

def metric_values(rows: list[dict], metric: str) -> list[float]:
    vals = []
    for row in rows:
        value = row.get(metric, '')
        if value == '' or value is None:
            continue
        vals.append(float(value))
    return vals

def aggregate_results(results: list[dict], group_cols: list[str]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in results:
        grouped[tuple((row.get(col, '') for col in group_cols))].append(row)
    aggregates = []
    for key, rows in sorted(grouped.items()):
        out = {col: key[i] for i, col in enumerate(group_cols)}
        out['runs'] = len(rows)
        for metric in METRIC_COLS:
            vals = metric_values(rows, metric)
            if vals:
                out[f'{metric}_mean'] = round(mean(vals), 6)
                out[f'{metric}_std'] = round(stdev(vals), 6)
                out[f'{metric}_mean_std'] = f'{mean(vals):.3f}+/-{stdev(vals):.3f}'
            else:
                out[f'{metric}_mean'] = ''
                out[f'{metric}_std'] = ''
                out[f'{metric}_mean_std'] = ''
        aggregates.append(out)
    return aggregates

def format_markdown_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return ''
    header = '| ' + ' | '.join(columns) + ' |'
    sep = '| ' + ' | '.join(['---'] * len(columns)) + ' |'
    lines = [header, sep]
    for row in rows:
        lines.append('| ' + ' | '.join((str(row.get(col, '')) for col in columns)) + ' |')
    return '\n'.join(lines) + '\n'

def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {'\\': '\\textbackslash{}', '_': '\\_', '%': '\\%', '&': '\\&', '#': '\\#'}
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text

def format_latex_table(rows: list[dict], columns: list[str], caption: str, label: str) -> str:
    if not rows:
        return ''
    align = 'l' * len(columns)
    lines = ['\\begin{table}[htbp]', '\\centering', f'\\caption{{{latex_escape(caption)}}}', f'\\label{{{latex_escape(label)}}}', f'\\begin{{tabular}}{{{align}}}', '\\toprule', ' & '.join((latex_escape(c) for c in columns)) + ' \\\\', '\\midrule']
    for row in rows:
        lines.append(' & '.join((latex_escape(row.get(col, '')) for col in columns)) + ' \\\\')
    lines.extend(['\\bottomrule', '\\end{tabular}', '\\end{table}', ''])
    return '\n'.join(lines)

def save_table_bundle(table_dir: Path, stem: str, rows: list[dict], columns: list[str], caption: str, label: str) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    trimmed_rows = [{col: row.get(col, '') for col in columns} for row in rows]
    write_csv_dicts(table_dir / f'{stem}.csv', trimmed_rows, columns)
    (table_dir / f'{stem}.md').write_text(format_markdown_table(trimmed_rows, columns), encoding='utf-8')
    (table_dir / f'{stem}.tex').write_text(format_latex_table(trimmed_rows, columns, caption, label), encoding='utf-8')
GUIDANCE_METRICS = ['base_f1', 'guided_f1', 'delta_f1', 'base_shd', 'guided_shd', 'shd_reduction', 'base_dtop_rate', 'guided_dtop_rate', 'dtop_rate_reduction', 'llm_order_pair_accuracy', 'llm_order_direct_edge_accuracy', 'oracle_f1', 'oracle_gap_f1']

def to_float_or_none(value: object) -> float | None:
    if value == '' or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def aggregate_guidance_results(rows: list[dict], group_cols: list[str]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple((row.get(col, '') for col in group_cols))].append(row)
    out_rows = []
    for key, group in sorted(grouped.items()):
        out = {col: key[i] for i, col in enumerate(group_cols)}
        out['runs'] = len(group)
        for metric in GUIDANCE_METRICS:
            vals = [value for value in (to_float_or_none(row.get(metric)) for row in group) if value is not None]
            if vals:
                out[f'{metric}_mean'] = round(mean(vals), 6)
                out[f'{metric}_std'] = round(stdev(vals), 6)
                out[f'{metric}_mean_std'] = f'{mean(vals):.3f}+/-{stdev(vals):.3f}'
            else:
                out[f'{metric}_mean'] = ''
                out[f'{metric}_std'] = ''
                out[f'{metric}_mean_std'] = ''
        out_rows.append(out)
    return out_rows

def make_llm_order_improvement_tables(table_dir: Path, results: list[dict]) -> None:
    by_key = {}
    key_cols = ['collection', 'dataset', 'sample_size', 'seed']
    for row in results:
        key = tuple((row.get(col, '') for col in key_cols)) + (row.get('algorithm', ''),)
        by_key[key] = row
    detail = []
    for row in results:
        algorithm = row.get('algorithm', '')
        base_algorithm, llm_suffix = strip_algorithm_suffix(algorithm, LLM_GUIDANCE_SUFFIXES)
        if not llm_suffix:
            continue
        run_key = tuple((row.get(col, '') for col in key_cols))
        base = by_key.get(run_key + (base_algorithm,))
        if base is None:
            continue
        oracle = by_key.get(run_key + (f'{base_algorithm}_oracle_order',))
        base_f1 = to_float_or_none(base.get('f1'))
        guided_f1 = to_float_or_none(row.get('f1'))
        base_shd = to_float_or_none(base.get('shd'))
        guided_shd = to_float_or_none(row.get('shd'))
        base_dtop = to_float_or_none(base.get('dtop_rate'))
        guided_dtop = to_float_or_none(row.get('dtop_rate'))
        oracle_f1 = to_float_or_none((oracle or {}).get('f1'))
        detail.append({'collection': row.get('collection', ''), 'dataset': row.get('dataset', ''), 'sample_size': row.get('sample_size', ''), 'seed': row.get('seed', ''), 'guidance_type': llm_suffix.strip('_'), 'base_algorithm': base_algorithm, 'guided_algorithm': algorithm, 'base_f1': base_f1 if base_f1 is not None else '', 'guided_f1': guided_f1 if guided_f1 is not None else '', 'delta_f1': round(guided_f1 - base_f1, 6) if base_f1 is not None and guided_f1 is not None else '', 'base_shd': base_shd if base_shd is not None else '', 'guided_shd': guided_shd if guided_shd is not None else '', 'shd_reduction': round(base_shd - guided_shd, 6) if base_shd is not None and guided_shd is not None else '', 'base_dtop_rate': base_dtop if base_dtop is not None else '', 'guided_dtop_rate': guided_dtop if guided_dtop is not None else '', 'dtop_rate_reduction': round(base_dtop - guided_dtop, 6) if base_dtop is not None and guided_dtop is not None else '', 'llm_order_pair_accuracy': row.get('llm_order_pair_accuracy', ''), 'llm_order_direct_edge_accuracy': row.get('llm_order_direct_edge_accuracy', ''), 'llm_order_source': row.get('llm_order_source', ''), 'llm_model': row.get('llm_model', ''), 'llm_prompt_mode': row.get('llm_prompt_mode', ''), 'llm_confidence': row.get('llm_confidence', ''), 'llm_data_conflicts': row.get('llm_data_conflicts', ''), 'llm_conflict_llm_wins': row.get('llm_conflict_llm_wins', ''), 'llm_conflict_data_wins': row.get('llm_conflict_data_wins', ''), 'oracle_f1': oracle_f1 if oracle_f1 is not None else '', 'oracle_gap_f1': round(oracle_f1 - guided_f1, 6) if oracle_f1 is not None and guided_f1 is not None else ''})
    if not detail:
        return
    table_dir.mkdir(parents=True, exist_ok=True)
    detail_cols = ['collection', 'dataset', 'sample_size', 'seed', 'guidance_type', 'base_algorithm', 'guided_algorithm', 'base_f1', 'guided_f1', 'delta_f1', 'base_shd', 'guided_shd', 'shd_reduction', 'base_dtop_rate', 'guided_dtop_rate', 'dtop_rate_reduction', 'llm_order_pair_accuracy', 'llm_order_direct_edge_accuracy', 'oracle_f1', 'oracle_gap_f1', 'llm_order_source', 'llm_model', 'llm_prompt_mode', 'llm_confidence', 'llm_data_conflicts', 'llm_conflict_llm_wins', 'llm_conflict_data_wins']
    write_csv_dicts(table_dir / 'table_llm_order_improvement_detail.csv', detail, detail_cols)
    overall = aggregate_guidance_results(detail, ['collection', 'guidance_type', 'base_algorithm', 'guided_algorithm'])
    by_dataset = aggregate_guidance_results(detail, ['collection', 'dataset', 'guidance_type', 'base_algorithm', 'guided_algorithm'])
    overall_cols = ['collection', 'guidance_type', 'base_algorithm', 'guided_algorithm', 'runs', 'base_f1_mean_std', 'guided_f1_mean_std', 'delta_f1_mean_std', 'shd_reduction_mean_std', 'dtop_rate_reduction_mean_std', 'llm_order_pair_accuracy_mean_std', 'oracle_gap_f1_mean_std']
    by_dataset_cols = ['collection', 'dataset', 'guidance_type', 'base_algorithm', 'guided_algorithm', 'runs', 'delta_f1_mean_std', 'shd_reduction_mean_std', 'dtop_rate_reduction_mean_std', 'llm_order_pair_accuracy_mean_std', 'oracle_gap_f1_mean_std']
    save_table_bundle(table_dir, 'table_llm_order_improvement_overall', overall, overall_cols, 'LLM-order improvement over data-driven base algorithms.', 'tab:llm_order_improvement_overall')
    save_table_bundle(table_dir, 'table_llm_order_improvement_by_dataset', by_dataset, by_dataset_cols, 'LLM-order improvement by dataset.', 'tab:llm_order_improvement_by_dataset')
CONTROL_GUIDANCE_METRICS = ['base_f1', 'control_f1', 'delta_f1', 'base_shd', 'control_shd', 'shd_reduction', 'base_dtop_rate', 'control_dtop_rate', 'dtop_rate_reduction', 'control_order_pair_accuracy', 'control_order_direct_edge_accuracy']

def aggregate_control_results(rows: list[dict], group_cols: list[str]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[tuple((row.get(col, '') for col in group_cols))].append(row)
    out_rows = []
    for key, group in sorted(grouped.items()):
        out = {col: key[i] for i, col in enumerate(group_cols)}
        out['runs'] = len(group)
        for metric in CONTROL_GUIDANCE_METRICS:
            vals = [value for value in (to_float_or_none(row.get(metric)) for row in group) if value is not None]
            if vals:
                out[f'{metric}_mean'] = round(mean(vals), 6)
                out[f'{metric}_std'] = round(stdev(vals), 6)
                out[f'{metric}_mean_std'] = f'{mean(vals):.3f}+/-{stdev(vals):.3f}'
            else:
                out[f'{metric}_mean'] = ''
                out[f'{metric}_std'] = ''
                out[f'{metric}_mean_std'] = ''
        out_rows.append(out)
    return out_rows

def make_order_control_tables(table_dir: Path, results: list[dict]) -> None:
    by_key = {}
    key_cols = ['collection', 'dataset', 'sample_size', 'seed']
    for row in results:
        key = tuple((row.get(col, '') for col in key_cols)) + (row.get('algorithm', ''),)
        by_key[key] = row
    detail = []
    for row in results:
        algorithm = row.get('algorithm', '')
        base_algorithm, control_suffix = strip_algorithm_suffix(algorithm, CONTROL_ORDER_SUFFIXES)
        if not control_suffix:
            continue
        run_key = tuple((row.get(col, '') for col in key_cols))
        base = by_key.get(run_key + (base_algorithm,))
        if base is None:
            continue
        base_f1 = to_float_or_none(base.get('f1'))
        control_f1 = to_float_or_none(row.get('f1'))
        base_shd = to_float_or_none(base.get('shd'))
        control_shd = to_float_or_none(row.get('shd'))
        base_dtop = to_float_or_none(base.get('dtop_rate'))
        control_dtop = to_float_or_none(row.get('dtop_rate'))
        detail.append({'collection': row.get('collection', ''), 'dataset': row.get('dataset', ''), 'sample_size': row.get('sample_size', ''), 'seed': row.get('seed', ''), 'control_type': control_suffix.strip('_'), 'base_algorithm': base_algorithm, 'control_algorithm': algorithm, 'base_f1': base_f1 if base_f1 is not None else '', 'control_f1': control_f1 if control_f1 is not None else '', 'delta_f1': round(control_f1 - base_f1, 6) if base_f1 is not None and control_f1 is not None else '', 'base_shd': base_shd if base_shd is not None else '', 'control_shd': control_shd if control_shd is not None else '', 'shd_reduction': round(base_shd - control_shd, 6) if base_shd is not None and control_shd is not None else '', 'base_dtop_rate': base_dtop if base_dtop is not None else '', 'control_dtop_rate': control_dtop if control_dtop is not None else '', 'dtop_rate_reduction': round(base_dtop - control_dtop, 6) if base_dtop is not None and control_dtop is not None else '', 'control_order_pair_accuracy': row.get('control_order_pair_accuracy', ''), 'control_order_direct_edge_accuracy': row.get('control_order_direct_edge_accuracy', '')})
    if not detail:
        return
    table_dir.mkdir(parents=True, exist_ok=True)
    detail_cols = ['collection', 'dataset', 'sample_size', 'seed', 'control_type', 'base_algorithm', 'control_algorithm', 'base_f1', 'control_f1', 'delta_f1', 'base_shd', 'control_shd', 'shd_reduction', 'base_dtop_rate', 'control_dtop_rate', 'dtop_rate_reduction', 'control_order_pair_accuracy', 'control_order_direct_edge_accuracy']
    write_csv_dicts(table_dir / 'table_order_control_detail.csv', detail, detail_cols)
    overall = aggregate_control_results(detail, ['collection', 'control_type', 'base_algorithm', 'control_algorithm'])
    by_dataset = aggregate_control_results(detail, ['collection', 'dataset', 'control_type', 'base_algorithm', 'control_algorithm'])
    overall_cols = ['collection', 'control_type', 'base_algorithm', 'control_algorithm', 'runs', 'base_f1_mean_std', 'control_f1_mean_std', 'delta_f1_mean_std', 'shd_reduction_mean_std', 'dtop_rate_reduction_mean_std', 'control_order_pair_accuracy_mean_std']
    by_dataset_cols = ['collection', 'dataset', 'control_type', 'base_algorithm', 'control_algorithm', 'runs', 'delta_f1_mean_std', 'shd_reduction_mean_std', 'dtop_rate_reduction_mean_std', 'control_order_pair_accuracy_mean_std']
    save_table_bundle(table_dir, 'table_order_control_overall', overall, overall_cols, 'Random and wrong order control results.', 'tab:order_control_overall')
    save_table_bundle(table_dir, 'table_order_control_by_dataset', by_dataset, by_dataset_cols, 'Random and wrong order controls by dataset.', 'tab:order_control_by_dataset')

def make_paper_tables(results_dir: Path, results: list[dict], skipped: list[dict]) -> None:
    table_dir = results_dir / 'paper_tables'
    by_dataset = aggregate_results(results, ['collection', 'dataset', 'sample_size', 'algorithm'])
    overall = aggregate_results(results, ['collection', 'algorithm'])
    sample_curve = aggregate_results(results, ['collection', 'sample_size', 'algorithm'])
    compact_cols = ['collection', 'dataset', 'sample_size', 'algorithm', 'runs', 'shd_mean_std', 'f1_mean_std', 'skeleton_f1_mean_std', 'dtop_rate_mean_std', 'runtime_sec_mean_std']
    overall_cols = ['collection', 'algorithm', 'runs', 'shd_mean_std', 'f1_mean_std', 'skeleton_f1_mean_std', 'dtop_rate_mean_std', 'runtime_sec_mean_std']
    curve_cols = ['collection', 'sample_size', 'algorithm', 'runs', 'shd_mean_std', 'f1_mean_std', 'skeleton_f1_mean_std', 'dtop_rate_mean_std']
    save_table_bundle(table_dir, 'table_by_dataset', by_dataset, compact_cols, 'Baseline results by dataset and sample size.', 'tab:baseline_by_dataset')
    save_table_bundle(table_dir, 'table_overall', overall, overall_cols, 'Overall baseline results.', 'tab:baseline_overall')
    save_table_bundle(table_dir, 'table_sample_size_curve', sample_curve, curve_cols, 'Baseline sample-size sensitivity.', 'tab:baseline_sample_size')
    if skipped:
        skip_cols = sorted({k for row in skipped for k in row.keys()})
        write_csv_dicts(table_dir / 'skipped_runs.csv', skipped, skip_cols)
    make_llm_order_improvement_tables(table_dir, results)
    make_order_control_tables(table_dir, results)

def load_bnlearn_parsed(ds_dir: Path) -> dict:
    nodes = []
    variables = {}
    for row in read_csv_dicts(ds_dir / 'nodes.csv'):
        node = row['node']
        states = row.get('states', '')
        state_values = states.split('|') if states else []
        nodes.append(node)
        variables[node] = {'states': state_values, 'state_count': int(row.get('state_count') or len(state_values))}
    cpts = json.loads((ds_dir / 'cpts.json').read_text(encoding='utf-8'))
    return {'nodes': nodes, 'variables': variables, 'cpts': cpts}

def sample_bnlearn_rows(parsed: dict, n: int, seed: int) -> tuple[list[str], list[dict[str, str]]]:
    rng = random.Random(seed)
    nodes = parsed['nodes']
    cpts = parsed['cpts']
    edges = []
    for child, spec in cpts.items():
        for parent in spec['parents']:
            edges.append((parent, child))
    order = topological_sort(nodes, set(edges))
    if order is None:
        raise ValueError('BNLearn graph contains a cycle')
    rows = []
    for _ in range(n):
        row = {}
        for node in order:
            states = parsed['variables'][node]['states']
            spec = cpts[node]
            key = '|'.join((row[parent] for parent in spec['parents'])) if spec['parents'] else ''
            probs = spec['rows'][key]
            row[node] = rng.choices(states, weights=probs, k=1)[0]
        rows.append({node: row[node] for node in nodes})
    return (nodes, rows)

def synthetic_sem_rows(nodes: list[str], edges: set[tuple[str, str]], n: int, seed: int) -> tuple[list[str], list[dict[str, str]]]:
    rng = random.Random(seed)
    order = topological_sort(nodes, edges)
    if order is None:
        raise ValueError('SEM graph contains a cycle')
    parents = defaultdict(list)
    node_rank = {node: i for i, node in enumerate(nodes)}
    edge_order = sorted(edges, key=lambda e: (node_rank.get(e[0], len(nodes)), node_rank.get(e[1], len(nodes)), e[0], e[1]))
    for src, dst in edge_order:
        parents[dst].append(src)
    for node in parents:
        parents[node].sort(key=lambda parent: (node_rank.get(parent, len(nodes)), parent))
    weights = {(src, dst): rng.uniform(0.35, 1.15) * (-1 if rng.random() < 0.35 else 1) for src, dst in edge_order}
    rows = []
    for _ in range(n):
        row = {}
        for node in order:
            value = rng.gauss(0, 1)
            if parents[node]:
                value += sum((weights[parent, node] * row[parent] for parent in parents[node]))
                value = math.tanh(value / (1 + 0.2 * len(parents[node]))) + rng.gauss(0, 0.35)
            row[node] = value
        rows.append({node: f'{row[node]:.8f}' for node in nodes})
    return (nodes, rows)

def bootstrap_rows(headers: list[str], rows: list[dict[str, str]], n: int, seed: int) -> tuple[list[str], list[dict[str, str]]]:
    rng = random.Random(seed)
    sampled = [dict(rng.choice(rows)) for _ in range(n)]
    return (headers, sampled)

def load_or_generate_rows(collection: str, ds_dir: Path, dataset: str, sample_size: int, seed: int, sample_mode: str) -> tuple[list[str], list[dict[str, str]], str]:
    if sample_mode == 'generated' and collection == 'bnlearn':
        return (*sample_bnlearn_rows(load_bnlearn_parsed(ds_dir), sample_size, seed), 'generated_bn_cpt')
    if sample_mode == 'generated' and collection != 'bnlearn':
        nodes = [row['node'] for row in read_csv_dicts(ds_dir / 'nodes.csv')]
        edges = read_edges(ds_dir / 'edges.csv')
        return (*synthetic_sem_rows(nodes, edges, sample_size, seed), 'generated_sem')
    sample_path = pick_sample_file(ds_dir, dataset, sample_size)
    if sample_path is None:
        raise FileNotFoundError(f'No CSV found for {dataset} with n={sample_size}')
    headers, rows = read_data_csv(sample_path)
    if sample_mode == 'bootstrap':
        return (*bootstrap_rows(headers, rows, sample_size, seed), 'bootstrap_csv')
    return (headers, rows, 'fixed_csv')

def infer_data_type(collection: str, sample_path: Path) -> str:
    if collection == 'bnlearn':
        return 'categorical'
    with sample_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        first = next(reader)
    try:
        for value in first.values():
            float(value)
        return 'continuous'
    except ValueError:
        return 'categorical'

def infer_rows_data_type(collection: str, rows: list[dict[str, str]]) -> str:
    if collection == 'bnlearn':
        return 'categorical'
    first = rows[0]
    try:
        for value in first.values():
            float(value)
        return 'continuous'
    except ValueError:
        return 'categorical'

def run_named_algorithm(algorithm: str, headers: list[str], rows: list[dict[str, str]], data_type: str, args: argparse.Namespace, seed: int, oracle_order: list[str] | None=None, llm_order: list[str] | None=None, llm_prior: dict | None=None, random_order: list[str] | None=None, wrong_order: list[str] | None=None, base_cache: dict[str, dict] | None=None, scorer_cache: dict[str, LocalBICScorer] | None=None) -> dict:
    algorithm = canonical_algorithm_name(algorithm)

    def cached_hc_scorer() -> LocalBICScorer | None:
        if scorer_cache is None:
            return None
        if 'hc_bic_lite' not in scorer_cache:
            scorer_cache['hc_bic_lite'] = LocalBICScorer(headers, rows, data_type)
        return scorer_cache['hc_bic_lite']

    def cached_base(base_algorithm: str) -> dict:
        if base_cache is None:
            return run_named_algorithm(base_algorithm, headers, rows, data_type, args, seed, scorer_cache=scorer_cache)
        computed_in_call = False
        if base_algorithm not in base_cache:
            base_start = time.perf_counter()
            base_pred = run_named_algorithm(base_algorithm, headers, rows, data_type, args, seed, scorer_cache=scorer_cache)
            base_pred['cached_runtime_sec'] = round(time.perf_counter() - base_start, 4)
            base_cache[base_algorithm] = base_pred
            computed_in_call = True
        cached = clone_prediction(base_cache[base_algorithm])
        cached.setdefault('extra', {})
        cached['extra']['base_runtime_sec'] = cached.get('cached_runtime_sec', '')
        cached['extra']['base_runtime_included_in_call'] = str(computed_in_call).lower()
        return cached
    if algorithm == 'pc_lite':
        return run_pc(headers, rows, data_type, args.alpha, args.max_cond)
    if algorithm == 'pc_lite_oracle_order':
        base = cached_base('pc_lite')
        return apply_oracle_order_to_prediction(headers, base, oracle_order, 'pc_lite')
    if algorithm == 'pc_lite_random_order':
        base = cached_base('pc_lite')
        return apply_named_order_to_prediction(headers, base, random_order, 'pc_lite', 'random_order')
    if algorithm == 'pc_lite_wrong_order':
        base = cached_base('pc_lite')
        return apply_named_order_to_prediction(headers, base, wrong_order, 'pc_lite', 'wrong_order')
    if algorithm == 'pc_lite_llm_order':
        base = cached_base('pc_lite')
        return apply_llm_order_to_prediction(headers, base, llm_order, 'pc_lite', llm_prior)
    if algorithm == 'pc_lite_llm_soft_order':
        base = cached_base('pc_lite')
        return apply_soft_llm_prior_to_prediction(headers, base, llm_order, 'pc_lite', llm_prior, args, False)
    if algorithm == 'pc_lite_llm_conflict_order':
        base = cached_base('pc_lite')
        return apply_soft_llm_prior_to_prediction(headers, base, llm_order, 'pc_lite', llm_prior, args, True)
    if algorithm == 'pc_lite_llm_adaptive_order':
        base = cached_base('pc_lite')
        return apply_adaptive_llm_prior_to_prediction(headers, base, llm_order, 'pc_lite', llm_prior, args)
    if algorithm == 'pc_lite_llm_consensus_order':
        base = cached_base('pc_lite')
        return apply_llm_order_to_prediction(headers, base, llm_order, 'pc_lite', llm_prior)
    if algorithm == 'pc_lite_llm_consensus_soft_order':
        base = cached_base('pc_lite')
        return apply_soft_llm_prior_to_prediction(headers, base, llm_order, 'pc_lite', llm_prior, args, False)
    if algorithm == 'pc_lite_llm_consensus_conflict_order':
        base = cached_base('pc_lite')
        return apply_soft_llm_prior_to_prediction(headers, base, llm_order, 'pc_lite', llm_prior, args, True)
    if algorithm == 'hc_bic_lite':
        return run_hc_bic(headers, rows, data_type, args.max_parents, args.max_iter, scorer=cached_hc_scorer())
    if algorithm == 'hc_bic_lite_oracle_order':
        return run_hc_bic(headers, rows, data_type, args.max_parents, args.max_iter, oracle_order=oracle_order, scorer=cached_hc_scorer())
    if algorithm == 'hc_bic_lite_random_order':
        return run_hc_bic(headers, rows, data_type, args.max_parents, args.max_iter, oracle_order=random_order, order_meta={'order_prior': 'random_order', 'order_mode': 'order_constrained_search', 'base_algorithm': 'hc_bic_lite'}, scorer=cached_hc_scorer())
    if algorithm == 'hc_bic_lite_wrong_order':
        if wrong_order is None:
            raise AlgorithmSkip('wrong order prior is unavailable')
        return run_hc_bic(headers, rows, data_type, args.max_parents, args.max_iter, oracle_order=wrong_order, order_meta={'order_prior': 'wrong_order', 'order_mode': 'order_constrained_search', 'base_algorithm': 'hc_bic_lite'}, scorer=cached_hc_scorer())
    if algorithm == 'hc_bic_lite_llm_order':
        if llm_order is None:
            raise AlgorithmSkip('LLM order prior is unavailable')
        return run_hc_bic(headers, rows, data_type, args.max_parents, args.max_iter, oracle_order=llm_order, order_meta={'llm_order': 'causal_order_prior', 'llm_mode': 'order_constrained_search', 'llm_order_source': (llm_prior or {}).get('source', ''), 'llm_model': (llm_prior or {}).get('model', ''), 'llm_prompt_mode': (llm_prior or {}).get('prompt_mode', ''), 'llm_confidence': (llm_prior or {}).get('confidence', ''), 'base_algorithm': 'hc_bic_lite'}, scorer=cached_hc_scorer())
    if algorithm == 'hc_bic_lite_llm_soft_order':
        base = cached_base('hc_bic_lite')
        return apply_soft_llm_prior_to_prediction(headers, base, llm_order, 'hc_bic_lite', llm_prior, args, False)
    if algorithm == 'hc_bic_lite_llm_conflict_order':
        base = cached_base('hc_bic_lite')
        return apply_soft_llm_prior_to_prediction(headers, base, llm_order, 'hc_bic_lite', llm_prior, args, True)
    if algorithm == 'hc_bic_lite_llm_adaptive_order':
        base = cached_base('hc_bic_lite')
        return apply_adaptive_llm_prior_to_prediction(headers, base, llm_order, 'hc_bic_lite', llm_prior, args)
    if algorithm == 'hc_bic_lite_llm_consensus_order':
        if llm_order is None:
            raise AlgorithmSkip('LLM consensus order prior is unavailable')
        return run_hc_bic(headers, rows, data_type, args.max_parents, args.max_iter, oracle_order=llm_order, order_meta={'llm_order': 'consensus_causal_order_prior', 'llm_mode': 'self_consistency_order_constrained_search', 'llm_order_source': (llm_prior or {}).get('source', ''), 'llm_model': (llm_prior or {}).get('model', ''), 'llm_prompt_mode': (llm_prior or {}).get('prompt_mode', ''), 'llm_confidence': (llm_prior or {}).get('confidence', ''), 'base_algorithm': 'hc_bic_lite'}, scorer=cached_hc_scorer())
    if algorithm == 'hc_bic_lite_llm_consensus_soft_order':
        base = cached_base('hc_bic_lite')
        return apply_soft_llm_prior_to_prediction(headers, base, llm_order, 'hc_bic_lite', llm_prior, args, False)
    if algorithm == 'hc_bic_lite_llm_consensus_conflict_order':
        base = cached_base('hc_bic_lite')
        return apply_soft_llm_prior_to_prediction(headers, base, llm_order, 'hc_bic_lite', llm_prior, args, True)
    if algorithm == 'pc_cl':
        return run_causallearn_pc(headers, rows, data_type, args.alpha, args.max_cond)
    if algorithm == 'pc_cl_oracle_order':
        base = cached_base('pc_cl')
        return apply_oracle_order_to_prediction(headers, base, oracle_order, 'pc_cl')
    if algorithm == 'pc_cl_random_order':
        base = cached_base('pc_cl')
        return apply_named_order_to_prediction(headers, base, random_order, 'pc_cl', 'random_order')
    if algorithm == 'pc_cl_wrong_order':
        base = cached_base('pc_cl')
        return apply_named_order_to_prediction(headers, base, wrong_order, 'pc_cl', 'wrong_order')
    if algorithm == 'pc_cl_llm_order':
        base = cached_base('pc_cl')
        return apply_llm_order_to_prediction(headers, base, llm_order, 'pc_cl', llm_prior)
    if algorithm == 'pc_cl_llm_soft_order':
        base = cached_base('pc_cl')
        return apply_soft_llm_prior_to_prediction(headers, base, llm_order, 'pc_cl', llm_prior, args, False)
    if algorithm == 'pc_cl_llm_conflict_order':
        base = cached_base('pc_cl')
        return apply_soft_llm_prior_to_prediction(headers, base, llm_order, 'pc_cl', llm_prior, args, True)
    if algorithm == 'pc_cl_llm_adaptive_order':
        base = cached_base('pc_cl')
        return apply_adaptive_llm_prior_to_prediction(headers, base, llm_order, 'pc_cl', llm_prior, args)
    if algorithm == 'pc_cl_llm_consensus_order':
        base = cached_base('pc_cl')
        return apply_llm_order_to_prediction(headers, base, llm_order, 'pc_cl', llm_prior)
    if algorithm == 'pc_cl_llm_consensus_soft_order':
        base = cached_base('pc_cl')
        return apply_soft_llm_prior_to_prediction(headers, base, llm_order, 'pc_cl', llm_prior, args, False)
    if algorithm == 'pc_cl_llm_consensus_conflict_order':
        base = cached_base('pc_cl')
        return apply_soft_llm_prior_to_prediction(headers, base, llm_order, 'pc_cl', llm_prior, args, True)
    if algorithm == 'ges_cl':
        return run_causallearn_ges(headers, rows, data_type, args.max_parents)
    if algorithm == 'ges_cl_oracle_order':
        base = cached_base('ges_cl')
        return apply_oracle_order_to_prediction(headers, base, oracle_order, 'ges_cl')
    if algorithm == 'ges_cl_random_order':
        base = cached_base('ges_cl')
        return apply_named_order_to_prediction(headers, base, random_order, 'ges_cl', 'random_order')
    if algorithm == 'ges_cl_wrong_order':
        base = cached_base('ges_cl')
        return apply_named_order_to_prediction(headers, base, wrong_order, 'ges_cl', 'wrong_order')
    if algorithm == 'ges_cl_llm_order':
        base = cached_base('ges_cl')
        return apply_llm_order_to_prediction(headers, base, llm_order, 'ges_cl', llm_prior)
    if algorithm == 'ges_cl_llm_soft_order':
        base = cached_base('ges_cl')
        return apply_soft_llm_prior_to_prediction(headers, base, llm_order, 'ges_cl', llm_prior, args, False)
    if algorithm == 'ges_cl_llm_conflict_order':
        base = cached_base('ges_cl')
        return apply_soft_llm_prior_to_prediction(headers, base, llm_order, 'ges_cl', llm_prior, args, True)
    if algorithm == 'ges_cl_llm_adaptive_order':
        base = cached_base('ges_cl')
        return apply_adaptive_llm_prior_to_prediction(headers, base, llm_order, 'ges_cl', llm_prior, args)
    if algorithm == 'ges_cl_llm_consensus_order':
        base = cached_base('ges_cl')
        return apply_llm_order_to_prediction(headers, base, llm_order, 'ges_cl', llm_prior)
    if algorithm == 'ges_cl_llm_consensus_soft_order':
        base = cached_base('ges_cl')
        return apply_soft_llm_prior_to_prediction(headers, base, llm_order, 'ges_cl', llm_prior, args, False)
    if algorithm == 'ges_cl_llm_consensus_conflict_order':
        base = cached_base('ges_cl')
        return apply_soft_llm_prior_to_prediction(headers, base, llm_order, 'ges_cl', llm_prior, args, True)
    if algorithm in {'direct_lingam', 'ica_lingam'}:
        return run_lingam(headers, rows, data_type, algorithm, seed, args.lingam_threshold)
    raise ValueError(f'Unknown algorithm: {algorithm}')

def run_experiments(args: argparse.Namespace) -> list[dict]:
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    graph_out_dir = results_dir / 'baseline_graphs'
    results_dir.mkdir(parents=True, exist_ok=True)
    graph_out_dir.mkdir(parents=True, exist_ok=True)
    index_rows = read_csv_dicts(data_dir / 'dataset_index.csv')
    collections = parse_csv_list(args.collections)
    algorithms = []
    seen_algorithms = set()
    for algorithm in parse_ordered_csv_list(args.algorithms):
        canonical = canonical_algorithm_name(algorithm)
        if canonical not in seen_algorithms:
            seen_algorithms.add(canonical)
            algorithms.append(canonical)
    sample_sizes = parse_int_list(args.sample_sizes)
    seeds = parse_int_list(args.seeds)
    selected_datasets = parse_csv_list(args.datasets) if args.datasets else None
    need_llm_order = any((algorithm_uses_llm_order(algorithm) for algorithm in algorithms))
    results = []
    skipped = []
    for ds in index_rows:
        if ds['collection'] not in collections:
            continue
        if selected_datasets and ds['dataset'] not in selected_datasets:
            continue
        ds_dir = data_dir / ds['path']
        true_edges = read_edges(ds_dir / 'edges.csv')
        llm_prior = None
        if need_llm_order:
            prior_nodes = [row['node'] for row in load_node_metadata(ds_dir) if row.get('node')]
            if not prior_nodes:
                prior_nodes = sorted({node for edge in true_edges for node in edge})
            try:
                llm_prior = load_or_create_llm_order_prior(ds['collection'], ds['dataset'], ds_dir, prior_nodes, args)
                if llm_prior:
                    print(f"[llm] {ds['collection']}/{ds['dataset']} source={llm_prior.get('source', '')} model={llm_prior.get('model', '')}")
                else:
                    print(f"[llm-skip] {ds['collection']}/{ds['dataset']}: no LLM prior available")
            except Exception as exc:
                print(f"[llm-error] {ds['collection']}/{ds['dataset']}: {type(exc).__name__}: {exc}")
        for sample_size in sample_sizes:
            for seed in seeds:
                try:
                    headers, rows, sample_source = load_or_generate_rows(ds['collection'], ds_dir, ds['dataset'], sample_size, seed, args.sample_mode)
                except Exception as exc:
                    skipped.append({'collection': ds['collection'], 'dataset': ds['dataset'], 'sample_size': sample_size, 'seed': seed, 'algorithm': '*', 'reason': f'data unavailable: {exc}'})
                    print(f"[skip] {ds['collection']}/{ds['dataset']} n={sample_size} seed={seed}: {exc}")
                    continue
                run_true_edges = {(a, b) for a, b in true_edges if a in headers and b in headers}
                oracle_order = oracle_order_from_edges(headers, run_true_edges)
                llm_order = restrict_order_to_nodes((llm_prior or {}).get('ordered_nodes'), headers)
                llm_order_metrics = evaluate_order_prior(headers, run_true_edges, llm_order)
                random_order = make_random_order(headers, ds['collection'], ds['dataset'], sample_size, seed, args.order_seed)
                wrong_order = make_wrong_order(headers, oracle_order)
                random_order_metrics = evaluate_order_prior(headers, run_true_edges, random_order)
                wrong_order_metrics = evaluate_order_prior(headers, run_true_edges, wrong_order)
                data_type = infer_rows_data_type(ds['collection'], rows)
                print(f"[run] {ds['collection']}/{ds['dataset']} n={len(rows)} seed={seed} type={data_type} source={sample_source}")
                base_prediction_cache: dict[str, dict] = {}
                scorer_cache: dict[str, LocalBICScorer] = {}
                for algorithm in algorithms:
                    canonical_algorithm = canonical_algorithm_name(algorithm)
                    start = time.perf_counter()
                    try:
                        pred = run_named_algorithm(canonical_algorithm, headers, rows, data_type, args, seed, oracle_order=oracle_order, llm_order=llm_order, llm_prior=llm_prior, random_order=random_order, wrong_order=wrong_order, base_cache=base_prediction_cache, scorer_cache=scorer_cache)
                    except AlgorithmSkip as exc:
                        skipped.append({'collection': ds['collection'], 'dataset': ds['dataset'], 'sample_size': sample_size, 'seed': seed, 'algorithm': canonical_algorithm, 'reason': str(exc)})
                        print(f'      [skip] {canonical_algorithm}: {exc}')
                        continue
                    except Exception as exc:
                        skipped.append({'collection': ds['collection'], 'dataset': ds['dataset'], 'sample_size': sample_size, 'seed': seed, 'algorithm': canonical_algorithm, 'reason': f'error: {type(exc).__name__}: {exc}'})
                        print(f'      [error] {canonical_algorithm}: {type(exc).__name__}: {exc}')
                        continue
                    runtime = time.perf_counter() - start
                    if canonical_algorithm in {'pc_lite', 'hc_bic_lite', 'pc_cl', 'ges_cl'}:
                        cached_pred = clone_prediction(pred)
                        cached_pred['cached_runtime_sec'] = round(runtime, 4)
                        base_prediction_cache.setdefault(canonical_algorithm, cached_pred)
                    reported_runtime = runtime
                    base_runtime = to_float_or_none(pred.get('extra', {}).get('base_runtime_sec', ''))
                    if base_runtime is not None and pred.get('extra', {}).get('base_runtime_included_in_call') != 'true':
                        reported_runtime += base_runtime
                        pred['extra']['orientation_runtime_sec'] = round(runtime, 4)
                    metrics = evaluate_graph(headers, run_true_edges, pred['directed_edges'], pred['undirected_edges'], pred['completed_edges'])
                    save_predicted_edges(graph_out_dir, ds['dataset'], canonical_algorithm, sample_size, seed, pred['directed_edges'], pred['undirected_edges'])
                    result = {'collection': ds['collection'], 'dataset': ds['dataset'], 'algorithm': canonical_algorithm, 'sample_size': len(rows), 'seed': seed, 'sample_source': sample_source, 'data_type': data_type, 'runtime_sec': round(reported_runtime, 4), **metrics, **pred['extra']}
                    if algorithm_uses_llm_order(canonical_algorithm):
                        result.update({**llm_order_metrics, 'llm_order_cache_path': (llm_prior or {}).get('cache_path', '')})
                    if algorithm_uses_control_order(canonical_algorithm):
                        order_type = 'random_order' if algorithm_uses_random_order(canonical_algorithm) else 'wrong_order'
                        control_metrics = random_order_metrics if order_type == 'random_order' else wrong_order_metrics
                        result.update({'control_order_type': order_type, 'control_order_direct_edge_accuracy': control_metrics['llm_order_direct_edge_accuracy'], 'control_order_pair_accuracy': control_metrics['llm_order_pair_accuracy'], 'control_order_dtop': control_metrics['llm_order_dtop'], 'control_order_dtop_rate': control_metrics['llm_order_dtop_rate']})
                    results.append(result)
                    print(f"      {canonical_algorithm}: SHD={result['shd']} TPR={result['tpr']} F1={result['f1']} SkelF1={result['skeleton_f1']} DtopRate={result['dtop_rate']}")
    if not results:
        if skipped:
            write_csv_dicts(results_dir / 'baseline_skipped.csv', skipped, sorted({k for row in skipped for k in row.keys()}))
        return []
    fieldnames = sorted({k for row in results for k in row.keys()})
    preferred = ['collection', 'dataset', 'algorithm', 'sample_size', 'seed', 'sample_source', 'data_type', 'runtime_sec', 'shd', 'tpr', 'precision', 'f1', 'skeleton_precision', 'skeleton_recall', 'skeleton_f1', 'dtop', 'dtop_rate', 'cyclic', 'true_edges', 'pred_directed_edges', 'pred_undirected_edges', 'pred_skeleton_edges', 'reversed_edges', 'extra_skeleton_edges', 'missing_skeleton_edges', 'oracle_order', 'oracle_mode', 'llm_order', 'llm_mode', 'llm_order_source', 'llm_model', 'llm_prompt_mode', 'llm_confidence', 'llm_order_direct_edge_accuracy', 'llm_order_pair_accuracy', 'llm_order_dtop', 'llm_order_dtop_rate', 'llm_order_cache_path', 'llm_soft_weight', 'llm_conflict_weight', 'llm_data_conflicts', 'llm_conflict_llm_wins', 'llm_conflict_data_wins', 'adaptive_llm_weight', 'adaptive_data_weight', 'adaptive_conflict_llm_weight', 'adaptive_llm_strength', 'adaptive_data_strength', 'adaptive_source_weight', 'adaptive_raw_confidence', 'control_order_type', 'control_order_direct_edge_accuracy', 'control_order_pair_accuracy', 'control_order_dtop', 'control_order_dtop_rate', 'order_prior', 'order_mode', 'base_algorithm']
    ordered = [f for f in preferred if f in fieldnames] + [f for f in fieldnames if f not in preferred]
    write_csv_dicts(results_dir / 'baseline_results.csv', results, ordered)
    aggregate_rows = aggregate_results(results, ['collection', 'dataset', 'sample_size', 'algorithm'])
    aggregate_fields = sorted({k for row in aggregate_rows for k in row.keys()})
    aggregate_preferred = ['collection', 'dataset', 'sample_size', 'algorithm', 'runs']
    aggregate_ordered = [f for f in aggregate_preferred if f in aggregate_fields] + [f for f in aggregate_fields if f not in aggregate_preferred]
    write_csv_dicts(results_dir / 'baseline_aggregate.csv', aggregate_rows, aggregate_ordered)
    summary_rows = aggregate_results(results, ['collection', 'algorithm'])
    summary_fields = sorted({k for row in summary_rows for k in row.keys()})
    summary_preferred = ['collection', 'algorithm', 'runs']
    summary_ordered = [f for f in summary_preferred if f in summary_fields] + [f for f in summary_fields if f not in summary_preferred]
    write_csv_dicts(results_dir / 'baseline_summary.csv', summary_rows, summary_ordered)
    if skipped:
        write_csv_dicts(results_dir / 'baseline_skipped.csv', skipped, sorted({k for row in skipped for k in row.keys()}))
    make_paper_tables(results_dir, results, skipped)
    metadata = {'sample_sizes': sample_sizes, 'seeds': seeds, 'sample_mode': args.sample_mode, 'collections': sorted(collections), 'algorithms': algorithms, 'alpha': args.alpha, 'max_cond': args.max_cond, 'max_parents': args.max_parents, 'max_iter': args.max_iter, 'lingam_threshold': args.lingam_threshold, 'order_seed': args.order_seed, 'llm_model': args.llm_model, 'llm_prompt_mode': args.llm_prompt_mode, 'llm_prior_dir': args.llm_prior_dir, 'llm_temperature': args.llm_temperature, 'llm_max_tokens': args.llm_max_tokens, 'llm_cache_only': args.llm_cache_only, 'llm_fallback': args.llm_fallback, 'llm_order_default_confidence': args.llm_order_default_confidence, 'llm_soft_weight': args.llm_soft_weight, 'llm_conflict_weight': args.llm_conflict_weight, 'llm_conflict_trust_threshold': args.llm_conflict_trust_threshold, 'data_directed_confidence': args.data_directed_confidence, 'notes': {'pc_cl': 'causal-learn stable PC with chi-square CI for categorical data and Fisher-z CI for continuous data.', 'ges_cl': 'causal-learn GES with BDeu for categorical data and BIC for continuous data.', 'pc_cl_oracle_order': 'Run causal-learn PC, keep the predicted skeleton, and orient every adjacency by the true topological order.', 'ges_cl_oracle_order': 'Run causal-learn GES, keep the predicted skeleton, and orient every adjacency by the true topological order.', 'pc_cl_llm_order': 'Run causal-learn PC, keep the predicted skeleton, and orient every adjacency by the cached LLM causal order prior.', 'ges_cl_llm_order': 'Run causal-learn GES, keep the predicted skeleton, and orient every adjacency by the cached LLM causal order prior.', 'pc_cl_llm_soft_order': 'Run causal-learn PC and orient the predicted skeleton by a weighted combination of data direction and pairwise LLM order probabilities.', 'ges_cl_llm_soft_order': 'Run causal-learn GES and orient the predicted skeleton by a weighted combination of data direction and pairwise LLM order probabilities.', 'hc_bic_lite_llm_soft_order': 'Run hc_bic_lite and orient the predicted skeleton by soft LLM order probabilities.', 'pc_cl_llm_conflict_order': 'Soft LLM order fusion with reduced LLM weight under low-confidence LLM/data direction conflicts.', 'ges_cl_llm_conflict_order': 'Soft LLM order fusion with reduced LLM weight under low-confidence LLM/data direction conflicts.', 'hc_bic_lite_llm_conflict_order': 'Soft LLM order fusion with reduced LLM weight under low-confidence LLM/data direction conflicts.', 'pc_cl_llm_adaptive_order': 'Reliability-calibrated LLM/data order fusion with pairwise adaptive LLM weights.', 'ges_cl_llm_adaptive_order': 'Reliability-calibrated LLM/data order fusion with pairwise adaptive LLM weights.', 'hc_bic_lite_llm_adaptive_order': 'Reliability-calibrated LLM/data order fusion with pairwise adaptive LLM weights.', 'pc_cl_llm_consensus_order': 'Self-consistency LLM order prior: orient the PC skeleton by an aggregated consensus causal order.', 'ges_cl_llm_consensus_order': 'Self-consistency LLM order prior: orient the GES skeleton by an aggregated consensus causal order.', 'hc_bic_lite_llm_consensus_order': 'Self-consistency LLM order prior: run order-constrained HC-BIC with an aggregated consensus causal order.', 'pc_cl_llm_consensus_soft_order': 'Self-consistency LLM prior: soft fusion using consensus pairwise order probabilities.', 'ges_cl_llm_consensus_soft_order': 'Self-consistency LLM prior: soft fusion using consensus pairwise order probabilities.', 'hc_bic_lite_llm_consensus_soft_order': 'Self-consistency LLM prior: soft fusion using consensus pairwise order probabilities.', 'pc_cl_llm_consensus_conflict_order': 'Self-consistency LLM prior: conflict-aware fusion using consensus pairwise order probabilities.', 'ges_cl_llm_consensus_conflict_order': 'Self-consistency LLM prior: conflict-aware fusion using consensus pairwise order probabilities.', 'hc_bic_lite_llm_consensus_conflict_order': 'Self-consistency LLM prior: conflict-aware fusion using consensus pairwise order probabilities.', 'pc_cl_random_order': 'Negative control: orient the PC skeleton with a deterministic random order.', 'ges_cl_random_order': 'Negative control: orient the GES skeleton with a deterministic random order.', 'hc_bic_lite_random_order': 'Negative control: run order-constrained HC-BIC using a deterministic random order.', 'pc_cl_wrong_order': 'Adversarial control: orient the PC skeleton by the reversed true topological order.', 'ges_cl_wrong_order': 'Adversarial control: orient the GES skeleton by the reversed true topological order.', 'hc_bic_lite_wrong_order': 'Adversarial control: run order-constrained HC-BIC using the reversed true topological order.', 'direct_lingam': 'lingam DirectLiNGAM; continuous data only.', 'ica_lingam': 'lingam ICALiNGAM; continuous data only.', 'pc_lite': 'Dependency-light PC implementation kept as a fallback and implementation sanity check.', 'hc_bic_lite': 'Dependency-light greedy hill-climbing BIC baseline kept as a fallback.', 'pc_lite_oracle_order': 'Run pc_lite, keep the predicted skeleton, and orient every adjacency by the true topological order.', 'hc_bic_lite_oracle_order': 'Run order-constrained hc_bic_lite using the true topological order.', 'pc_lite_llm_order': 'Run pc_lite, keep the predicted skeleton, and orient every adjacency by the cached LLM causal order prior.', 'hc_bic_lite_llm_order': 'Run order-constrained hc_bic_lite using the cached LLM causal order prior.', 'SHD': 'Counts missing/extra/reversed directed edges; correctly adjacent but unoriented PC edges contribute 0.5.', 'Dtop': 'Number/rate of true ancestor-order violations in a deterministic DAG completion of the prediction.', 'LLMOrder': 'The LLM is used as a noisy semantic expert that supplies an order prior, not a final graph.', 'PromptPerturbation': 'Use --llm-prompt-mode to run full, names_only, shuffled_descriptions, masked_descriptions, or generic_descriptions robustness experiments.'}}
    (results_dir / 'baseline_run_config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return results

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run traditional data-driven causal discovery baselines on prepared public datasets.')
    parser.add_argument('--data-dir', default=str(DEFAULT_DATA_DIR), help='Prepared data directory.')
    parser.add_argument('--results-dir', default=str(DEFAULT_RESULTS_DIR), help='Directory for CSV outputs.')
    parser.add_argument('--collections', default='bnlearn,llm_causal_order_graphs', help='Comma-separated collections from dataset_index.csv.')
    parser.add_argument('--datasets', default='', help='Optional comma-separated dataset names.')
    parser.add_argument('--sample-sizes', default='250,1000', help='Comma-separated sample sizes. For repeated experiments, each seed regenerates/resamples data.')
    parser.add_argument('--seeds', default='0,1', help='Comma-separated random seeds.')
    parser.add_argument('--sample-mode', choices=['generated', 'bootstrap', 'fixed'], default='generated', help='generated: regenerate from BN CPTs or graph SEM; bootstrap: bootstrap from prepared CSV; fixed: use the prepared CSV once per seed.')
    parser.add_argument('--algorithms', default='pc_cl,ges_cl,direct_lingam,ica_lingam,pc_lite,hc_bic_lite', help='Comma-separated algorithms: pc_cl,ges_cl,direct_lingam,ica_lingam,pc_lite,hc_bic_lite,pc_cl_oracle_order,ges_cl_oracle_order,pc_lite_oracle_order,hc_bic_lite_oracle_order,pc_cl_llm_order,ges_cl_llm_order,pc_lite_llm_order,hc_bic_lite_llm_order,pc_cl_llm_soft_order,ges_cl_llm_soft_order,hc_bic_lite_llm_soft_order,pc_cl_llm_conflict_order,ges_cl_llm_conflict_order,hc_bic_lite_llm_conflict_order,pc_cl_llm_adaptive_order,ges_cl_llm_adaptive_order,hc_bic_lite_llm_adaptive_order,pc_cl_llm_consensus_order,ges_cl_llm_consensus_order,hc_bic_lite_llm_consensus_order,pc_cl_llm_consensus_soft_order,ges_cl_llm_consensus_soft_order,hc_bic_lite_llm_consensus_soft_order,pc_cl_llm_consensus_conflict_order,ges_cl_llm_consensus_conflict_order,hc_bic_lite_llm_consensus_conflict_order,pc_cl_random_order,ges_cl_random_order,hc_bic_lite_random_order,pc_cl_wrong_order,ges_cl_wrong_order,hc_bic_lite_wrong_order. Legacy aliases pc,hc_bic map to lite versions.')
    parser.add_argument('--alpha', type=float, default=0.05, help='PC independence-test alpha.')
    parser.add_argument('--max-cond', type=int, default=2, help='PC maximum conditioning-set size.')
    parser.add_argument('--max-parents', type=int, default=3, help='HC-BIC maximum parents per node.')
    parser.add_argument('--max-iter', type=int, default=100, help='HC-BIC maximum greedy iterations.')
    parser.add_argument('--lingam-threshold', type=float, default=0.05, help='Absolute coefficient threshold for LiNGAM edges.')
    parser.add_argument('--order-seed', type=int, default=777, help='Extra seed for RandomOrder controls.')
    parser.add_argument('--llm-prior-dir', default=str(DEFAULT_LLM_PRIOR_DIR), help='Directory for cached LLM order priors.')
    parser.add_argument('--llm-model', default='glm-4.5-flash', help='Zhipu GLM model used to create LLM order priors.')
    parser.add_argument('--llm-prompt-mode', choices=['full', 'names_only', 'state_only', 'shuffled_descriptions', 'masked_descriptions', 'generic_descriptions'], default='full', help='Variable-description perturbation used when prompting the LLM.')
    parser.add_argument('--llm-api-key-env', default='ZAI_API_KEY,ZHIPUAI_API_KEY,GLM_API_KEY', help='Comma-separated environment variable names to read the GLM API key from.')
    parser.add_argument('--llm-temperature', type=float, default=0.0, help='LLM temperature for deterministic order priors.')
    parser.add_argument('--llm-max-tokens', type=int, default=2048, help='Maximum tokens for one LLM prior response.')
    parser.add_argument('--llm-timeout', type=int, default=60, help='Timeout in seconds for one LLM API request.')
    parser.add_argument('--llm-refresh', action='store_true', help='Ignore existing cached LLM priors and call the API again.')
    parser.add_argument('--llm-cache-only', action='store_true', help='Use cached LLM priors only; do not call the API.')
    parser.add_argument('--llm-order-default-confidence', type=float, default=0.7, help='Fallback confidence for pairwise probabilities derived from an LLM order.')
    parser.add_argument('--llm-soft-weight', type=float, default=0.7, help='Weight of the LLM pairwise order probability in soft fusion.')
    parser.add_argument('--llm-conflict-weight', type=float, default=0.35, help='Reduced LLM weight when conflict-aware fusion detects a low-confidence LLM/data conflict.')
    parser.add_argument('--llm-conflict-trust-threshold', type=float, default=0.35, help='LLM reliability threshold above which conflict-aware fusion keeps the normal LLM weight.')
    parser.add_argument('--adaptive-min-llm-weight', type=float, default=0.1, help='Minimum pairwise LLM weight in adaptive reliability-calibrated fusion.')
    parser.add_argument('--adaptive-max-llm-weight', type=float, default=0.95, help='Maximum pairwise LLM weight in adaptive reliability-calibrated fusion.')
    parser.add_argument('--adaptive-data-reliability', type=float, default=1.0, help='Multiplier for empirical direction reliability in adaptive fusion.')
    parser.add_argument('--adaptive-rank-separation-gain', type=float, default=0.45, help='Converts LLM rank separation into an additional pairwise order margin.')
    parser.add_argument('--adaptive-reliability-power', type=float, default=1.0, help='Power applied to LLM/data directional strengths before computing adaptive weights.')
    parser.add_argument('--adaptive-confidence-floor', type=float, default=0.35, help='Minimum contribution of LLM confidence to adaptive reliability.')
    parser.add_argument('--adaptive-undirected-data-discount', type=float, default=0.25, help='Discount applied to empirical reliability for undirected base-learner adjacencies.')
    parser.add_argument('--adaptive-conflict-data-boost', type=float, default=1.35, help='Boost applied to empirical reliability when LLM and data directions conflict.')
    parser.add_argument('--adaptive-low-conflict-llm-discount', type=float, default=0.5, help='Discount applied to LLM reliability in low-confidence conflicts.')
    parser.add_argument('--adaptive-api-source-weight', type=float, default=1.0, help='Source reliability multiplier for API LLM priors.')
    parser.add_argument('--adaptive-consensus-source-weight', type=float, default=1.1, help='Source reliability multiplier for self-consistency consensus priors.')
    parser.add_argument('--adaptive-heuristic-source-weight', type=float, default=0.85, help='Source reliability multiplier for development heuristic fallback priors.')
    parser.add_argument('--adaptive-perturbed-prompt-source-weight', type=float, default=0.55, help='Additional reliability multiplier for non-full LLM prompt modes in adaptive fusion.')
    parser.add_argument('--data-directed-confidence', type=float, default=0.62, help='Pseudo-confidence assigned to a directed edge produced by a data-driven base method.')
    parser.add_argument('--llm-fallback', choices=['heuristic', 'skip'], default='skip', help='When no LLM prior can be obtained, either use a weak semantic heuristic or skip LLM-order algorithms.')
    return parser

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    results = run_experiments(args)
    if results:
        print(f"\nSaved detailed results to: {Path(args.results_dir) / 'baseline_results.csv'}")
        print(f"Saved aggregate results to: {Path(args.results_dir) / 'baseline_aggregate.csv'}")
        print(f"Saved summary results to:   {Path(args.results_dir) / 'baseline_summary.csv'}")
        print(f"Saved paper tables to:      {Path(args.results_dir) / 'paper_tables'}")
    else:
        print('No runs were executed. Check --collections, --datasets, --sample-sizes, and --seeds.')
if __name__ == '__main__':
    main()
