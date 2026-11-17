from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import math
import os
import time
from pathlib import Path
import baseline
ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / 'data'
DEFAULT_RESULTS_DIR = ROOT / 'results_final_strong_baselines'
try:
    import numpy as np
except Exception:
    np = None
STRONG_BASELINE_ALGORITHMS = ['notears_castle', 'notears_linear_lite', 'golem_castle', 'golem_linear_lite', 'dagma_linear', 'dag_gnn_castle', 'gran_dag_castle', 'fci_cl', 'direct_lingam', 'ica_lingam', 'resit_lingam', 'cam_uv_lingam']

def parse_csv_list(value: str) -> list[str]:
    seen = set()
    out = []
    for item in value.split(','):
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out

def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(',') if item.strip()]

def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def write_partial_results(results_dir: Path, results: list[dict], skipped: list[dict]) -> None:
    if results:
        fieldnames = sorted({k for row in results for k in row.keys()})
        preferred = ['collection', 'dataset', 'algorithm', 'sample_size', 'seed', 'sample_source', 'data_type', 'runtime_sec', 'shd', 'tpr', 'precision', 'f1', 'skeleton_precision', 'skeleton_recall', 'skeleton_f1', 'dtop', 'dtop_rate', 'cyclic', 'true_edges', 'pred_directed_edges', 'pred_undirected_edges', 'pred_skeleton_edges']
        ordered = [f for f in preferred if f in fieldnames] + [f for f in fieldnames if f not in preferred]
        baseline.write_csv_dicts(results_dir / 'baseline_results_partial.csv', results, ordered)
    if skipped:
        baseline.write_csv_dicts(results_dir / 'baseline_skipped_partial.csv', skipped, sorted({k for row in skipped for k in row.keys()}))

def module_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None

def attribute_available(module: str, attribute: str) -> bool:
    if not module_available(module):
        return False
    try:
        imported = __import__(module, fromlist=[attribute])
        return hasattr(imported, attribute)
    except Exception:
        return False

def write_status(results_dir: Path, algorithms: list[str]) -> list[dict]:
    status_specs = {'notears_castle': {'method': 'NOTEARS', 'modules_checked': 'castle.algorithms.Notears', 'available': np is not None and module_available('castle.algorithms'), 'note': 'Mature gCastle NOTEARS implementation; continuous-data baseline.'}, 'notears_linear_lite': {'method': 'NOTEARS linear lite', 'modules_checked': 'numpy,scipy', 'available': np is not None and module_available('scipy'), 'note': 'Self-contained linear NOTEARS-style augmented Lagrangian implementation.'}, 'golem_castle': {'method': 'GOLEM', 'modules_checked': 'castle.algorithms.GOLEM,torch', 'available': np is not None and module_available('castle.algorithms') and module_available('torch'), 'note': 'Mature gCastle GOLEM implementation; continuous-data baseline.'}, 'golem_linear_lite': {'method': 'GOLEM linear lite', 'modules_checked': 'numpy,scipy', 'available': np is not None and module_available('scipy'), 'note': 'Self-contained GOLEM-style squared-loss acyclicity-penalized implementation.'}, 'dagma_linear': {'method': 'DAGMA', 'modules_checked': 'dagma.linear.DagmaLinear', 'available': np is not None and module_available('dagma.linear'), 'note': 'Official DAGMA linear model; continuous-data baseline.'}, 'dag_gnn_castle': {'method': 'DAG-GNN', 'modules_checked': 'castle.algorithms.DAG_GNN,torch', 'available': np is not None and module_available('castle.algorithms') and module_available('torch'), 'note': 'Mature gCastle DAG-GNN implementation with CPU-safe iteration defaults.'}, 'gran_dag_castle': {'method': 'GraN-DAG', 'modules_checked': 'castle.algorithms.GraNDAG,torch', 'available': np is not None and module_available('castle.algorithms') and module_available('torch'), 'note': 'Mature gCastle GraN-DAG implementation with CPU-safe iteration defaults.'}, 'fci_cl': {'method': 'FCI', 'modules_checked': 'causallearn.search.ConstraintBased.FCI', 'available': module_available('causallearn.search.ConstraintBased.FCI'), 'note': 'causal-learn FCI PAG baseline; non-tail-arrow marks are evaluated as undirected adjacencies.'}, 'direct_lingam': {'method': 'DirectLiNGAM', 'modules_checked': 'lingam', 'available': module_available('lingam'), 'note': 'External lingam package wrapper from baseline.py.'}, 'ica_lingam': {'method': 'ICALiNGAM', 'modules_checked': 'lingam', 'available': module_available('lingam'), 'note': 'External lingam package wrapper from baseline.py.'}, 'resit_lingam': {'method': 'RESIT', 'modules_checked': 'lingam.RESIT,sklearn', 'available': attribute_available('lingam', 'RESIT') and module_available('sklearn.ensemble'), 'note': 'lingam RESIT with random-forest regression; continuous nonlinear additive-noise baseline.'}, 'cam_uv_lingam': {'method': 'CAM-family CAM-UV', 'modules_checked': 'lingam.CAMUV', 'available': attribute_available('lingam', 'CAMUV'), 'note': 'CAM-family baseline via lingam.CAMUV; used because the original R CAM package is not a stable pure-Python dependency.'}, 'rfci': {'method': 'RFCI', 'modules_checked': 'causallearn RFCI', 'available': False, 'note': 'Current causal-learn install exposes FCI but not RFCI.'}}
    rows = []
    for algorithm, spec in status_specs.items():
        requested = algorithm in algorithms or algorithm in {'dagma', 'rfci'}
        rows.append({'algorithm': algorithm, 'method': spec['method'], 'requested': str(requested).lower(), 'modules_checked': spec['modules_checked'], 'available': str(bool(spec['available'])).lower(), 'action': 'run' if requested and spec['available'] and (algorithm in algorithms) else 'record_only', 'note': spec['note']})
    write_csv(results_dir / 'strong_baseline_status.csv', rows, ['algorithm', 'method', 'requested', 'modules_checked', 'available', 'action', 'note'])
    return rows

def continuous_matrix(headers: list[str], rows: list[dict[str, str]], data_type: str):
    if data_type != 'continuous':
        raise baseline.AlgorithmSkip('continuous optimization baselines require continuous data')
    if np is None:
        raise baseline.AlgorithmSkip('numpy is not available')
    return baseline.encode_rows_for_numpy(headers, rows, data_type, standardize=True)

def deterministic_subsample_matrix(X, max_samples: int, seed: int):
    if max_samples <= 0 or X.shape[0] <= max_samples:
        return (X, False)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(X.shape[0], size=max_samples, replace=False))
    return (X[idx], True)

def acyclicity(W):
    from scipy.linalg import expm
    return float(np.trace(expm(W * W)) - W.shape[0])

def acyclicity_grad(W):
    from scipy.linalg import expm
    E = expm(W * W)
    G = E.T * (2.0 * W)
    np.fill_diagonal(G, 0.0)
    return (float(np.trace(E) - W.shape[0]), G)

def squared_loss_grad(X, W):
    n = X.shape[0]
    residual = X @ W - X
    loss = 0.5 / n * float(np.sum(residual * residual))
    grad = X.T @ residual / n
    np.fill_diagonal(grad, 0.0)
    return (loss, grad)

def split_to_matrix(w, d: int):
    W = w[:d * d].reshape(d, d) - w[d * d:].reshape(d, d)
    np.fill_diagonal(W, 0.0)
    return W

def split_bounds(d: int):
    bounds = []
    for block in range(2):
        for i in range(d):
            for j in range(d):
                bounds.append((0.0, 0.0) if i == j else (0.0, None))
    return bounds

def fit_notears_linear(X, lambda1: float, max_outer: int, max_inner: int, h_tol: float):
    from scipy.optimize import minimize
    d = X.shape[1]
    w_est = np.zeros(2 * d * d, dtype=float)
    bounds = split_bounds(d)
    rho = 1.0
    alpha = 0.0
    h_prev = math.inf
    result_status = 'not_started'

    def objective(w, rho_value: float, alpha_value: float):
        W = split_to_matrix(w, d)
        loss, grad_loss = squared_loss_grad(X, W)
        h_value, grad_h = acyclicity_grad(W)
        smooth = grad_loss + (rho_value * h_value + alpha_value) * grad_h
        obj = loss + 0.5 * rho_value * h_value * h_value + alpha_value * h_value + lambda1 * float(np.sum(w))
        grad = np.concatenate([(smooth + lambda1).ravel(), (-smooth + lambda1).ravel()])
        return (obj, grad)
    h_value = math.inf
    for _ in range(max_outer):
        while rho < 1e+16:
            res = minimize(lambda w: objective(w, rho, alpha), w_est, method='L-BFGS-B', jac=True, bounds=bounds, options={'maxiter': max_inner, 'ftol': 1e-10})
            candidate = split_to_matrix(res.x, d)
            h_value = acyclicity(candidate)
            result_status = str(res.message)
            if h_value <= 0.25 * h_prev:
                w_est = res.x
                break
            rho *= 10.0
        alpha += rho * h_value
        h_prev = h_value
        if h_value <= h_tol:
            break
    W = split_to_matrix(w_est, d)
    return (W, {'notears_h': round(float(acyclicity(W)), 10), 'notears_status': result_status})

def fit_golem_linear(X, lambda1: float, lambda_h: float, max_iter: int):
    from scipy.optimize import minimize
    d = X.shape[1]
    w0 = np.zeros(2 * d * d, dtype=float)
    bounds = split_bounds(d)

    def objective(w):
        W = split_to_matrix(w, d)
        loss, grad_loss = squared_loss_grad(X, W)
        h_value, grad_h = acyclicity_grad(W)
        smooth = grad_loss + lambda_h * grad_h
        obj = loss + lambda_h * h_value + lambda1 * float(np.sum(w))
        grad = np.concatenate([(smooth + lambda1).ravel(), (-smooth + lambda1).ravel()])
        return (obj, grad)
    res = minimize(objective, w0, method='L-BFGS-B', jac=True, bounds=bounds, options={'maxiter': max_iter, 'ftol': 1e-10})
    W = split_to_matrix(res.x, d)
    return (W, {'golem_h': round(float(acyclicity(W)), 10), 'golem_status': str(res.message)})

def weighted_matrix_to_dag(headers: list[str], W, threshold: float):
    weighted = []
    for i, src in enumerate(headers):
        for j, dst in enumerate(headers):
            if i == j:
                continue
            weight = float(W[i, j])
            if abs(weight) >= threshold:
                weighted.append((src, dst, abs(weight), weight))
    edges: set[tuple[str, str]] = set()
    for src, dst, _, _signed in sorted(weighted, key=lambda item: item[2], reverse=True):
        candidate = set(edges)
        candidate.add((src, dst))
        if baseline.is_acyclic(headers, candidate):
            edges = candidate
    return edges

def source_target_matrix_to_dag(headers: list[str], matrix, threshold: float):
    W = np.asarray(matrix, dtype=float)
    W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    weighted = []
    for i, src in enumerate(headers):
        for j, dst in enumerate(headers):
            if i == j:
                continue
            weight = float(W[i, j])
            if abs(weight) >= threshold:
                weighted.append((src, dst, abs(weight), weight))
    edges: set[tuple[str, str]] = set()
    for src, dst, _, _signed in sorted(weighted, key=lambda item: item[2], reverse=True):
        candidate = set(edges)
        candidate.add((src, dst))
        if baseline.is_acyclic(headers, candidate):
            edges = candidate
    return edges

def lingam_target_source_matrix_to_dag(headers: list[str], matrix, threshold: float):
    adjacency = np.asarray(matrix, dtype=float)
    adjacency = np.nan_to_num(adjacency, nan=0.0, posinf=0.0, neginf=0.0)
    directed = set()
    for target_idx, source_idx in zip(*np.where(np.abs(adjacency) > threshold)):
        source = headers[int(source_idx)]
        target = headers[int(target_idx)]
        if source == target:
            continue
        candidate = set(directed)
        candidate.add((source, target))
        if baseline.is_acyclic(headers, candidate):
            directed = candidate
    return directed

def castle_causal_matrix(model):
    for attr in ('causal_matrix', 'weight_causal_matrix', 'adjacency_matrix'):
        if hasattr(model, attr):
            value = getattr(model, attr)
            if value is not None:
                return value
    raise baseline.AlgorithmSkip('gCastle model did not expose a causal matrix')

def run_castle_notears(headers, rows, data_type, args, seed: int):
    X = continuous_matrix(headers, rows, data_type)
    os.environ.setdefault('CASTLE_BACKEND', 'pytorch')
    try:
        from castle.algorithms import Notears
    except Exception as exc:
        raise baseline.AlgorithmSkip(f'gCastle NOTEARS unavailable: {exc}') from exc
    model = Notears(lambda1=args.castle_notears_lambda1, loss_type='l2', max_iter=args.castle_notears_max_iter, h_tol=args.castle_notears_h_tol, w_threshold=args.castle_notears_threshold)
    model.learn(X, columns=headers)
    directed = source_target_matrix_to_dag(headers, castle_causal_matrix(model), args.castle_notears_matrix_threshold)
    return {'directed_edges': directed, 'undirected_edges': set(), 'completed_edges': directed, 'extra': {'library': 'gCastle', 'variant': 'NOTEARS', 'castle_notears_lambda1': args.castle_notears_lambda1, 'castle_notears_threshold': args.castle_notears_threshold, 'seed': seed}}

def run_castle_golem(headers, rows, data_type, args, seed: int):
    X = continuous_matrix(headers, rows, data_type)
    os.environ.setdefault('CASTLE_BACKEND', 'pytorch')
    try:
        from castle.algorithms import GOLEM
    except Exception as exc:
        raise baseline.AlgorithmSkip(f'gCastle GOLEM unavailable: {exc}') from exc
    model = GOLEM(lambda_1=args.castle_golem_lambda1, lambda_2=args.castle_golem_lambda2, equal_variances=True, learning_rate=args.castle_golem_lr, num_iter=args.castle_golem_num_iter, checkpoint_iter=max(1, min(args.castle_golem_num_iter, args.castle_golem_checkpoint_iter)), seed=seed, graph_thres=args.castle_golem_threshold, device_type='cpu')
    model.learn(X, columns=headers)
    directed = source_target_matrix_to_dag(headers, castle_causal_matrix(model), args.castle_golem_matrix_threshold)
    return {'directed_edges': directed, 'undirected_edges': set(), 'completed_edges': directed, 'extra': {'library': 'gCastle', 'variant': 'GOLEM', 'castle_golem_lambda1': args.castle_golem_lambda1, 'castle_golem_lambda2': args.castle_golem_lambda2, 'castle_golem_num_iter': args.castle_golem_num_iter, 'seed': seed}}

def run_dagma_linear(headers, rows, data_type, args, seed: int):
    X = continuous_matrix(headers, rows, data_type)
    try:
        from dagma.linear import DagmaLinear
    except Exception as exc:
        raise baseline.AlgorithmSkip(f'DAGMA unavailable: {exc}') from exc
    model = DagmaLinear(loss_type='l2', verbose=False)
    W = model.fit(X, lambda1=args.dagma_lambda1, w_threshold=args.dagma_threshold, warm_iter=args.dagma_warm_iter, max_iter=args.dagma_max_iter, checkpoint=max(1, args.dagma_checkpoint), lr=args.dagma_lr)
    directed = weighted_matrix_to_dag(headers, W, args.dagma_matrix_threshold)
    return {'directed_edges': directed, 'undirected_edges': set(), 'completed_edges': directed, 'extra': {'library': 'dagma', 'variant': 'DAGMA linear', 'dagma_lambda1': args.dagma_lambda1, 'dagma_threshold': args.dagma_threshold, 'dagma_warm_iter': args.dagma_warm_iter, 'dagma_max_iter': args.dagma_max_iter, 'seed': seed}}

def run_castle_dag_gnn(headers, rows, data_type, args, seed: int):
    X = continuous_matrix(headers, rows, data_type)
    os.environ.setdefault('CASTLE_BACKEND', 'pytorch')
    try:
        from castle.algorithms import DAG_GNN
    except Exception as exc:
        raise baseline.AlgorithmSkip(f'gCastle DAG-GNN unavailable: {exc}') from exc
    model = DAG_GNN(epochs=args.dag_gnn_epochs, k_max_iter=args.dag_gnn_k_max_iter, batch_size=min(args.dag_gnn_batch_size, max(1, X.shape[0])), lr=args.dag_gnn_lr, graph_threshold=args.dag_gnn_threshold, seed=seed, device_type='cpu')
    model.learn(X, columns=headers)
    directed = source_target_matrix_to_dag(headers, castle_causal_matrix(model), args.dag_gnn_matrix_threshold)
    return {'directed_edges': directed, 'undirected_edges': set(), 'completed_edges': directed, 'extra': {'library': 'gCastle', 'variant': 'DAG-GNN', 'dag_gnn_epochs': args.dag_gnn_epochs, 'dag_gnn_k_max_iter': args.dag_gnn_k_max_iter, 'seed': seed}}

def run_castle_gran_dag(headers, rows, data_type, args, seed: int):
    X = continuous_matrix(headers, rows, data_type)
    os.environ.setdefault('CASTLE_BACKEND', 'pytorch')
    try:
        from castle.algorithms import GraNDAG
    except Exception as exc:
        raise baseline.AlgorithmSkip(f'gCastle GraN-DAG unavailable: {exc}') from exc
    model = GraNDAG(input_dim=len(headers), hidden_num=args.gran_dag_hidden_num, hidden_dim=args.gran_dag_hidden_dim, batch_size=min(args.gran_dag_batch_size, max(1, X.shape[0])), lr=args.gran_dag_lr, iterations=args.gran_dag_iterations, device_type='cpu', random_seed=seed, normalize=True)
    model.learn(X, columns=headers)
    directed = source_target_matrix_to_dag(headers, castle_causal_matrix(model), args.gran_dag_matrix_threshold)
    return {'directed_edges': directed, 'undirected_edges': set(), 'completed_edges': directed, 'extra': {'library': 'gCastle', 'variant': 'GraN-DAG', 'gran_dag_iterations': args.gran_dag_iterations, 'seed': seed}}

def run_resit_lingam(headers, rows, data_type, args, seed: int):
    X = continuous_matrix(headers, rows, data_type)
    X_fit, subsampled = deterministic_subsample_matrix(X, args.resit_max_samples, seed)
    try:
        import lingam
        from sklearn.ensemble import RandomForestRegressor
    except Exception as exc:
        raise baseline.AlgorithmSkip(f'RESIT dependencies unavailable: {exc}') from exc
    regressor = RandomForestRegressor(n_estimators=args.resit_n_estimators, max_depth=args.resit_max_depth if args.resit_max_depth > 0 else None, random_state=seed, n_jobs=1)
    model = lingam.RESIT(regressor=regressor, random_state=seed, alpha=args.resit_alpha)
    model.fit(X_fit)
    directed = lingam_target_source_matrix_to_dag(headers, model.adjacency_matrix_, args.resit_threshold)
    return {'directed_edges': directed, 'undirected_edges': set(), 'completed_edges': directed, 'extra': {'library': 'lingam', 'variant': 'RESIT', 'resit_alpha': args.resit_alpha, 'resit_threshold': args.resit_threshold, 'resit_n_estimators': args.resit_n_estimators, 'resit_max_samples': args.resit_max_samples, 'resit_subsampled': str(subsampled).lower(), 'seed': seed}}

def run_cam_uv_lingam(headers, rows, data_type, args, seed: int):
    X = continuous_matrix(headers, rows, data_type)
    X_fit, subsampled = deterministic_subsample_matrix(X, args.cam_max_samples, seed)
    try:
        import lingam
    except Exception as exc:
        raise baseline.AlgorithmSkip(f'CAM-UV dependencies unavailable: {exc}') from exc
    model = lingam.CAMUV(alpha=args.cam_alpha, num_explanatory_vals=args.cam_num_explanatory_vals, independence=args.cam_independence, ind_corr=args.cam_ind_corr)
    model.fit(X_fit)
    directed = lingam_target_source_matrix_to_dag(headers, model.adjacency_matrix_, args.cam_threshold)
    return {'directed_edges': directed, 'undirected_edges': set(), 'completed_edges': directed, 'extra': {'library': 'lingam', 'variant': 'CAM-UV', 'cam_alpha': args.cam_alpha, 'cam_threshold': args.cam_threshold, 'cam_independence': args.cam_independence, 'cam_max_samples': args.cam_max_samples, 'cam_subsampled': str(subsampled).lower(), 'seed': seed}}

def run_notears_linear_lite(headers, rows, data_type, args):
    X = continuous_matrix(headers, rows, data_type)
    W, extra = fit_notears_linear(X, lambda1=args.notears_lambda1, max_outer=args.notears_max_outer, max_inner=args.notears_max_inner, h_tol=args.notears_h_tol)
    directed = weighted_matrix_to_dag(headers, W, args.notears_threshold)
    return {'directed_edges': directed, 'undirected_edges': set(), 'completed_edges': directed, 'extra': {'library': 'strong_baselines.py', 'variant': 'notears_linear_lite', 'notears_threshold': args.notears_threshold, 'notears_lambda1': args.notears_lambda1, **extra}}

def run_golem_linear_lite(headers, rows, data_type, args):
    X = continuous_matrix(headers, rows, data_type)
    W, extra = fit_golem_linear(X, lambda1=args.golem_lambda1, lambda_h=args.golem_lambda_h, max_iter=args.golem_max_iter)
    directed = weighted_matrix_to_dag(headers, W, args.golem_threshold)
    return {'directed_edges': directed, 'undirected_edges': set(), 'completed_edges': directed, 'extra': {'library': 'strong_baselines.py', 'variant': 'golem_linear_lite', 'golem_threshold': args.golem_threshold, 'golem_lambda1': args.golem_lambda1, 'golem_lambda_h': args.golem_lambda_h, **extra}}

def run_fci_cl(headers, rows, data_type, args):
    try:
        from causallearn.search.ConstraintBased.FCI import fci
    except Exception as exc:
        raise baseline.AlgorithmSkip(f'causal-learn FCI unavailable: {exc}') from exc
    arr = baseline.encode_rows_for_numpy(headers, rows, data_type, standardize=False)
    indep_test = 'chisq' if data_type == 'categorical' else 'fisherz'
    graph, _ = fci(arr, independence_test_method=indep_test, alpha=args.alpha, depth=args.fci_depth, max_path_length=args.fci_max_path_length, verbose=False, show_progress=False, node_names=headers)
    directed, undirected = baseline.parse_causallearn_general_graph(graph, headers)
    completed = baseline.complete_cpdag_to_dag(headers, directed, undirected)
    return {'directed_edges': directed, 'undirected_edges': undirected, 'completed_edges': completed, 'extra': {'library': 'causal-learn', 'variant': 'FCI', 'indep_test': indep_test, 'alpha': args.alpha, 'fci_depth': args.fci_depth, 'fci_max_path_length': args.fci_max_path_length}}

def run_strong_algorithm(algorithm: str, headers, rows, data_type, args, seed: int):
    if algorithm == 'notears_castle':
        return run_castle_notears(headers, rows, data_type, args, seed)
    if algorithm == 'notears_linear_lite':
        return run_notears_linear_lite(headers, rows, data_type, args)
    if algorithm == 'golem_castle':
        return run_castle_golem(headers, rows, data_type, args, seed)
    if algorithm == 'golem_linear_lite':
        return run_golem_linear_lite(headers, rows, data_type, args)
    if algorithm == 'dagma_linear':
        return run_dagma_linear(headers, rows, data_type, args, seed)
    if algorithm == 'dag_gnn_castle':
        return run_castle_dag_gnn(headers, rows, data_type, args, seed)
    if algorithm == 'gran_dag_castle':
        return run_castle_gran_dag(headers, rows, data_type, args, seed)
    if algorithm == 'fci_cl':
        return run_fci_cl(headers, rows, data_type, args)
    if algorithm in {'direct_lingam', 'ica_lingam'}:
        return baseline.run_lingam(headers, rows, data_type, algorithm, seed, args.lingam_threshold)
    if algorithm == 'resit_lingam':
        return run_resit_lingam(headers, rows, data_type, args, seed)
    if algorithm == 'cam_uv_lingam':
        return run_cam_uv_lingam(headers, rows, data_type, args, seed)
    raise ValueError(f'Unknown strong baseline algorithm: {algorithm}')

def run_experiments(args: argparse.Namespace) -> list[dict]:
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    graph_out_dir = results_dir / 'baseline_graphs'
    results_dir.mkdir(parents=True, exist_ok=True)
    graph_out_dir.mkdir(parents=True, exist_ok=True)
    requested_algorithms = parse_csv_list(args.algorithms)
    status_rows = write_status(results_dir, requested_algorithms)
    availability = {row['algorithm']: row['available'] == 'true' for row in status_rows}
    algorithms = [alg for alg in requested_algorithms if availability.get(alg, False)]
    missing = [alg for alg in requested_algorithms if alg not in algorithms]
    if missing:
        print(f"[strong-baselines] unavailable/skipped algorithms: {','.join(missing)}", flush=True)
    if not algorithms:
        print('[strong-baselines] no requested strong baseline is available.', flush=True)
        return []
    index_rows = baseline.read_csv_dicts(data_dir / 'dataset_index.csv')
    collections = set(parse_csv_list(args.collections))
    selected_datasets = set(parse_csv_list(args.datasets)) if args.datasets else None
    sample_sizes = parse_int_list(args.sample_sizes)
    seeds = parse_int_list(args.seeds)
    results = []
    skipped = []
    for ds_row in index_rows:
        collection = ds_row['collection']
        dataset = ds_row['dataset']
        if collection not in collections:
            continue
        if selected_datasets and dataset not in selected_datasets:
            continue
        ds_dir = data_dir / ds_row['path']
        true_edges = baseline.read_edges(ds_dir / 'edges.csv')
        for sample_size in sample_sizes:
            for seed in seeds:
                try:
                    headers, rows, sample_source = baseline.load_or_generate_rows(collection, ds_dir, dataset, sample_size, seed, args.sample_mode)
                except Exception as exc:
                    skipped.append({'collection': collection, 'dataset': dataset, 'sample_size': sample_size, 'seed': seed, 'algorithm': '*', 'reason': str(exc)})
                    continue
                data_type = baseline.infer_rows_data_type(collection, rows)
                print(f'[run] {collection}/{dataset} n={sample_size} seed={seed} type={data_type} source={sample_source}', flush=True)
                for algorithm in algorithms:
                    start = time.perf_counter()
                    try:
                        prediction = run_strong_algorithm(algorithm, headers, rows, data_type, args, seed)
                    except baseline.AlgorithmSkip as exc:
                        skipped.append({'collection': collection, 'dataset': dataset, 'sample_size': sample_size, 'seed': seed, 'algorithm': algorithm, 'reason': str(exc)})
                        print(f'      {algorithm}: skipped ({exc})', flush=True)
                        continue
                    runtime_sec = round(time.perf_counter() - start, 6)
                    directed_edges = prediction['directed_edges']
                    undirected_edges = prediction['undirected_edges']
                    completed_edges = prediction['completed_edges']
                    metrics = baseline.evaluate_graph(headers, true_edges, directed_edges, undirected_edges, completed_edges)
                    result = {'collection': collection, 'dataset': dataset, 'algorithm': algorithm, 'sample_size': sample_size, 'seed': seed, 'sample_source': sample_source, 'data_type': data_type, 'runtime_sec': runtime_sec, **metrics, **prediction.get('extra', {})}
                    results.append(result)
                    baseline.save_predicted_edges(graph_out_dir, dataset, algorithm, sample_size, seed, directed_edges, undirected_edges)
                    if args.flush_every > 0 and len(results) % args.flush_every == 0:
                        write_partial_results(results_dir, results, skipped)
                    print(f"      {algorithm}: SHD={result['shd']} TPR={result['tpr']} F1={result['f1']} SkelF1={result['skeleton_f1']} DtopRate={result['dtop_rate']}", flush=True)
    if results:
        fieldnames = sorted({k for row in results for k in row.keys()})
        preferred = ['collection', 'dataset', 'algorithm', 'sample_size', 'seed', 'sample_source', 'data_type', 'runtime_sec', 'shd', 'tpr', 'precision', 'f1', 'skeleton_precision', 'skeleton_recall', 'skeleton_f1', 'dtop', 'dtop_rate', 'cyclic', 'true_edges', 'pred_directed_edges', 'pred_undirected_edges', 'pred_skeleton_edges']
        ordered = [f for f in preferred if f in fieldnames] + [f for f in fieldnames if f not in preferred]
        baseline.write_csv_dicts(results_dir / 'baseline_results.csv', results, ordered)
        aggregate_rows = baseline.aggregate_results(results, ['collection', 'dataset', 'sample_size', 'algorithm'])
        aggregate_fields = sorted({k for row in aggregate_rows for k in row.keys()})
        aggregate_ordered = ['collection', 'dataset', 'sample_size', 'algorithm', 'runs'] + [f for f in aggregate_fields if f not in {'collection', 'dataset', 'sample_size', 'algorithm', 'runs'}]
        baseline.write_csv_dicts(results_dir / 'baseline_aggregate.csv', aggregate_rows, aggregate_ordered)
        summary_rows = baseline.aggregate_results(results, ['collection', 'algorithm'])
        summary_fields = sorted({k for row in summary_rows for k in row.keys()})
        summary_ordered = ['collection', 'algorithm', 'runs'] + [f for f in summary_fields if f not in {'collection', 'algorithm', 'runs'}]
        baseline.write_csv_dicts(results_dir / 'baseline_summary.csv', summary_rows, summary_ordered)
    if skipped:
        baseline.write_csv_dicts(results_dir / 'baseline_skipped.csv', skipped, sorted({k for row in skipped for k in row.keys()}))
    if results:
        baseline.make_paper_tables(results_dir, results, skipped)
    config = {'created_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'collections': sorted(collections), 'datasets': sorted(selected_datasets) if selected_datasets else 'all', 'sample_sizes': sample_sizes, 'seeds': seeds, 'requested_algorithms': requested_algorithms, 'executed_algorithms': algorithms, 'sample_mode': args.sample_mode, 'castle_notears_lambda1': args.castle_notears_lambda1, 'castle_notears_threshold': args.castle_notears_threshold, 'castle_notears_max_iter': args.castle_notears_max_iter, 'notears_lambda1': args.notears_lambda1, 'notears_threshold': args.notears_threshold, 'castle_golem_lambda1': args.castle_golem_lambda1, 'castle_golem_lambda2': args.castle_golem_lambda2, 'castle_golem_num_iter': args.castle_golem_num_iter, 'golem_lambda1': args.golem_lambda1, 'golem_lambda_h': args.golem_lambda_h, 'golem_threshold': args.golem_threshold, 'dagma_lambda1': args.dagma_lambda1, 'dagma_threshold': args.dagma_threshold, 'dagma_warm_iter': args.dagma_warm_iter, 'dagma_max_iter': args.dagma_max_iter, 'dag_gnn_epochs': args.dag_gnn_epochs, 'dag_gnn_k_max_iter': args.dag_gnn_k_max_iter, 'gran_dag_iterations': args.gran_dag_iterations, 'fci_depth': args.fci_depth, 'fci_max_path_length': args.fci_max_path_length, 'lingam_threshold': args.lingam_threshold, 'resit_alpha': args.resit_alpha, 'resit_threshold': args.resit_threshold, 'resit_max_samples': args.resit_max_samples, 'cam_alpha': args.cam_alpha, 'cam_threshold': args.cam_threshold, 'cam_max_samples': args.cam_max_samples}
    (results_dir / 'strong_baseline_run_config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    return results

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run stronger causal discovery baselines.')
    parser.add_argument('--data-dir', default=str(DEFAULT_DATA_DIR))
    parser.add_argument('--results-dir', default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument('--collections', default='llm_causal_order_graphs')
    parser.add_argument('--datasets', default='')
    parser.add_argument('--sample-sizes', default='250,500,1000,5000')
    parser.add_argument('--seeds', default='0,1,2,3,4')
    parser.add_argument('--sample-mode', choices=['generated', 'bootstrap', 'fixed'], default='generated')
    parser.add_argument('--algorithms', default=','.join(STRONG_BASELINE_ALGORITHMS))
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--fci-depth', type=int, default=2)
    parser.add_argument('--fci-max-path-length', type=int, default=2)
    parser.add_argument('--lingam-threshold', type=float, default=0.05)
    parser.add_argument('--castle-notears-lambda1', type=float, default=0.1)
    parser.add_argument('--castle-notears-threshold', type=float, default=0.3)
    parser.add_argument('--castle-notears-matrix-threshold', type=float, default=0.5)
    parser.add_argument('--castle-notears-max-iter', type=int, default=100)
    parser.add_argument('--castle-notears-h-tol', type=float, default=1e-08)
    parser.add_argument('--notears-lambda1', type=float, default=0.01)
    parser.add_argument('--notears-threshold', type=float, default=0.12)
    parser.add_argument('--notears-max-outer', type=int, default=10)
    parser.add_argument('--notears-max-inner', type=int, default=100)
    parser.add_argument('--notears-h-tol', type=float, default=1e-08)
    parser.add_argument('--castle-golem-lambda1', type=float, default=0.02)
    parser.add_argument('--castle-golem-lambda2', type=float, default=5.0)
    parser.add_argument('--castle-golem-lr', type=float, default=0.001)
    parser.add_argument('--castle-golem-num-iter', type=int, default=1500)
    parser.add_argument('--castle-golem-checkpoint-iter', type=int, default=500)
    parser.add_argument('--castle-golem-threshold', type=float, default=0.3)
    parser.add_argument('--castle-golem-matrix-threshold', type=float, default=0.5)
    parser.add_argument('--golem-lambda1', type=float, default=0.01)
    parser.add_argument('--golem-lambda-h', type=float, default=5.0)
    parser.add_argument('--golem-threshold', type=float, default=0.12)
    parser.add_argument('--golem-max-iter', type=int, default=300)
    parser.add_argument('--dagma-lambda1', type=float, default=0.03)
    parser.add_argument('--dagma-threshold', type=float, default=0.3)
    parser.add_argument('--dagma-matrix-threshold', type=float, default=0.05)
    parser.add_argument('--dagma-warm-iter', type=int, default=1000)
    parser.add_argument('--dagma-max-iter', type=int, default=2000)
    parser.add_argument('--dagma-checkpoint', type=int, default=500)
    parser.add_argument('--dagma-lr', type=float, default=0.0003)
    parser.add_argument('--dag-gnn-epochs', type=int, default=50)
    parser.add_argument('--dag-gnn-k-max-iter', type=float, default=20.0)
    parser.add_argument('--dag-gnn-batch-size', type=int, default=100)
    parser.add_argument('--dag-gnn-lr', type=float, default=0.003)
    parser.add_argument('--dag-gnn-threshold', type=float, default=0.3)
    parser.add_argument('--dag-gnn-matrix-threshold', type=float, default=0.5)
    parser.add_argument('--gran-dag-iterations', type=int, default=1000)
    parser.add_argument('--gran-dag-hidden-num', type=int, default=2)
    parser.add_argument('--gran-dag-hidden-dim', type=int, default=10)
    parser.add_argument('--gran-dag-batch-size', type=int, default=64)
    parser.add_argument('--gran-dag-lr', type=float, default=0.001)
    parser.add_argument('--gran-dag-matrix-threshold', type=float, default=0.5)
    parser.add_argument('--resit-alpha', type=float, default=0.01)
    parser.add_argument('--resit-threshold', type=float, default=0.05)
    parser.add_argument('--resit-n-estimators', type=int, default=50)
    parser.add_argument('--resit-max-depth', type=int, default=0)
    parser.add_argument('--resit-max-samples', type=int, default=500)
    parser.add_argument('--cam-alpha', type=float, default=0.01)
    parser.add_argument('--cam-threshold', type=float, default=0.05)
    parser.add_argument('--cam-max-samples', type=int, default=250)
    parser.add_argument('--cam-num-explanatory-vals', type=int, default=2)
    parser.add_argument('--cam-independence', default='fcorr')
    parser.add_argument('--cam-ind-corr', type=float, default=0.5)
    parser.add_argument('--flush-every', type=int, default=1, help='Write baseline_results_partial.csv every N completed runs.')
    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    results = run_experiments(args)
    if results:
        results_dir = Path(args.results_dir)
        print(f"\nSaved strong baseline results to: {results_dir / 'baseline_results.csv'}")
        print(f"Saved strong baseline tables to:  {results_dir / 'paper_tables'}")
    else:
        print('No strong baseline runs were executed.')
if __name__ == '__main__':
    main()
