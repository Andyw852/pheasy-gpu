#!/usr/bin/env python3
"""Audited full CLI fit and grouped holdout on a prebuilt real sensing matrix.

Run under a scheduler allocation, with this checkout on PYTHONPATH:
  python dev/validate_large_fit.py full DATA_DIR --output RESULTS_DIR
  python dev/validate_large_fit.py holdout DATA_DIR --output RESULTS_DIR
No full sensing-matrix product is materialized or saved by this driver.
"""
import argparse
import inspect
import gc
import json
import os
from pathlib import Path
import pickle
import sys
import threading
import time
import traceback
import warnings

import numpy as np
from scipy import sparse as sp
import torch

from pheasy_gpu.core import optimizer as om


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=json_value) + '\n')


class Monitor:
    def __init__(self, output, devices):
        self.output, self.devices = output, devices
        self.start = time.monotonic()
        self.phase = 'start'
        self.stop = threading.Event()
        self.peak_rss = 0
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self):
        status = {}
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith(('VmRSS:', 'VmHWM:')):
                key, value = line.split(':', 1)
                status[key + '_bytes'] = int(value.split()[0]) * 1024
        self.peak_rss = max(self.peak_rss, status.get('VmHWM_bytes', 0))
        return dict(seconds=time.monotonic() - self.start, phase=self.phase, **status,
                    gpu=[dict(device=d, allocated=torch.cuda.memory_allocated(d),
                              reserved=torch.cuda.memory_reserved(d),
                              peak_allocated=torch.cuda.max_memory_allocated(d),
                              peak_reserved=torch.cuda.max_memory_reserved(d)) for d in self.devices])

    def run(self):
        with (self.output / 'resources.jsonl').open('a') as stream:
            while not self.stop.is_set():
                stream.write(json.dumps(self.sample()) + '\n')
                stream.flush()
                self.stop.wait(15)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        write_json(self.output / 'resource_summary.json', self.sample())


def force_metrics(prediction, target, rows_per_config, config_ids=None):
    prediction = np.asarray(prediction).reshape(-1, rows_per_config)
    target = np.asarray(target).reshape(prediction.shape)
    errors = prediction - target
    norm_f = np.linalg.norm(target, axis=1)
    fc = target - target.mean(axis=1, keepdims=True)
    pc = prediction - prediction.mean(axis=1, keepdims=True)
    correlation = np.sum(fc * pc, axis=1) / np.maximum(np.linalg.norm(fc, axis=1) * np.linalg.norm(pc, axis=1), 1e-300)
    return dict(rmse_eV_A=float(np.sqrt(np.mean(errors**2))),
                relative_error=float(np.linalg.norm(errors) / max(np.linalg.norm(target), 1e-300)),
                config_ids=np.arange(len(target)) if config_ids is None else config_ids,
                per_config_rmse=np.sqrt(np.mean(errors**2, axis=1)),
                per_config_relative_error=np.linalg.norm(errors, axis=1) / np.maximum(norm_f, 1e-300),
                per_config_correlation=correlation,
                minimum_correlation=float(correlation.min()))


def require_gpu(operator, devices):
    backend = getattr(operator, '_gpu_mv', None)
    if backend is None or backend._devs != devices:
        raise AssertionError('expected GPU-SM devices %s; CPU fallback or wrong devices' % devices)
    return backend


def audited_fit(optimizer, operator, forces, original_fit, devices, output, label, monitor,
                rows_per_config=1488, config_ids=None):
    backend = require_gpu(operator, devices)
    monitor.phase = label + '/fit'
    start = time.monotonic()
    records = []
    original_info = om._iterative_solver_info
    original_fista = om._fista_lasso
    fista_signature = inspect.signature(original_fista)
    calls_before = getattr(backend, "_n_calls", 0)

    def record_info(result, solver):
        info = original_info(result, solver)
        records.append(info)
        print('[SOLVER]', json.dumps(info), flush=True)
        write_json(output / (label + '_solvers.json'), records)
        if not info['converged']:
            raise AssertionError('unconverged ' + repr(info))
        return info

    def record_fista(*args, **kwargs):
        bound = fista_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if bound.arguments['_info'] is None:
            bound.arguments['_info'] = {}
        coef = original_fista(*bound.args, **bound.kwargs)
        info = dict(solver='FISTA', **bound.arguments['_info'],
                    alpha=float(bound.arguments['alpha']), tol=float(bound.arguments['tol']))
        records.append(info)
        write_json(output / (label + '_solvers.json'), records)
        print('[SOLVER]', json.dumps(info), flush=True)
        if not info.get('converged', False):
            raise AssertionError('FISTA lacks stationarity: ' + repr(info))
        return coef

    om._fista_lasso = record_fista
    om._iterative_solver_info = record_info
    try:
        original_fit(optimizer, operator, forces)
    finally:
        om._iterative_solver_info = original_info
        om._fista_lasso = original_fista
    fit_seconds = time.monotonic() - start
    require_gpu(operator, devices)
    if not records or not all(item['converged'] for item in records):
        raise AssertionError('missing or unsuccessful iterative convergence evidence')
    coefficient = np.asarray(optimizer.results['coef'])
    prediction = np.asarray(optimizer.predict(operator)).ravel()
    if not np.isfinite(coefficient).all() or not np.isfinite(prediction).all():
        raise AssertionError('non-finite coefficients or predictions')
    monitor.phase = label + '/verify'
    cpu_prediction = operator.SM_prime @ (operator.NS @ coefficient)
    error = float(np.linalg.norm(cpu_prediction - prediction) / max(np.linalg.norm(cpu_prediction), 1e-300))
    if error > 1e-10:
        raise AssertionError('real CPU/GPU force prediction mismatch: %g' % error)
    result = dict(label=label, shape=operator.shape, matrix_dtype=str(operator.dtype),
                  devices=devices, gpu_spmv_calls=backend._n_calls - calls_before,
                  fit_seconds=fit_seconds, solver_records=records,
                  nonzero=int(np.count_nonzero(coefficient)),
                  cpu_gpu_prediction_relative_error=error,
                  metrics=force_metrics(prediction, forces, rows_per_config, config_ids),
                  optimizer_metrics=optimizer.metrics)
    np.save(output / (label + '_coef.npy'), coefficient)
    write_json(output / (label + '.json'), result)
    return result


def check_exports(data, output):
    import h5py
    report = {}
    with np.load(data / 'phi.npz') as archive:
        phi = archive['Phi']
        assert phi.shape == (66375,) and np.isfinite(phi).all(), phi.shape
        report['phi'] = dict(shape=phi.shape, max_abs=float(np.max(np.abs(phi))))
    for order, shape in [(2, (62, 496, 3, 3)), (3, (62, 496, 496, 3, 3, 3))]:
        with h5py.File(data / ('fc%d.hdf5' % order)) as archive:
            dataset = archive['fc%d' % order]
            assert dataset.shape == shape, dataset.shape
            max_value, max_asr = 0.0, 0.0
            for i in range(shape[0]):
                block = dataset[i]
                assert np.isfinite(block).all()
                max_value = max(max_value, float(np.max(np.abs(block))))
                # Each remaining supercell atom index must obey the ASR.
                for atom_axis in range(order - 1):
                    max_asr = max(max_asr, float(np.max(np.abs(block.sum(axis=atom_axis)))))
            relative_asr = max_asr / max(max_value, 1e-300)
            report['fc%d' % order] = dict(shape=shape, max_abs=max_value,
                                          asr_max_absolute=max_asr, asr_relative=relative_asr)
            write_json(output / 'exports.json', report)
            if relative_asr > 1e-8:
                raise AssertionError('fc%d acoustic sum rule residual %g' % (order, relative_asr))
    return report


def full(data, output, devices, monitor):
    from pheasy_gpu.run_pheasy import main as cli_main
    original_fit = om.Optimizer.fit
    summary = {}

    def instrumented(optimizer, operator, forces, *args, **kwargs):
        assert operator.shape == (1040112, 52283), operator.shape
        summary.update(audited_fit(optimizer, operator, forces, original_fit,
                                  devices, output, 'full_ols', monitor))
        return optimizer

    om.Optimizer.fit = instrumented
    os.chdir(data)
    sys.argv = ['pheasy-gpu', '--dim', '2', '2', '2', '-w', '3', '--c2', '7.0', '--c3', '4.5',
                '--eps', '0.001', '-f', '--ndata', '699', '-l', 'OLS', '--hdf5']
    monitor.phase = 'full/load'
    try:
        cli_main()
    finally:
        om.Optimizer.fit = original_fit
    gc.collect()
    torch.cuda.empty_cache()
    monitor.phase = 'full/exports'
    summary['exports'] = check_exports(data, output)
    return summary


def holdout(data, output, devices, monitor, methods=None, nalpha=5, tol=1e-6, max_iter=50000):
    monitor.phase = 'holdout/load'
    selected = np.random.default_rng(20260907).permutation(699)[:100]
    train_ids, test_ids = np.sort(selected[:80]), np.sort(selected[80:])
    write_json(output / 'split.json', dict(seed=20260907, train=train_ids, holdout=test_ids,
                                          index_base=0, rows_per_config=1488))
    row_ids = lambda ids: (ids[:, None] * 1488 + np.arange(1488)).ravel()
    prime = sp.load_npz(data / 'sm_prime.npz')
    training = prime[row_ids(train_ids)]
    testing = prime[row_ids(test_ids)]
    del prime
    gc.collect()
    ns = sp.block_diag([sp.load_npz(data / 'ns_harm.npz'), sp.load_npz(data / 'ns_anharm3.npz')], format='csr')
    with (data / 'force_matrix.pkl').open('rb') as stream:
        forces = pickle.load(stream)
    y_train, y_test = forces[train_ids].ravel(), forces[test_ids].ravel()
    monitor.phase = 'holdout/gpu_load'
    operator = om.TwoLevelSM(training, ns, dtype=np.float64)
    require_gpu(operator, devices)
    results = {}
    try:
        for method in (methods or ['OLS', 'RFE-OLS-TSQR']):
            label = 'holdout_' + ('rfe' if method == 'RFE-OLS-TSQR' else 'rfe_plain' if method == 'RFE' else method.lower())
            regularized = method in ('LASSO', 'ALASSO', 'RIDGE')
            optimizer = om.Optimizer(method, cv=3, rand_seed=20260907, use_gpu=True,
                                     nalpha=nalpha, tol=tol, max_iter=max_iter,
                                     standardize=regularized)
            result = audited_fit(optimizer, operator, y_train, om.Optimizer.fit,
                                 devices, output, label, monitor, config_ids=train_ids)
            test_prediction = testing @ (ns @ optimizer.results['coef'])
            result['holdout'] = force_metrics(test_prediction, y_test, 1488, test_ids)
            if "pre_debias_coef" in optimizer.results:
                before = np.asarray(optimizer.results["pre_debias_coef"])
                after = np.asarray(optimizer.results["coef"])
                if np.any(after[before == 0] != 0):
                    raise AssertionError("debias introduced features outside selected support")
                np.save(output / (label + "_pre_debias_coef.npy"), before)
                result["pre_debias"] = dict(
                    nonzero=int(np.count_nonzero(before)),
                    metrics=force_metrics(operator @ before, y_train, 1488, train_ids),
                    holdout=force_metrics(testing @ (ns @ before), y_test, 1488, test_ids))
                require_gpu(operator, devices)
            result['method'] = method
            result['algorithm'] = 'FISTA on TwoLevelSM' if method in ('LASSO', 'ALASSO') else 'iterative least squares on TwoLevelSM'
            result['selected_alpha'] = optimizer.results.get('alpha')
            write_json(output / (label + '.json'), result)
            results[label] = result
    finally:
        if operator._gpu_mv is not None:
            operator._gpu_mv.close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['full', 'holdout'])
    parser.add_argument('data', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--devices', default='0,1,2,3,4,5')
    parser.add_argument('--methods', nargs='+', choices=['OLS', 'RFE-OLS-TSQR', 'RFE', 'LASSO', 'ALASSO', 'RIDGE'])
    parser.add_argument('--nalpha', type=int, default=5)
    parser.add_argument('--tol', type=float, default=1e-6)
    parser.add_argument('--max-iter', type=int, default=50000)
    args = parser.parse_args()
    data, output = args.data.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    devices = [int(d) for d in args.devices.split(',')]
    assert 1 <= len(devices) <= 6 and len(set(devices)) == len(devices)
    assert torch.cuda.is_available() and max(devices) < torch.cuda.device_count()
    defaults = dict(PHEASY_USE_GPU='1', PHEASY_GPU_SM='1', PHEASY_GPU_SM_NGPU=str(len(devices)),
                    PHEASY_GPU_SM_DEVICES=args.devices, PHEASY_GPU_DEVICE=str(devices[0]),
                    PHEASY_SM_DTYPE='float64', PHEASY_OLS_TWOLEVEL='1',
                    PHEASY_TWOLEVEL_CACHE_T='0', PHEASY_OLS_JACOBI='1',
                    PHEASY_COL_NORM_BLOCK_BYTES='268435456',
                    PHEASY_OLS_RIDGE='0', PHEASY_OLS_ATOL='1e-8', PHEASY_OLS_BTOL='1e-8',
                    PHEASY_OLS_MAXITER='50000', PHEASY_LSQR_ATOL='1e-8',
                    PHEASY_LSQR_BTOL='1e-8', PHEASY_LSQR_MAXITER='50000',
                    PHEASY_RFE_TWOLEVEL='1', PHEASY_RFE_JACOBI='1',
                    PHEASY_RFE_N_JOBS='1', PHEASY_TSQR_STEP='0.5',
                    PHEASY_TSQR_MIN_FEATURES='13071', PHEASY_TSQR_CRITERION='cv',
                    PHEASY_CV_GROUP_SIZE='1488', PHEASY_RFE_PATIENCE='5')
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    write_json(output / 'configuration.json', dict(mode=args.mode, data=str(data),
                methods=args.methods, nalpha=args.nalpha, tol=args.tol, max_iter=args.max_iter,
                source=str(Path(om.__file__).resolve()), torch_version=torch.__version__,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                environment={key: value for key, value in os.environ.items() if key.startswith("PHEASY_")}))
    for device in devices:
        with torch.cuda.device(device):
            probe = torch.empty(0, device=device)
            del probe
            torch.cuda.reset_peak_memory_stats(device)
    with Monitor(output, devices) as monitor:
        try:
            results = (full(data, output, devices, monitor) if args.mode == 'full' else
                       holdout(data, output, devices, monitor, args.methods, args.nalpha, args.tol, args.max_iter))
            write_json(output / 'result.json', dict(status='PASS', results=results))
        except Exception:
            write_json(output / 'result.json', dict(status='FAIL', traceback=traceback.format_exc()))
            raise


if __name__ == '__main__':
    main()
