import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Tuple, Dict, Any
import random
from models import TrialResult
from des_engine import DESEngine

def run_single_trial(args: Tuple[int, float, float, float, int, int, int]) -> TrialResult:
    """
    Independent worker function executing one trial of the simulation.
    Accepts: (seed, lam, mean_service, std_service, c, max_patients, warmup_patients)
    """
    seed, lam, mean_service, std_service, c, max_patients, warmup_patients = args
    
    # Deterministic seed for trial-level reproducibility
    rng = random.Random(seed)
    
    engine = DESEngine(
        lam=lam,
        mean_service=mean_service,
        std_service=std_service,
        c=c,
        max_patients=max_patients,
        rng=rng,
        warmup_patients=warmup_patients
    )
    result = engine.run()
    result.seed = seed
    return result

def run_simulation_suite(
    n_replications: int,
    n_workers: int,
    lam: float,
    mean_service: float,
    std_service: float,
    c: int,
    max_patients: int,
    warmup_patients: int,
    mode: str = "sequential"
) -> Tuple[List[TrialResult], float]:
    """
    Orchestrates the execution of multiple independent replications of the simulation
    using sequential (standard for loop), multiprocessing (ProcessPoolExecutor),
    or multithreading (ThreadPoolExecutor) mode.
    Returns a tuple of: (results_list, elapsed_time_seconds)
    """
    # Create distinct seeds based on replication index (0, 1, ..., N-1)
    tasks = [
        (seed, lam, mean_service, std_service, c, max_patients, warmup_patients)
        for seed in range(n_replications)
    ]
    
    start_time = time.perf_counter()
    
    if mode == "sequential":
        results = [run_single_trial(task) for task in tasks]
    elif mode == "threading":
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(run_single_trial, tasks))
    elif mode == "multiprocessing":
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(run_single_trial, tasks))
    else:
        raise ValueError(f"Unsupported mode: {mode}")
        
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    
    return results, elapsed_time
