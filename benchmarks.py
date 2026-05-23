import os
import csv
import statistics
import time
from typing import List, Dict, Any, Tuple
import psutil
from parallel_runner import run_simulation_suite

def get_total_memory_usage() -> float:
    """
    Measures the current memory usage of the parent process and all active child processes.
    Returns memory in Megabytes (MB).
    """
    try:
        parent = psutil.Process(os.getpid())
        total_mem = parent.memory_info().rss
        # Accumulate memory for child processes (multiprocessing pool)
        for child in parent.children(recursive=True):
            try:
                total_mem += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total_mem / (1024 * 1024)  # Convert bytes to MB
    except Exception:
        return 0.0

def run_performance_sweep(
    n_replications: int,
    n_workers: int,
    patient_counts: List[int],
    lam: float,
    mean_service: float,
    std_service: float,
    c: int,
    warmup_patients: int,
    repetitions: int = 3,
    results_dir: str = "results"
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Runs a performance sweep sweeping over patient counts [100, 300, 500]
    across three execution modes: Sequential (baseline), Multiprocessing, and Threading.
    Measures elapsed time, calculates speedup, efficiency, memory, and computation throughput,
    and saves results to CSV.
    """
    os.makedirs(results_dir, exist_ok=True)
    csv_filename = os.path.join(results_dir, "timing_results.csv")
    
    sweep_results = {
        "sequential": [],
        "multiprocessing": [],
        "threading": []
    }
    
    modes = ["sequential", "multiprocessing", "threading"]
    
    print(f"\n==========================================")
    print(f"Starting Performance Benchmark Sweep")
    print(f"Parameters: workers={n_workers}, reps={repetitions}, replications={n_replications}")
    print(f"==========================================")
    
    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "mode", "patient_count", "replications", "runs", "raw_times", 
            "median_time", "speedup", "efficiency", "peak_memory_mb", "computation_throughput"
        ])
        
        for n_patients in patient_counts:
            print(f"\n--- Patient Count: {n_patients} Patients ---")
            
            # We will use sequential median time as the baseline for this patient count
            sequential_median = None
            
            # We enforce running sequential first to establish the baseline
            ordered_modes = ["sequential", "multiprocessing", "threading"]
            
            for mode in ordered_modes:
                print(f"Running mode: {mode.upper()}...")
                raw_times = []
                peak_memory_observed = 0.0
                
                for rep in range(repetitions):
                    mem_before = get_total_memory_usage()
                    
                    # Run the simulation suite
                    # Mode mappings to run_simulation_suite modes
                    _, elapsed = run_simulation_suite(
                        n_replications=n_replications,
                        n_workers=n_workers,
                        lam=lam,
                        mean_service=mean_service,
                        std_service=std_service,
                        c=c,
                        max_patients=n_patients,
                        warmup_patients=warmup_patients,
                        mode=mode
                    )
                    
                    mem_after = get_total_memory_usage()
                    peak_memory_observed = max(peak_memory_observed, mem_before, mem_after)
                    raw_times.append(elapsed)
                    
                    print(f"  Run {rep + 1}/{repetitions}: {elapsed:.4f}s | Mem: {peak_memory_observed:.2f} MB")
                
                median_time = statistics.median(raw_times)
                if mode == "sequential":
                    sequential_median = median_time
                
                # Compute Speedup and Efficiency
                baseline = sequential_median if sequential_median is not None else median_time
                speedup = baseline / median_time if median_time > 0 else 1.0
                
                # In multiprocessing or threading, we divide by n_workers. Sequential has 1 worker.
                effective_workers = n_workers if mode in ["multiprocessing", "threading"] else 1
                efficiency = speedup / effective_workers
                
                # Computation throughput: total patients processed across all replications per second
                comp_throughput = (n_replications * n_patients) / median_time if median_time > 0 else 0.0
                
                result_entry = {
                    "patient_count": n_patients,
                    "replications": n_replications,
                    "runs": repetitions,
                    "raw_times": [round(t, 5) for t in raw_times],
                    "median_time": round(median_time, 5),
                    "speedup": round(speedup, 4),
                    "efficiency": round(efficiency, 4),
                    "peak_memory_mb": round(peak_memory_observed, 2),
                    "computation_throughput": round(comp_throughput, 2)
                }
                
                sweep_results[mode].append(result_entry)
                
                # Write to CSV
                writer.writerow([
                    mode,
                    n_patients,
                    n_replications,
                    repetitions,
                    str(result_entry["raw_times"]),
                    result_entry["median_time"],
                    result_entry["speedup"],
                    result_entry["efficiency"],
                    result_entry["peak_memory_mb"],
                    result_entry["computation_throughput"]
                ])
                
                print(f"👉 {mode.upper():15s} | Median: {median_time:.4f}s | Speedup: {speedup:.2f}x | Efficiency: {efficiency:.2%} | Throughput: {comp_throughput:.1f} pat/s | Peak Mem: {peak_memory_observed:.1f} MB")
                
    return sweep_results
