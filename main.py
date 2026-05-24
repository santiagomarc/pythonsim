#!/usr/bin/env python3
import os
import sys
import time
import csv
import random
import threading
import multiprocessing
import psutil
import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

# Ensure current directory is in Python PATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# MULTIPROCESSING WORKER FUNCTIONS (Top-level for pickle safety on macOS)
# ============================================================

def worker_process(
    worker_id: int, 
    queue: multiprocessing.Queue, 
    results_queue: multiprocessing.Queue, 
    arrivals_finished: multiprocessing.Event,
    mean_service: float,
    std_service: float,
    start_time: float
):
    """
    Parallel worker process representing a registration counter.
    Pulls patients from the shared multiprocessing Queue and processes them.
    """
    # Seed the local random generator based on PID and time to ensure thread-safety
    random.seed(os.getpid() + worker_id)
    
    while True:
        # Check if we should exit: arrivals are done and the queue is empty
        if queue.empty() and arrivals_finished.is_set():
            break
            
        try:
            # Try to get a patient from the queue with a tiny timeout to prevent deadlocks
            patient_id, arrival_time = queue.get(timeout=0.05)
        except Exception:
            continue
            
        # Record service start time
        service_start_time = time.time() - start_time
        waiting_time = service_start_time - arrival_time
        
        # Format printing safely
        sys.stdout.write(f"\nCounter {worker_id} assigned to Patient {patient_id}\n")
        sys.stdout.flush()
        
        # Generate service duration from Truncated Normal distribution
        service_time = random.normalvariate(mean_service, std_service)
        if service_time < 0.1: # Truncate service times at 0.1 seconds (min = 50s real time)
            service_time = 0.1
            
        # Simulate processing via sleeping
        time.sleep(service_time)
        
        # Record departure metrics
        departure_time = time.time() - start_time
        turnaround_time = departure_time - arrival_time
        
        # Push results back to main process
        results_queue.put((
            patient_id, 
            arrival_time, 
            service_start_time, 
            waiting_time, 
            service_time, 
            departure_time, 
            turnaround_time, 
            worker_id
        ))
        
        # Formatted output matching C++ console format
        output = (
            f"Patient {patient_id} processed by Counter {worker_id}\n"
            f"Arrival Time       : {arrival_time:.2f} sec\n"
            f"Service Start Time : {service_start_time:.2f} sec\n"
            f"Waiting Time       : {waiting_time:.2f} sec\n"
            f"Service Duration   : {service_time:.2f} sec\n"
            f"Departure Time     : {departure_time:.2f} sec\n"
            f"Turnaround Time    : {turnaround_time:.2f} sec\n"
            f"----------------------------------\n"
        )
        sys.stdout.write(output)
        sys.stdout.flush()


def arrival_generator(
    total_patients: int, 
    arrival_interval: float, 
    queue: multiprocessing.Queue, 
    arrivals_finished: multiprocessing.Event,
    start_time: float
):
    """
    Generates patient arrivals at fixed intervals and places them in the queue.
    """
    for i in range(1, total_patients + 1):
        now = time.time() - start_time
        queue.put((i, now))
        
        sys.stdout.write(f"Patient {i:>3d} arrived at {now:.2f} sec\n")
        sys.stdout.flush()
        
        time.sleep(arrival_interval)
        
    arrivals_finished.set()


def monitor_cpu(stop_event: threading.Event, cpu_samples: List[float]):
    """
    Background thread that samples system-wide and process-tree CPU utilization.
    Accumulates CPU consumption of the main process and all parallel multiprocessing children.
    """
    parent = psutil.Process(os.getpid())
    while not stop_event.is_set():
        try:
            # Get CPU percent for the main process
            cpu_pct = parent.cpu_percent(interval=None)
            
            # Accumulate CPU percent for all multiprocessing children
            for child in parent.children(recursive=True):
                try:
                    cpu_pct += child.cpu_percent(interval=None)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            cpu_samples.append(cpu_pct)
        except Exception:
            pass
        time.sleep(0.1)


# ============================================================
# MAIN SIMULATION ENGINE & PLOTTING
# ============================================================

def generate_performance_plots(results: List[Tuple], cpu_samples: List[float], output_dir: str):
    """
    Generates high-resolution scientific plots representing the simulation run.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Unpack results:
    # 0: pid, 1: arr, 2: start, 3: wait, 4: dur, 5: dep, 6: turn, 7: worker
    results = sorted(results, key=lambda x: x[0])
    pids = [r[0] for r in results]
    waits = [r[3] for r in results]
    durations = [r[4] for r in results]
    turnarounds = [r[6] for r in results]
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Color palette
    primary_color = "#3B82F6"  # Premium Blue
    secondary_color = "#10B981" # Emerald Green
    accent_color = "#EF4444"    # Coral Red
    dark_bg = "#1F2937"
    
    # Plot 1: Patient Waiting Times
    axs[0, 0].plot(pids, waits, color=primary_color, marker='o', linestyle='-', linewidth=2, markersize=4)
    axs[0, 0].set_title("Patient Waiting Times in Queue", fontsize=12, fontweight='bold', pad=10)
    axs[0, 0].set_xlabel("Patient ID", fontsize=10)
    axs[0, 0].set_ylabel("Wait Time (seconds)", fontsize=10)
    axs[0, 0].fill_between(pids, waits, color=primary_color, alpha=0.15)
    
    # Plot 2: Wait vs Service Times (Stacked bar preview)
    width = 0.35
    axs[0, 1].bar(pids[:20], waits[:20], width, label='Wait Time', color=accent_color, alpha=0.8)
    axs[0, 1].bar(pids[:20], durations[:20], width, bottom=waits[:20], label='Service Time', color=secondary_color, alpha=0.8)
    axs[0, 1].set_title("Timeline Breakdown (First 20 Patients)", fontsize=12, fontweight='bold', pad=10)
    axs[0, 1].set_xlabel("Patient ID", fontsize=10)
    axs[0, 1].set_ylabel("Time (seconds)", fontsize=10)
    axs[0, 1].legend()
    
    # Plot 3: CPU Utilization Over Time
    time_points = np.linspace(0, len(cpu_samples) * 0.1, len(cpu_samples))
    axs[1, 0].plot(time_points, cpu_samples, color="#8B5CF6", linewidth=2)
    axs[1, 0].set_title("Process Tree CPU Utilization", fontsize=12, fontweight='bold', pad=10)
    axs[1, 0].set_xlabel("Elapsed Time (seconds)", fontsize=10)
    axs[1, 0].set_ylabel("CPU Usage (%)", fontsize=10)
    axs[1, 0].fill_between(time_points, cpu_samples, color="#8B5CF6", alpha=0.15)
    
    # Plot 4: Distribution of Turnaround Times
    axs[1, 1].hist(turnarounds, bins=15, color="#EC4899", edgecolor='black', alpha=0.7)
    axs[1, 1].set_title("Distribution of Turnaround Times", fontsize=12, fontweight='bold', pad=10)
    axs[1, 1].set_xlabel("Turnaround Time (seconds)", fontsize=10)
    axs[1, 1].set_ylabel("Frequency", fontsize=10)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "python_queue_performance.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"📊 Premium performance plot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Real-Time Multiprocessing Queue Simulation (Python)")
    parser.add_argument("--patients", type=int, default=100, help="Total number of patients/jobs to simulate")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes (registration counters)")
    parser.add_argument("--output", type=str, default="results", help="Directory to save plots and CSV logs")
    args = parser.parse_args()

    # Time scaling factor matches C++ (1/500):
    # 12 real seconds inter-arrival -> 24 ms.
    # 600 real seconds service -> 1200 ms.
    # 120 real seconds service std dev -> 240 ms.
    scale_factor = 500.0
    arrival_interval = 0.024 # 24 ms
    mean_service = 1.2       # 1200 ms
    std_service = 0.24       # 240 ms

    # Initialize shared multiprocessing constructs
    queue = multiprocessing.Queue()
    results_queue = multiprocessing.Queue()
    arrivals_finished = multiprocessing.Event()

    print("=========================================")
    print("PYTHON MULTIPROCESSING QUEUE SIMULATION  ")
    print("=========================================")
    print(f"Patients (Jobs)      : {args.patients}")
    print(f"Worker Processes     : {args.workers}")
    print(f"Arrival Interval     : {arrival_interval * 1000:.0f} ms (scaled 1/500 from 12s)")
    print(f"Service mean/std     : {mean_service * 1000:.0f} / {std_service * 1000:.0f} ms (scaled 1/500 from 10 min)")
    print(f"Traffic Intensity (ρ): 12.50 (Overloaded Queue)")
    print("=========================================\n")

    # Prime CPU utilization counter
    psutil.cpu_percent(interval=None)

    # Start CPU sampling in background
    cpu_samples = []
    stop_cpu_event = threading.Event()
    cpu_thread = threading.Thread(target=monitor_cpu, args=(stop_cpu_event, cpu_samples), daemon=True)
    cpu_thread.start()

    simulation_start = time.time()

    # Launch Parallel Worker Processes
    processes = []
    for w in range(args.workers):
        p = multiprocessing.Process(
            target=worker_process,
            args=(w, queue, results_queue, arrivals_finished, mean_service, std_service, simulation_start)
        )
        p.start()
        processes.append(p)

    # Launch Arrival Generator Thread
    arrival_thread = threading.Thread(
        target=arrival_generator,
        args=(args.patients, arrival_interval, queue, arrivals_finished, simulation_start)
    )
    arrival_thread.start()

    # Wait for completion
    arrival_thread.join()
    for p in processes:
        p.join()

    simulation_end = time.time()
    total_execution_time = simulation_end - simulation_start

    # Stop CPU monitoring
    stop_cpu_event.set()
    cpu_thread.join(timeout=1.0)

    # Drain results queue
    results = []
    total_waiting_time = 0.0
    total_turnaround_time = 0.0
    total_worker_busy_time = 0.0
    
    while not results_queue.empty():
        r = results_queue.get()
        results.append(r)
        total_waiting_time += r[3] # wait time
        total_worker_busy_time += r[4] # service dur
        total_turnaround_time += r[6] # turnaround time

    processed_patients = len(results)
    if processed_patients == 0:
        print("Error: No patients processed. Exiting.")
        sys.exit(1)

    # Core statistics calculations
    avg_wait = total_waiting_time / processed_patients
    avg_turnaround = total_turnaround_time / processed_patients
    throughput = processed_patients / total_execution_time
    avg_cpu = np.mean(cpu_samples) if cpu_samples else 0.0

    # Scale back to real-world minutes (scaling factor is 1/500)
    avg_wait_real_min = (avg_wait * scale_factor) / 60.0
    avg_turnaround_real_min = (avg_turnaround * scale_factor) / 60.0
    
    # Calculate Counter Utilization
    worker_util = (total_worker_busy_time / (args.workers * total_execution_time)) * 100.0

    # Write to CSV log
    os.makedirs(args.output, exist_ok=True)
    csv_path = os.path.join(args.output, "timing_results.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "arrival_sec", "service_start_sec", "wait_sec", "service_sec", "departure_sec", "turnaround_sec", "counter_id"])
        for r in sorted(results, key=lambda x: x[0]):
            writer.writerow(r)

    # Print Summary matching C++ formatting exactly
    print("\n===== SIMULATION SUMMARY =====\n")
    print(f"Total Patients Processed : {processed_patients}")
    print(f"Counters/Workers Used    : {args.workers}")
    print(f"Total Execution Time     : {total_execution_time:.2f} sec (scaled: {(total_execution_time * scale_factor) / 60.0:.2f} real min)")
    print(f"Average Waiting Time     : {avg_wait:.2f} sec (scaled: {avg_wait_real_min:.2f} real min)")
    print(f"Average Turnaround Time  : {avg_turnaround:.2f} sec (scaled: {avg_turnaround_real_min:.2f} real min)")
    print(f"Throughput               : {throughput:.2f} patients/sec (scaled: {throughput * (60.0 / scale_factor):.2f} patients/real min)")
    print(f"Worker Utilization (ρ)   : {worker_util:.2f} %")
    print(f"System CPU Utilization   : {avg_cpu:.2f} %")
    print("\n=========================================\n")

    # Generate visual plots
    generate_performance_plots(results, cpu_samples, args.output)


if __name__ == "__main__":
    # Fix spawn issue on macOS
    multiprocessing.freeze_support()
    main()
