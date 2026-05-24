import os
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from typing import List, Dict, Any
from models import TrialResult

# Define a premium, cohesive color palette for our paper figures
COLORS = {
    "sequential": "#64748b",       # Slate gray
    "multiprocessing": "#0284c7",   # Sky blue
    "threading": "#f97316",         # Warm orange
    "emergency": "#ef4444",         # Red
    "regular": "#3b82f6",           # Blue
    "grid": "#f1f5f9",              # Slate-100 grid lines
    "text": "#0f172a"               # Slate-900 text
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.edgecolor": "#cbd5e1",
    "axes.labelcolor": COLORS["text"],
    "xtick.color": COLORS["text"],
    "ytick.color": COLORS["text"],
    "figure.titlesize": 13,
    "axes.titlesize": 12,
    "grid.color": COLORS["grid"],
    "grid.linestyle": "--"
})

def generate_performance_plots(
    sweep_results: Dict[str, List[Dict[str, Any]]],
    output_dir: str = "results/plots"
):
    """
    Generates high-resolution, publication-quality Speedup, Execution Time,
    and Computational Throughput curves comparing Sequential, Multiprocessing,
    and Threading execution modes.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    seq_data = sweep_results["sequential"]
    proc_data = sweep_results["multiprocessing"]
    thrd_data = sweep_results["threading"]
    
    patient_counts = [d["patient_count"] for d in seq_data]
    
    seq_times = [d["median_time"] for d in seq_data]
    proc_times = [d["median_time"] for d in proc_data]
    thrd_times = [d["median_time"] for d in thrd_data]
    
    proc_speedup = [d["speedup"] for d in proc_data]
    thrd_speedup = [d["speedup"] for d in thrd_data]
    
    seq_throughput = [d["computation_throughput"] for d in seq_data]
    proc_throughput = [d["computation_throughput"] for d in proc_data]
    thrd_throughput = [d["computation_throughput"] for d in thrd_data]
    
    # ----------------------------------------------------
    # Plot 1: Execution Time vs Patient Count
    # ----------------------------------------------------
    plt.figure(figsize=(7.5, 5.5), dpi=300)
    plt.plot(patient_counts, seq_times, marker="o", markersize=6, label="Sequential (Baseline)", 
             color=COLORS["sequential"], linewidth=2.5)
    plt.plot(patient_counts, proc_times, marker="s", markersize=6, label="Multiprocessing (4 Workers)", 
             color=COLORS["multiprocessing"], linewidth=2.5)
    plt.plot(patient_counts, thrd_times, marker="^", markersize=6, label="Threading (4 Workers - GIL-Bound)", 
             color=COLORS["threading"], linewidth=2.5)
    
    plt.title("Simulation Execution Time vs Patient Count", fontweight="bold", pad=15)
    plt.xlabel("Patient Count (N)")
    plt.ylabel("Execution Time (seconds)")
    plt.xticks(patient_counts)
    plt.grid(True)
    plt.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1")
    plt.tight_layout()
    
    exec_time_path = os.path.join(output_dir, "execution_time.png")
    plt.savefig(exec_time_path, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Execution Time Plot to: {exec_time_path}")
    
    # ----------------------------------------------------
    # Plot 2: Speedup Curves
    # ----------------------------------------------------
    plt.figure(figsize=(7.5, 5.5), dpi=300)
    
    # Ideal Linear Reference
    plt.plot(patient_counts, [4.0] * len(patient_counts), 
             label="Ideal Multi-Core Speedup (4x)", color="#2ca02c", 
             linestyle=":", linewidth=2)
    
    # Process Pool Speedup
    plt.plot(patient_counts, proc_speedup, 
             marker="s", markersize=6, label="Multiprocessing Speedup", 
             color=COLORS["multiprocessing"], linewidth=2.5)
    
    # Thread Pool Speedup
    plt.plot(patient_counts, thrd_speedup, 
             marker="^", markersize=6, label="Threading Speedup (GIL-Bound)", 
             color=COLORS["threading"], linewidth=2.5)
    
    plt.title("Parallel Speedup: Multiprocessing vs. Threading", fontweight="bold", pad=15)
    plt.xlabel("Patient Count (N)")
    plt.ylabel("Speedup ($S_p = T_{seq} / T_p$)")
    plt.xticks(patient_counts)
    plt.ylim(0, 4.5)
    plt.grid(True)
    plt.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1")
    plt.tight_layout()
    
    speedup_path = os.path.join(output_dir, "speedup_curve.png")
    plt.savefig(speedup_path, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Speedup Curve to: {speedup_path}")
    
    # ----------------------------------------------------
    # Plot 3: Computational Throughput
    # ----------------------------------------------------
    plt.figure(figsize=(7.5, 5.5), dpi=300)
    
    plt.plot(patient_counts, seq_throughput, marker="o", markersize=6, label="Sequential (Baseline)", 
             color=COLORS["sequential"], linewidth=2.5)
    plt.plot(patient_counts, proc_throughput, marker="s", markersize=6, label="Multiprocessing (4 Workers)", 
             color=COLORS["multiprocessing"], linewidth=2.5)
    plt.plot(patient_counts, thrd_throughput, marker="^", markersize=6, label="Threading (4 Workers - GIL-Bound)", 
             color=COLORS["threading"], linewidth=2.5)
    
    plt.title("Computational Throughput vs. Patient Count", fontweight="bold", pad=15)
    plt.xlabel("Patient Count (N)")
    plt.ylabel("Throughput (Patients Processed / Second)")
    plt.xticks(patient_counts)
    plt.grid(True)
    plt.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1")
    plt.tight_layout()
    
    throughput_path = os.path.join(output_dir, "computational_throughput.png")
    plt.savefig(throughput_path, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Computational Throughput Plot to: {throughput_path}")
    
    # ----------------------------------------------------
    # Plot 4: CPU Utilization Comparison
    # ----------------------------------------------------
    # Check that cpu utilization data exists in sweep results
    if "avg_cpu_utilization" in seq_data[0]:
        seq_cpu = [d["avg_cpu_utilization"] for d in seq_data]
        proc_cpu = [d["avg_cpu_utilization"] for d in proc_data]
        thrd_cpu = [d["avg_cpu_utilization"] for d in thrd_data]
        
        seq_peak_cpu = [d.get("peak_cpu_utilization", 0) for d in seq_data]
        proc_peak_cpu = [d.get("peak_cpu_utilization", 0) for d in proc_data]
        thrd_peak_cpu = [d.get("peak_cpu_utilization", 0) for d in thrd_data]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
        
        # Left subplot: Average CPU Utilization
        ax1.plot(patient_counts, seq_cpu, marker="o", markersize=6, label="Sequential (Baseline)", 
                 color=COLORS["sequential"], linewidth=2.5)
        ax1.plot(patient_counts, proc_cpu, marker="s", markersize=6, label="Multiprocessing (4 Workers)", 
                 color=COLORS["multiprocessing"], linewidth=2.5)
        ax1.plot(patient_counts, thrd_cpu, marker="^", markersize=6, label="Threading (4 Workers - GIL-Bound)", 
                 color=COLORS["threading"], linewidth=2.5)
        
        ax1.set_title("Average CPU Utilization", fontweight="bold", pad=15)
        ax1.set_xlabel("Patient Count (N)")
        ax1.set_ylabel("CPU Utilization (%)")
        ax1.set_xticks(patient_counts)
        ax1.grid(True)
        ax1.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9)
        
        # Right subplot: Peak CPU Utilization
        ax2.plot(patient_counts, seq_peak_cpu, marker="o", markersize=6, label="Sequential (Baseline)", 
                 color=COLORS["sequential"], linewidth=2.5)
        ax2.plot(patient_counts, proc_peak_cpu, marker="s", markersize=6, label="Multiprocessing (4 Workers)", 
                 color=COLORS["multiprocessing"], linewidth=2.5)
        ax2.plot(patient_counts, thrd_peak_cpu, marker="^", markersize=6, label="Threading (4 Workers - GIL-Bound)", 
                 color=COLORS["threading"], linewidth=2.5)
        
        ax2.set_title("Peak CPU Utilization", fontweight="bold", pad=15)
        ax2.set_xlabel("Patient Count (N)")
        ax2.set_ylabel("CPU Utilization (%)")
        ax2.set_xticks(patient_counts)
        ax2.grid(True)
        ax2.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=9)
        
        fig.suptitle("CPU Utilization: Sequential vs. Multiprocessing vs. Threading", 
                      fontweight="bold", fontsize=13, y=1.02)
        plt.tight_layout()
        
        cpu_util_path = os.path.join(output_dir, "cpu_utilization.png")
        fig.savefig(cpu_util_path, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Saved CPU Utilization Plot to: {cpu_util_path}")

def generate_queueing_plots(
    queueing_data: Dict[int, TrialResult],
    output_dir: str = "results/plots"
):
    """
    Generates queueing theory analysis plots: Patient Wait Times broken down by priority.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    patient_counts = sorted(queueing_data.keys())
    avg_waits = [queueing_data[n].avg_wait_time for n in patient_counts]
    emergency_waits = [queueing_data[n].avg_wait_time_emergency for n in patient_counts]
    regular_waits = [queueing_data[n].avg_wait_time_regular for n in patient_counts]
    
    plt.figure(figsize=(7.5, 5.5), dpi=300)
    
    plt.plot(patient_counts, avg_waits, marker="o", markersize=6, label="Overall Average", 
             color=COLORS["text"], linewidth=2.5)
    plt.plot(patient_counts, emergency_waits, marker="s", markersize=6, label="Emergency (High Priority)", 
             color=COLORS["emergency"], linewidth=2.5)
    plt.plot(patient_counts, regular_waits, marker="^", markersize=6, label="Regular (Normal Priority)", 
             color=COLORS["regular"], linewidth=2.5)
    
    plt.title("Queue Wait Time by Patient Priority Class", fontweight="bold", pad=15)
    plt.xlabel("Patient Count (N)")
    plt.ylabel("Average Waiting Time (minutes)")
    plt.xticks(patient_counts)
    plt.grid(True)
    plt.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1")
    plt.tight_layout()
    
    wait_time_path = os.path.join(output_dir, "waiting_time_by_priority.png")
    plt.savefig(wait_time_path, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Wait Time Priority Plot to: {wait_time_path}")

def generate_qq_plots(
    inter_arrival_times: List[float],
    service_times: List[float],
    lam: float,
    mean_service: float,
    std_service: float,
    output_dir: str = "results/plots"
):
    """
    Generates Q-Q (Quantile-Quantile) plots for inter-arrival and service times
    against theoretical distributions to visually validate random generation models.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # ----------------------------------------------------
    # Inter-Arrival Times Q-Q Plot
    # ----------------------------------------------------
    plt.figure(figsize=(7.5, 5.5), dpi=300)
    stats.probplot(inter_arrival_times, dist="expon", sparams=(0, 1.0/lam), plot=plt)
    
    plt.title("Q-Q Plot: Inter-Arrival Times vs. Exp(λ=5)", fontweight="bold", pad=15)
    plt.xlabel("Theoretical Exponential Quantiles")
    plt.ylabel("Sample Ordered Values (minutes)")
    
    ax = plt.gca()
    if len(ax.lines) >= 2:
        ax.lines[0].set_color(COLORS["multiprocessing"])  # Sample points
        ax.lines[0].set_markerfacecolor(COLORS["multiprocessing"])
        ax.lines[0].set_markeredgecolor(COLORS["multiprocessing"])
        ax.lines[1].set_color(COLORS["emergency"])  # Reference fit line
        ax.lines[1].set_linewidth(2)
        
    plt.grid(True)
    plt.tight_layout()
    
    qq_interarrival_path = os.path.join(output_dir, "qq_interarrival.png")
    plt.savefig(qq_interarrival_path, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Inter-arrival Q-Q Plot to: {qq_interarrival_path}")
    
    # ----------------------------------------------------
    # Service Times Q-Q Plot
    # ----------------------------------------------------
    plt.figure(figsize=(7.5, 5.5), dpi=300)
    stats.probplot(service_times, dist="norm", sparams=(mean_service, std_service), plot=plt)
    
    plt.title(f"Q-Q Plot: Service Times vs. Normal({mean_service:.1f}, {std_service:.1f})", fontweight="bold", pad=15)
    plt.xlabel("Theoretical Normal Quantiles")
    plt.ylabel("Sample Ordered Values (minutes)")
    
    ax = plt.gca()
    if len(ax.lines) >= 2:
        ax.lines[0].set_color(COLORS["threading"])  # Sample points
        ax.lines[0].set_markerfacecolor(COLORS["threading"])
        ax.lines[0].set_markeredgecolor(COLORS["threading"])
        ax.lines[1].set_color(COLORS["emergency"])  # Reference fit line
        ax.lines[1].set_linewidth(2)
        
    plt.grid(True)
    plt.tight_layout()
    
    qq_service_path = os.path.join(output_dir, "qq_service.png")
    plt.savefig(qq_service_path, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved Service Q-Q Plot to: {qq_service_path}")
