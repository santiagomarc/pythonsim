import numpy as np
import scipy.stats as stats
from typing import List, Dict, Any, Tuple
from models import TrialResult

def validate_inter_arrival_times(data: List[float], rate: float) -> Dict[str, Any]:
    """
    Validates if inter-arrival times follow an exponential distribution with parameter 'rate'
    using Kolmogorov-Smirnov (KS) test.
    """
    if not data:
        return {"status": "No data"}
    
    scale = 1.0 / rate
    
    # Kolmogorov-Smirnov Test
    # H0: The data follows the theoretical exponential distribution with parameter 'rate'
    ks_res = stats.kstest(data, 'expon', args=(0, scale))
    
    return {
        "ks_stat": float(ks_res.statistic),
        "ks_p_value": float(ks_res.pvalue),
        "ks_valid": ks_res.pvalue > 0.05,
        "sample_size": len(data),
        "sample_mean": float(np.mean(data)),
        "expected_mean": scale
    }

def validate_service_times(data: List[float], mean: float, std: float) -> Dict[str, Any]:
    """
    Validates if service times follow a Normal distribution with specified mean and std
    using Shapiro-Wilk test (normality) and Kolmogorov-Smirnov test (exact parameters).
    """
    if not data:
        return {"status": "No data"}
    
    # 1. Kolmogorov-Smirnov Test against Normal(mean, std)
    ks_res = stats.kstest(data, 'norm', args=(mean, std))
    
    # 2. Shapiro-Wilk Test for Normality (limited to max 5000 samples)
    shapiro_res = stats.shapiro(data[:5000]) if len(data) >= 3 else None
    
    return {
        "ks_stat": float(ks_res.statistic),
        "ks_p_value": float(ks_res.pvalue),
        "ks_valid": ks_res.pvalue > 0.05,
        "shapiro_stat": float(shapiro_res.statistic) if shapiro_res else 1.0,
        "shapiro_p_value": float(shapiro_res.pvalue) if shapiro_res else 1.0,
        "shapiro_valid": (shapiro_res.pvalue > 0.05) if shapiro_res else True,
        "sample_size": len(data),
        "sample_mean": float(np.mean(data)),
        "sample_std": float(np.std(data, ddof=1)) if len(data) > 1 else 0.0,
        "expected_mean": mean,
        "expected_std": std
    }

def analyze_convergence(results_by_n: Dict[int, List[TrialResult]]) -> Dict[int, Dict[str, Any]]:
    """
    Computes per-trial size summary statistics (mean, median, std, 95% CI, CV) for convergence analysis.
    As patient count N increases, system metrics (waiting time, throughput) should stabilize.
    """
    convergence_summary = {}
    
    metrics = ["avg_wait_time", "avg_system_time", "avg_queue_length", "server_utilization"]
    
    for n, trials in results_by_n.items():
        n_trials = len(trials)
        if n_trials == 0:
            continue
            
        summary = {}
        
        for metric in metrics:
            values = [getattr(t, metric) for t in trials]
            mean_val = np.mean(values)
            median_val = np.median(values)
            std_val = np.std(values, ddof=1) if n_trials > 1 else 0.0
            cv_val = std_val / mean_val if mean_val > 0 and n_trials > 1 else 0.0
            
            if n_trials >= 2:
                sem = std_val / np.sqrt(n_trials)
                ci_lower, ci_upper = stats.t.interval(0.95, df=n_trials-1, loc=mean_val, scale=sem)
            else:
                ci_lower, ci_upper = mean_val, mean_val
                
            summary[metric] = {
                "mean": float(mean_val),
                "median": float(median_val),
                "std": float(std_val),
                "cv": float(cv_val),
                "ci_95": (float(ci_lower), float(ci_upper))
            }
            
        # Also compute throughput separately as it's a key comparison metric
        # Throughput = total_patients_served / sim_duration
        throughput_values = [t.total_patients_served / t.sim_duration for t in trials if t.sim_duration > 0]
        if throughput_values:
            mean_tp = np.mean(throughput_values)
            std_tp = np.std(throughput_values, ddof=1) if len(throughput_values) > 1 else 0.0
            cv_tp = std_tp / mean_tp if mean_tp > 0 and len(throughput_values) > 1 else 0.0
            
            if len(throughput_values) >= 2:
                sem_tp = std_tp / np.sqrt(len(throughput_values))
                ci_l_tp, ci_u_tp = stats.t.interval(0.95, df=len(throughput_values)-1, loc=mean_tp, scale=sem_tp)
            else:
                ci_l_tp, ci_u_tp = mean_tp, mean_tp
                
            summary["throughput"] = {
                "mean": float(mean_tp),
                "median": float(np.median(throughput_values)),
                "std": float(std_tp),
                "cv": float(cv_tp),
                "ci_95": (float(ci_l_tp), float(ci_u_tp))
            }
        else:
            summary["throughput"] = {
                "mean": 0.0, "median": 0.0, "std": 0.0, "cv": 0.0, "ci_95": (0.0, 0.0)
            }
            
        convergence_summary[n] = summary
        
    return convergence_summary
