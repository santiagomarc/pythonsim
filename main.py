import os
import sys
import psutil
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import TrialResult
from des_engine import DESEngine

def main():
    lam = 5.0
    mean_service = 10.0
    std_service = 2.0
    c = 4
    
    patient_counts = [100, 300, 500]
    
    output_lines = []
    
    for i, n in enumerate(patient_counts, 1):
        # We start tracking CPU utilization
        psutil.cpu_percent(interval=None)
        
        engine = DESEngine(
            lam=lam,
            mean_service=mean_service,
            std_service=std_service,
            c=c,
            max_patients=n,
            rng=random.Random(42),
            warmup_patients=0
        )
        
        result = engine.run()
        
        cpu_util = psutil.cpu_percent(interval=None)
        
        throughput = result.total_patients_served / result.sim_duration if result.sim_duration > 0 else 0
        
        block = (
            f"===== TRIAL {i}: {n} Patients =====\n"
            f"Total Simulation Time   : {result.sim_duration:.2f} min\n"
            f"Throughput              : {throughput:.2f} patients/min\n"
            f"Avg Queue Wait Time     : {result.avg_wait_time:.2f} min\n"
            f"Avg Service Time        : {result.avg_service_time:.2f} min\n"
            f"CPU Utilization         : {cpu_util:.1f}%\n"
        )
        for counter_idx in range(c):
            block += f"Counter {counter_idx + 1} Utilization   : {result.counter_utilizations[counter_idx] * 100:.1f}%\n"
            
        output_lines.append(block)
        
    final_output = "\n".join(output_lines)
    print(final_output, end="")
    
    # Write to results.txt
    with open("results.txt", "w") as f:
        f.write(final_output)

if __name__ == "__main__":
    main()
