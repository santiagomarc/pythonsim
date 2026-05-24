import heapq
import random
from collections import deque
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from models import Event, EventType, Patient, TrialResult

class DESEngine:
    def __init__(
        self,
        lam: float,
        mean_service: float,
        std_service: float,
        c: int,
        max_patients: int,
        rng: random.Random,
        warmup_patients: int = 20,
        verbose: bool = False
    ):
        self.lam = lam
        self.mean_service = mean_service
        self.std_service = std_service
        self.c = c
        self.max_patients = max_patients
        self.rng = rng
        self.warmup_patients = warmup_patients
        self.verbose = verbose

        # Simulation State
        self.current_time = 0.0
        self.event_queue: List[Event] = []
        
        # Simple FIFO waiting queue (no priority — all patients are equal)
        self.waiting_queue: deque = deque()
        
        self.counters_busy = [False] * c  # Track which registration counters are busy
        self.patients: Dict[int, Patient] = {}
        
        # Stat Accumulators
        self.patients_served = 0
        self.last_state_change_time = 0.0
        
        # Time-weighted statistics
        self.area_queue_length = 0.0
        self.area_counter_utilization = 0.0
        
        # Warmup transition settings
        if self.warmup_patients == 0:
            self.warmup_reset_done = True
            self.stats_start_time = 0.0
        else:
            self.warmup_reset_done = False
            self.stats_start_time = 0.0
            
        # Pools for storing post-warmup results
        self.recorded_patients_served: List[Patient] = []
        self.recorded_inter_arrival_times: List[float] = []
        self.recorded_service_times: List[float] = []

    def get_truncated_normal_service_time(self) -> float:
        """Generates service time from Truncated Normal distribution (min=1 min) using standard library Random."""
        while True:
            val = self.rng.normalvariate(self.mean_service, self.std_service)
            if val >= 1.0:
                return val

    def schedule_event(self, time: float, event_type: EventType, patient_id: int, server_id: Optional[int] = None):
        event = Event(time, event_type, patient_id, server_id)
        heapq.heappush(self.event_queue, event)

    def _get_idle_counter(self) -> Optional[int]:
        for i, busy in enumerate(self.counters_busy):
            if not busy:
                return i
        return None

    def _reset_warmup_stats(self):
        """Clears all accumulated stats and resets integration start to remove initialization bias."""
        self.area_queue_length = 0.0
        self.area_counter_utilization = 0.0
        self.stats_start_time = self.current_time
        
        self.recorded_patients_served.clear()
        self.recorded_inter_arrival_times.clear()
        self.recorded_service_times.clear()
        
        self.warmup_reset_done = True

    def run(self) -> TrialResult:
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ==== TRIAL 1: {self.max_patients} patients ====")
            
        # Schedule first arrival
        first_interarrival = self.rng.expovariate(self.lam)
        self.schedule_event(first_interarrival, EventType.ARRIVAL, patient_id=1)
        
        next_patient_id = 2

        # Run until we have served max_patients + warmup_patients
        target_patients_served = self.max_patients + self.warmup_patients
        
        while self.event_queue and self.patients_served < target_patients_served:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time
            
            # Update time-weighted accumulators
            time_diff = self.current_time - self.last_state_change_time
            if time_diff > 0:
                self.area_queue_length += len(self.waiting_queue) * time_diff
                self.area_counter_utilization += sum(self.counters_busy) * time_diff
            self.last_state_change_time = self.current_time

            if event.event_type == EventType.ARRIVAL:
                if self.verbose:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Patient {event.patient_id} arrived.")
                    
                # All patients are equal — no priority classification
                patient = Patient(patient_id=event.patient_id, arrival_time=self.current_time)
                self.patients[patient.patient_id] = patient
                
                # Check for idle registration counter
                idle_counter_id = self._get_idle_counter()
                if idle_counter_id is not None:
                    # Serve immediately
                    self.counters_busy[idle_counter_id] = True
                    patient.service_start_time = self.current_time
                    patient.counter_id = idle_counter_id
                    
                    if self.verbose:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Patient {patient.patient_id} done registration and start processing.")
                    
                    service_dur = self.get_truncated_normal_service_time()
                    patient.service_duration = service_dur
                    
                    if self.warmup_reset_done:
                        self.recorded_service_times.append(service_dur)
                        
                    self.schedule_event(
                        self.current_time + service_dur,
                        EventType.DEPARTURE,
                        patient.patient_id,
                        idle_counter_id
                    )
                else:
                    # Place in FIFO waiting queue
                    self.waiting_queue.append(patient.patient_id)

                # Schedule next arrival
                inter_arr = self.rng.expovariate(self.lam)
                self.schedule_event(self.current_time + inter_arr, EventType.ARRIVAL, next_patient_id)
                
                if self.warmup_reset_done:
                    self.recorded_inter_arrival_times.append(inter_arr)
                    
                next_patient_id += 1

            elif event.event_type == EventType.DEPARTURE:
                patient = self.patients.get(event.patient_id)
                if patient:
                    patient.departure_time = self.current_time
                    self.patients_served += 1
                    
                    # Record patient metrics if post-warmup
                    if self.warmup_reset_done:
                        self.recorded_patients_served.append(patient)
                    
                    # Handle warmup transition
                    if not self.warmup_reset_done and self.patients_served == self.warmup_patients:
                        self._reset_warmup_stats()

                    # Free this counter
                    counter_id = event.server_id
                    self.counters_busy[counter_id] = False
                    
                    # If queue is not empty, serve next patient (FIFO order)
                    if self.waiting_queue:
                        pid = self.waiting_queue.popleft()
                        next_patient = self.patients[pid]
                        
                        self.counters_busy[counter_id] = True
                        next_patient.service_start_time = self.current_time
                        next_patient.counter_id = counter_id
                        
                        if self.verbose:
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] Patient {next_patient.patient_id} done registration and start processing.")
                        
                        service_dur = self.get_truncated_normal_service_time()
                        next_patient.service_duration = service_dur
                        
                        if self.warmup_reset_done:
                            self.recorded_service_times.append(service_dur)
                            
                        self.schedule_event(
                            self.current_time + service_dur,
                            EventType.DEPARTURE,
                            next_patient.patient_id,
                            counter_id
                        )

        # Post-simulation metric calculations
        sim_duration_post_warmup = self.current_time - self.stats_start_time
        if sim_duration_post_warmup <= 0:
            sim_duration_post_warmup = 1e-9

        avg_q_len = self.area_queue_length / sim_duration_post_warmup
        avg_util = self.area_counter_utilization / (self.c * sim_duration_post_warmup)
        
        # Filter patients served post-warmup
        wait_times = [p.wait_time for p in self.recorded_patients_served if p.wait_time is not None]
        system_times = [p.system_time for p in self.recorded_patients_served if p.system_time is not None]
        
        avg_wait = sum(wait_times) / len(wait_times) if wait_times else 0.0
        avg_sys = sum(system_times) / len(system_times) if system_times else 0.0

        # Pick a random seed for the returned result
        seed_val = self.rng.randint(0, 10000000)

        return TrialResult(
            seed=seed_val,
            avg_wait_time=avg_wait,
            avg_system_time=avg_sys,
            avg_queue_length=avg_q_len,
            server_utilization=avg_util,
            sim_duration=sim_duration_post_warmup,
            total_patients_served=len(self.recorded_patients_served),
            inter_arrival_times=self.recorded_inter_arrival_times,
            service_times=self.recorded_service_times
        )
