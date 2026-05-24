from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional

class EventType(Enum):
    ARRIVAL = 1
    DEPARTURE = 2

@dataclass(order=False)
class Event:
    time: float
    event_type: EventType
    patient_id: int
    server_id: Optional[int] = None  # In hospital context, server_id is counter_id

    def __lt__(self, other: "Event") -> bool:
        # Sort primarily by time. In case of ties, prioritize departures over arrivals, then by patient_id
        if abs(self.time - other.time) < 1e-9:
            if self.event_type != other.event_type:
                return self.event_type == EventType.DEPARTURE
            return self.patient_id < other.patient_id
        return self.time < other.time

@dataclass
class Patient:
    patient_id: int
    arrival_time: float
    service_start_time: Optional[float] = None
    departure_time: Optional[float] = None
    service_duration: Optional[float] = None
    counter_id: Optional[int] = None  # ID of the registration counter serving this patient

    @property
    def wait_time(self) -> Optional[float]:
        if self.service_start_time is not None:
            return self.service_start_time - self.arrival_time
        return None

    @property
    def system_time(self) -> Optional[float]:
        if self.departure_time is not None:
            return self.departure_time - self.arrival_time
        return None

@dataclass
class TrialResult:
    seed: int
    avg_wait_time: float
    avg_system_time: float
    avg_queue_length: float
    server_utilization: float  # counter utilization
    sim_duration: float
    total_patients_served: int
    
    # Validation data pools
    inter_arrival_times: List[float] = field(default_factory=list)
    service_times: List[float] = field(default_factory=list)
