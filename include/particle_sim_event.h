#ifndef EVENT_SIM_PARTICLE_SIM_EVENT_H
#define EVENT_SIM_PARTICLE_SIM_EVENT_H

#include "event_simulation_data.h"

void transport_particle_event_based(EventSimulationData& sim_data);
void process_init_events(EventSimulationData& sim_data);
void process_advance_particle_events(EventSimulationData& sim_data);
void process_surface_crossing_events(EventSimulationData& sim_data);
void process_collision_events(EventSimulationData& sim_data);

#endif
