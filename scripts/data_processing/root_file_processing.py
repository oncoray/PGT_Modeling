#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 09:52:33 2023

@author: kiesli21
"""
import uproot
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load the files
file1 = uproot.open("/bigdata/invivo/machine_learning/pgt-range-reconstruction/PMMA_study/data/01_ROOT/1607161215/6_Calibrated_LMD.root")
file2 = uproot.open("/bigdata/invivo/machine_learning/pgt-range-reconstruction/PMMA_study/data/01_ROOT/1607161215/7_SpotAssignment.root")

# Load the data
data1 = file1["data"]
data2 = file2["spots"]

# Extract the arrays
energy = data1["energy"].array()
triggertime = data1["triggertime"].array()

Spot_start = data2["Spot_start"].array()
Spot_end = data2["Spot_end"].array()

# Define the sampling frequency (in Hz)
sampling_frequency = 106e6  # 106MHz

time_period = 1 / sampling_frequency

TimestampsPerSecond = 217088000000

print(time_period * 217088000000)

# Convert the trigger times to full frequency
triggertime_converted = (triggertime / TimestampsPerSecond) % time_period
triggertime_cycle = triggertime % 2048
print(len(np.unique(triggertime_converted)))
print(len(np.unique(triggertime % 2048)))
# Create a list to hold events for each spot
events_per_spot = []

all_energies = []

# Create a list to hold all timings
all_timings = []

# Loop through each spot
for i, (start, end) in enumerate(tqdm(zip(Spot_start, Spot_end), total = len(Spot_start))):
    # Find the indices of the events for this spot
    spot_indices = (triggertime >= start) & (triggertime < end)

    # Extract the events for this spot
    spot_energy = energy[spot_indices]
    
    spot_timing = triggertime_converted[spot_indices]

    # Save the tuples of energy and timing for this spot
    spot_events = list(zip(spot_energy, spot_timing))

    # Append this spot's events to the list
    events_per_spot.append(spot_events)

    # Add the timings to the all_timings list
    all_timings.extend(spot_timing)
    
    all_energies.extend(spot_energy)

# Create a histogram of all the timings
plt.hist(all_timings, bins=2048)
plt.xlabel('Converted Triggertime (s)')
plt.ylabel('Frequency')
plt.title('Histogram of Converted Triggertimes for All Spots')
plt.show()


all_timings_2d = np.array(all_timings)
all_energies_2d = np.array(all_energies)

all_timings_2d = all_timings_2d[all_energies_2d <= 10]
all_energies_2d = all_energies_2d[all_energies_2d <= 10]


# Create a 2D histogram of all the energies and timings
plt.hist2d(all_timings_2d,all_energies_2d, bins=(2048,300), cmap=plt.cm.jet, norm = 'log')
plt.colorbar(label='Frequency')
plt.xlabel('Energy')
plt.ylabel('Converted Triggertime (s)')
plt.title('2D Histogram of Energy and Converted Triggertimes for All Spots')
plt.show()

#### background ####

all_energies = []

# Create a list to hold all timings
all_timings = []

# Loop through each spot, excluding the last one
# for i in tqdm(range(len(Spot_start) - 1)):
#     # Find the indices of the events for this spot
#     spot_indices = (triggertime >= Spot_end[i]) & (triggertime < Spot_start[i + 1])

#     # Extract the events for this spot
#     spot_energy = energy[spot_indices]
#     spot_timing = triggertime_converted[spot_indices]

#     # Save the tuples of energy and timing for this spot
#     spot_events = list(zip(spot_energy, spot_timing))

#     # Append this spot's events to the list
#     events_per_spot.append(spot_events)

#     # Add the timings to the all_timings list
#     all_timings.extend(spot_timing)
    
#     all_energies.extend(spot_energy)


spot_indices = (triggertime >= Spot_end[-1]) | ((triggertime <= Spot_start[0])  & triggertime > Spot_start[0] - 2 * (Spot_end[0].astype(np.int64) - Spot_start[0].astype(np.int64)))

# Extract the events for this spot
spot_energy = energy[spot_indices]
spot_timing = triggertime_converted[spot_indices]

# Save the tuples of energy and timing for this spot
spot_events = list(zip(spot_energy, spot_timing))

# Append this spot's events to the list
events_per_spot.append(spot_events)

# Add the timings to the all_timings list
all_timings.extend(spot_timing)

all_energies.extend(spot_energy)

# Create a histogram of all the timings
plt.hist(all_timings, bins=2048)
plt.xlabel('Converted Triggertime (s)')
plt.ylabel('Frequency')
plt.title('Histogram of Converted Triggertimes for All Spots')
plt.show()


all_timings_2d = np.array(all_timings)
all_energies_2d = np.array(all_energies)

all_timings_2d = all_timings_2d[all_energies_2d <= 10]
all_energies_2d = all_energies_2d[all_energies_2d <= 10]


# Create a 2D histogram of all the energies and timings
plt.hist2d(all_timings_2d,all_energies_2d, bins=(2048,300), cmap=plt.cm.jet, norm = "log")
plt.colorbar(label='Frequency')
plt.xlabel('Energy')
plt.ylabel('Converted Triggertime (s)')
plt.title('2D Histogram of Energy and Converted Triggertimes for All Spots')
plt.show()
