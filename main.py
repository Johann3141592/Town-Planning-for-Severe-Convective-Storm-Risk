#This is the main file for the simulation. It imports the necessary libraries and functions and runs the simulation according to the setup.

#Make sure to activate the conda environment before running this file.

#configuration for the simulation
samplesize = 1000000 #number of minutes the simulation should run for
buiseness_park_cords = "A" #cordinates of the buisness 
relocation = True #whether to run the simulation with relocation or not
props_of_interest = ["Buiseness Park"]
#getting the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as random
import time
from functions import get_scs_chords, calculate_damage, plot_histogram, plot_convergence, plot_occurence_exceedence, get_conditional_probability, analyse_conditional_probaility, plot_convergence_of_exceedence, simul_for_setup

summary = pd.DataFrame(columns=["Scenario", "Buiseness Park", "Average Annual Loss", "Std Loss", "Exceedence Probability", "Std Exceedence Probability"])
summary_list = []

start_time = time.time()

for relocation in [True, False]:
    for buiseness_park in ["A", "B", False]:
        #print(f"Running simulation for {'relocation' if relocation else 'normal'} scenario \n {f'with buiseness park in Location {buiseness_park_cords}' if buiseness_park else 'without buiseness park'}")
        #running the simulation for the current setup
        results = simul_for_setup(relocation, buiseness_park,props_of_interest=props_of_interest, samplesize=samplesize)
        losses, types_events, loss_mean, loss_mean_std, exceedence_probabilities, std_exceedence_probabilities = results
        summary_list.append({"Scenario": "Relocation" if relocation else "Normal", "Buiseness Park": f"Location {buiseness_park_cords}" if buiseness_park else "No Buiseness Park", "Average Annual Loss": loss_mean[-1], "Std Loss": loss_mean_std[-1], "Exceedence Probability": exceedence_probabilities[-1], "Std Exceedence Probability": std_exceedence_probabilities[-1]})

summary = pd.concat([summary, pd.DataFrame(summary_list)], ignore_index=True)

summary.to_excel("./results/summary.xlsx", index=False)

end_time = time.time()
print(f"Simulation for samplesize of {samplesize} completed in {end_time - start_time:.2f} seconds.")