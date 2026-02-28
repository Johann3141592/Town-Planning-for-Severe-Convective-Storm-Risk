#This is the main file for the simulation. It imports the necessary Librarys and functions and runs the simulation according to the setup.



#Make sure to activate the conda environment before running this file.

#configuration for the simulation
samplesize = 100000 #number of minutes the simulation should run for
props_of_interest = ["Buiseness Park"] #This is what property is of intered for conditional probabilities
#getting the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as random
import time
import pickle
from multiprocessing import Pool
from functions import get_scs_chords, calculate_damage, plot_histogram, plot_convergence, plot_occurence_exceedence, get_conditional_probability, analyse_conditional_probaility, plot_convergence_of_exceedence, simul_for_setup, simul_for_setup_multiplocessing, plot_property_map



if __name__ == "__main__":
    #doing this is necessary to ensure that the multiprocessing code runs correctly on Windows
    simul_params = []
    summary = pd.DataFrame(columns=["Scenario", "Buiseness Park", "Average Annual Loss", "Std Loss", "Exceedence Probability", "Std Exceedence Probability"])
    summary_list = []

    start_time = time.time()
    #creating the necessary parameter combinations for the simulations, in this case we want to run the simulation for both the relocation and normal scenario and for both the buiseness park being in location A, location B or not being present at all. This results in a total of 6 simulations to run (2 scenarios x 3 buiseness park setups).
    for relocation in [True, False]:
        for buiseness_park in ["A", "B", False]:          
            simul_params.append((relocation, buiseness_park, props_of_interest, samplesize))
            
    #using pool to allow multiple simulations to run at the same time to allow higher samplesizes
    with Pool() as pool:
        summary_list = pool.map(simul_for_setup_multiplocessing, simul_params)

    summary = pd.concat([summary, pd.DataFrame(summary_list)], ignore_index=True)
    #saving the key results to an excel file for easier analysis and plotting in other software if desired
    summary.to_excel("./results/summary.xlsx", index=False)

    end_time = time.time()
    print(f"Simulation for samplesize of {samplesize} completed in {end_time - start_time:.2f} seconds.")