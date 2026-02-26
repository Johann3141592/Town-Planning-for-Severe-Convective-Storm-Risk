#This is the main file for the simulation. It imports the necessary libraries and functions and runs the simulation according to the setup.

#Make sure to activate the conda environment before running this file.

#configuration for the simulation
samplesize = 1000 #number of minutes the simulation should run for
buiseness_park_cords = "A" #cordinates of the buisness 
relocation = True #whether to run the simulation with relocation or not
props_of_interest = ["Buiseness Park"]
#getting the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as random
import datetime as dt
from functions import get_scs_chords, calculate_damage, plot_histogram, plot_convergence, plot_occurence_exceedence, get_conditional_probability, analyse_conditional_probaility, plot_convergence_of_exceedence


#filestructure for the simulation
setupdirectory = "./Setup-Files/"
outdirectory = "./results/"

#getting prop cords and values into a dataframe
props_normal_df = pd.read_csv(setupdirectory + "Buildings_FL_E_1.2_2026.txt", sep="\s+", header=None, engine='python')
props_normal_df = props_normal_df.rename(columns={0: "Type", 1: "x", 2: "y", 3: "value"})
props_normal_df["Cords"] = list(zip(props_normal_df.x, props_normal_df.y))
props_relocation_df = pd.read_csv(setupdirectory + "Buildings_FL_F_1.3_2026.txt", sep="\s+", header=None, engine='python')
props_relocation_df = props_relocation_df.rename(columns={0: "Type", 1: "x", 2: "y", 3: "value"})
props_relocation_df["Cords"] = list(zip(props_relocation_df.x, props_relocation_df.y))

#adding buisness park coordinates to dataframe
if buiseness_park_cords == "A":
    props_normal_df = props_normal_df._append({"Type": "Buisness Park", "x": 1, "y": 12, "value": 120, "Cords": (1, 12)}, ignore_index=True)
    props_relocation_df = props_relocation_df._append({"Type": "Buisness Park", "x": 1, "y": 12, "value": 120, "Cords": (1, 12)}, ignore_index=True)
    print("Buisness Park added at coordinates (1, 12) with value 120.")
    buiseness_park = True
elif buiseness_park_cords == "B":
    props_normal_df = props_normal_df._append({"Type": "Buisness Park", "x": 5, "y": 6, "value": 60, "Cords": (5, 6)}, ignore_index=True)
    props_relocation_df = props_relocation_df._append({"Type": "Buisness Park", "x": 5, "y": 6, "value": 60, "Cords": (5, 6)}, ignore_index=True)
    print("Buisness Park added at coordinates (5, 6) with value 60.")
    buiseness_park = True
else:
    buiseness_park = False

#running the simulation
if relocation:
    df = props_relocation_df
else:
    df = props_normal_df

losses = []
types_events = []
loss_mean = []
loss_mean_std = []
exceedence_probabilities = []
std_exceedence_probabilities = []

#main loop for the simulation, it runs for the number of minutes specified in the samplesize variable. In each iteration, it gets the coordinates of the severe convective storm, calculates the damage and appends the loss and event type to the respective lists. It also calculates the mean and standard deviation of the losses after each iteration for convergence analysis.
for i in range(samplesize):
    scs_cords = get_scs_chords()
    loss, event_type = calculate_damage(scs_cords,  df)
    losses.append(loss)
    types_events.append(event_type)
    loss_mean.append(np.mean(losses))
    loss_mean_std.append(np.std(losses))
    p_exc = np.mean(np.array(losses) > 120)
    exceedence_probabilities.append(p_exc)
    std_exceedence_probabilities.append(np.sqrt(p_exc * (1 - p_exc) / len(losses)))

#title = f"Histogram of losses for {'relocation' if relocation else 'normal'} scenario {f'with buiseness park in Location {buiseness_park_cords}' if buiseness_park else 'without buiseness park'}"

#plotting the results
plot_histogram(losses, title=f"Histogram of losses for {'relocation' if relocation else 'normal'} scenario \n {f'with buiseness park in Location {buiseness_park_cords}' if buiseness_park else 'without buiseness park'}", keepzeros=True)
plot_histogram(losses, title=f"Histogram of losses for {'relocation' if relocation else 'normal'} scenario \n {f'with buiseness park in Location {buiseness_park_cords}' if buiseness_park else 'without buiseness park'}", keepzeros=False)
plot_convergence(loss_mean, loss_mean_std, title=f"Convergence of mean losses for {'relocation' if relocation else 'normal'} scenario \n {f'with buiseness park in Location {buiseness_park_cords}' if buiseness_park else 'without buiseness park'}")
plot_occurence_exceedence(losses, title=f"Occurrence exceedence for {'relocation' if relocation else 'normal'} scenario \n {f'with buiseness park in Location {buiseness_park_cords}' if buiseness_park else 'without buiseness park'}")
andprob, orprob, houseprob = get_conditional_probability(losses, types_events, props_of_interest, cutoff=240, housecutoff=3)
print(f"Conditional probability of an event affecting {props_of_interest} given that there is a loss: {andprob:.4f}")
analyse_conditional_probaility(losses, types_events, props_of_interest)
plot_convergence_of_exceedence(exceedence_probabilities, std_exceedence_probabilities, title=f"Convergence of exceedence probability for {'relocation' if relocation else 'normal'} scenario \n {f'with buiseness park in Location {buiseness_park_cords}' if buiseness_park else 'without buiseness park'}")
