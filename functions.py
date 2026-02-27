import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#filestructure for the simulation
setupdirectory = "./Setup-Files/"
outdirectory = "./results/"

def get_scs_chords():
    #propertycoordinates work like the following: the property with the coordinates (1,1) is the bottom left property, thus it covers [0,1]x[0,1], the property with the coordinates (2,1) is the property to the right of it and covers [1,2]x[0,1].
    #stormcenter coordinates are on edges of properties, thus the stormcenter with coordinates (1,1) is on the edge of the property with coordinates (1,1) and (2,1), the stormcenter with coordinates (0,0) is on the edge of the property with coordinates (0,0) and (1,0) and so on.
    scs_cords = []
    x_cord = random.randint(-4, 16)
    y_cord = random.randint(-4, 16)
    #print(f"Storm Center Cords: ({x_cord}, {y_cord})")
    dice1 = random.randint(1, 6)
    dice2 = random.randint(1, 6)
    sum_dice = dice1 + dice2
    #print(f"Dice 1: {dice1}, Dice 2: {dice2}, Sum: {sum_dice}")
    if sum_dice <= 6:
        pass
    elif sum_dice == 7:
        for i in range(0, 2):
            for j in range(0, 2):
                scs_cords.append((x_cord + i, y_cord + j))
    elif sum_dice == 8 or sum_dice == 9:
        for i in range(-1, 3):
            for j in range(-1, 3):
                scs_cords.append((x_cord + i, y_cord + j))
    elif sum_dice == 10 or sum_dice == 11:
        for i in range(-2, 4):
            for j in range(-2, 4):
                scs_cords.append((x_cord + i, y_cord + j))
    elif sum_dice == 12:
        for i in range(-4, 6):
            for j in range(-4, 6):
                scs_cords.append((x_cord + i, y_cord + j))
    #print(len(scs_cords))
    return scs_cords

def calculate_damage(scs_cords, prop_df):
    damage = 0
    types = []
    for index, row in prop_df.iterrows():
        if row["Cords"] in scs_cords:
            damage += row["value"]
            types.append(row["Type"])
    
    return damage, types

def plot_histogram(losslist, title='Histogram of Losses', bins=50, keepzeros=True):

    #creating histogram with mean, std of mean saving to file
    #keepzeros is a boolean that determines whether to include zero losses in the calculation of the mean and standard error of the mean, and whether to show the number of simulated years in the title. If keepzeros is False, zero losses are excluded from the calculations and the title is updated to reflect this.

    # Calculate statistics
    if keepzeros == True:
        filename = title.replace("\n", "").replace(" ", "_")
        mean_val = np.mean(losslist)

        std_val = np.std(losslist)
        sem_val = std_val / np.sqrt(len(losslist))  # Standard error of the mean
    elif keepzeros == False:
        n = len(losslist)
        numzeros = len([loss for loss in losslist if loss == 0])
        losslist_nonzero = [loss for loss in losslist if loss > 0]
        mean_val = np.mean(losslist_nonzero)
        sem_val = np.std(losslist_nonzero) / np.sqrt(len(losslist_nonzero))  # Standard error of the mean
        entries  =  len(losslist_nonzero)
        filename = title.replace("\n", "").replace(" ", "_")
        filename += "nozeros"
        filename = filename.replace("\n", "").replace(" ", "_")
        title += f'\n (Excluding zero loss years which represent ({entries/n:.2f} of all simulated years))'

    plt.figure(figsize=(10, 6))
    
    # Create histogramg
    n, bins, patches = plt.hist(losslist, bins=bins, alpha=0.85, 
                                  color='#3498db', edgecolor='#2c3e50', linewidth=1.2)
    # Add mean and median lines
    plt.axvline(mean_val, color='#e74c3c', linestyle='--', linewidth=1, 
                label=f'AAL: {mean_val:,.0f}M$')

    # Add shaded region for SEM around mean
    plt.axvspan(mean_val - sem_val, mean_val + sem_val, alpha=0.4, color='#e74c3c', 
                label=f'Standard error of AAL: ±M${sem_val:,.1f}')
    

    
    # Styling
    plt.xlabel('Losses (Million $)', fontsize=14, fontweight='bold')
    plt.ylabel(f'Frequency (n={len(losslist)/1000:.0f}k)', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    plt.legend(fontsize=11, framealpha=0.9)
    
    # Improve layout
    plt.tight_layout()

    plt.savefig(outdirectory + filename + ".png", dpi=300)
    #plt.show()
    plt.close()

def plot_convergence(mean_loss_list, std_of_mean_loss_list, title='Convergence of Average Annual Loss (AAL)', outdirectory="./results/"):
    
    #takes a list of mean values and a list of standard errors of the mean, and plots the convergence of the mean values with a shaded area representing the standard error of the mean.
    #The plot is saved to the results directory with a filename derived from the title.
    
    #the central limit theorem states that the distribution of the sample mean will approach a normal distribution as the sample size increases, regardless of the shape of the population distribution. This means that as we increase the number of simulations, the mean loss should converge to a stable value, and the standard error of the mean should decrease, indicating that our estimate of the mean loss is becoming more precise.
    plt.figure(figsize=(10, 6))
    plt.plot(mean_loss_list, label='AAL', color='#3498db')
    plt.fill_between(range(len(mean_loss_list)), 
                     np.array(mean_loss_list) - np.array(std_of_mean_loss_list), 
                     np.array(mean_loss_list) + np.array(std_of_mean_loss_list), 
                     color='#3498db', alpha=0.2, label='Standard Error of AAL (68% Confidence Interval)')
    
    plt.fill_between(range(len(mean_loss_list)), 
                     np.array(mean_loss_list) - 2*np.array(std_of_mean_loss_list), 
                     np.array(mean_loss_list) + 2*np.array(std_of_mean_loss_list), 
                     color="#eb0808", alpha=0.1, label='Standard Error of AAL (95% Confidence Interval)')
    #making sure the axes are readable
    step = max(len(mean_loss_list)//10, 1)
    labellist = range(0, len(mean_loss_list), step)
    # Styling
    plt.ylim(0, max(mean_loss_list)*1.5)
    plt.xlabel('Number of Simulations [k]', fontsize=14, fontweight='bold')
    plt.ylabel('AAL (Million $)', fontsize=14, fontweight='bold')
    plt.xticks(labellist, [f"{x/1000:.0f}" for x in labellist])
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    plt.legend(fontsize=11, framealpha=0.9)
    
    # Improve layout
    plt.tight_layout()
    filename = title.replace("\n", "").replace(" ", "_")
    plt.savefig(outdirectory + filename + ".png", dpi=300)
    plt.close()

def get_conditional_probability(losslist: list, typelist: list, type_of_interest: list, cutoff: float = 0, housecutoff: int = 0, title="Conditional Probabilities of Types of Interest Given Losses Above Cutoff", outdirectory="./results/"):

    #takes losslist and event typelist as well as a list of types of interest and a cutoff value
    #calculates the conditional probability: if loss is higher than cutoff, what is the probablity that the type of interest properties are involved (and, or and exculsive or)
    
    #converting to numpy arrays for easier manipulation
    losslist = np.array(losslist)
    typelist = np.array(typelist, dtype=object)


    boollist = losslist > cutoff
    typelist_after_cutoff = typelist[boollist]

    #and condition: all types of interest must be involved
    counterand = 0
    for destr_props in typelist_after_cutoff:
        if all(proptype in destr_props for proptype in type_of_interest):
            counterand += 1

    #or condition: at least one type of interest must be involved
    counteror = 0
    for destr_props in typelist_after_cutoff:
        if any(proptype in destr_props for proptype in type_of_interest):
            counteror += 1
    
    #conditional probabilities that houses (more than cutoff) are involved
    counterhouse = 0
    for destr_props in typelist_after_cutoff:
        if len([prop for prop in destr_props if prop == "H"]) > housecutoff:
            counterhouse += 1
    
    total_events = len(typelist_after_cutoff)
    proband = counterand / total_events if total_events > 0 else 0
    probor = counteror / total_events if total_events > 0 else 0
    probhouse = counterhouse / total_events if total_events > 0 else 0
    return proband, probor, probhouse

def plot_occurence_exceedence(losslist, title="Occurence Exceedence Plot", outdirectory="./results/"):
    #creating a plot of OAP
    sorted_losses = np.sort(losslist)[::-1]  # Sort losses in descending order
    exceedence_prob = np.arange(1, len(sorted_losses) + 1) / len(sorted_losses)  # Exceedance probability
    return_period = 1 / exceedence_prob  # Return period in years

    fundexceedence = np.interp(240, sorted_losses[::-1], return_period[::-1])


    ten_year_event = np.interp(10, return_period[::-1], sorted_losses[::-1])
    fifty_year_event = np.interp(50, return_period[::-1], sorted_losses[::-1])
    hundred_year_event = np.interp(100, return_period[::-1], sorted_losses[::-1])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(return_period, sorted_losses, marker='o', linestyle='-', color='#3498db')
    #ax.set_xscale('log')

    ax.set_xlabel('Return Period (Years)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Losses (Million $)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_ylim(0, 300)
    ax.set_xlim(0, 200)
    
    # Create legend-style text box
    textstr = (f'10-year event: {ten_year_event:.1f}M$\n'
               f'50-year event: {fifty_year_event:.1f}M$\n'
               f'100-year event: {hundred_year_event:.1f}M$\n'
               f'Exceeding 240M$ fund: {fundexceedence:.1f}-year event')
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#e74c3c', linewidth=2)
    ax.text(0.98, 0.50, textstr, transform=ax.transAxes, fontsize=11, 
            verticalalignment='top', horizontalalignment='right', bbox=props, color='#e74c3c')
    plt.tight_layout()
    filename = title.replace("\n", "").replace(" ", "_")
    plt.savefig(outdirectory + filename + ".png", dpi=300)
    plt.close()

def analyse_conditional_probaility(losslist: list, typelist: list, type_of_interest:list):
    #analysing conditional probability for different cutoffs
    cutoffs = []
    andprobs = []
    orprobs = []
    houseprobs = []
    for cutoff in range(0, 300, 10):
        proband, probor, probhouse = get_conditional_probability(losslist, typelist, type_of_interest, cutoff=cutoff)
        andprobs.append(proband)
        orprobs.append(probor)
        houseprobs.append(probhouse)
        cutoffs.append(cutoff)
    plt.figure(figsize=(10, 6))
    plt.plot(cutoffs, andprobs, color='#3498db')
    plt.title(f"Conditional Probability of a loss given that \n the {type_of_interest[0]} is destroyed ", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Loss (Million $)", fontsize=14, fontweight='bold')
    plt.ylabel("Conditional Probability", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    plt.tight_layout()
    filename = f"Conditional_Probability_{type_of_interest[0]}_Destroyed"
    plt.savefig(outdirectory + filename + ".png", dpi=300)
    plt.close()

def plot_convergence_of_exceedence(exceedence_probabilities, std_exceedence_probabilities, title='Convergence of Occurrence Exceedence Probability (240M$)', outdirectory="./results/"):
    
    #takes a list of exceedence probabilities and a list of standard errors of the exceedence probabilities, and plots the convergence of the exceedence probabilities with a shaded area representing the standard error.
    #The plot is saved to the results directory with a filename derived from the title.
    
    #the central limit theorem states that the distribution of the sample mean will approach a normal distribution as the sample size increases, regardless of the shape of the population distribution. This means that as we increase the number of simulations, the exceedence probability should converge to a stable value, and the standard error should decrease, indicating that our estimate of the exceedence probability is becoming more precise.
    plt.figure(figsize=(10, 6))
    plt.plot(exceedence_probabilities, label='Exceedence Probability', color='#3498db')
    plt.fill_between(range(len(exceedence_probabilities)), 
                     np.array(exceedence_probabilities) - np.array(std_exceedence_probabilities), 
                     np.array(exceedence_probabilities) + np.array(std_exceedence_probabilities), 
                     color='#3498db', alpha=0.2, label='Standard Error of Exceedence Probability (68% Confidence Interval)')
    
    plt.fill_between(range(len(exceedence_probabilities)), 
                     np.array(exceedence_probabilities) - 2*np.array(std_exceedence_probabilities), 
                     np.array(exceedence_probabilities) + 2*np.array(std_exceedence_probabilities), 
                     color="#eb0808", alpha=0.1, label='Standard Error of Exceedence Probability (95% Confidence Interval)')
    #making sure the axes are readable
    step = max(len(exceedence_probabilities)//10, 1)
    labellist = range(0, len(exceedence_probabilities), step)
    # Styling
    plt.ylim(0, 0.2)
    plt.xlabel('Number of Simulations [k]', fontsize=14, fontweight='bold')
    plt.ylabel('Exceedence Probability', fontsize=14, fontweight='bold')
    plt.xticks(labellist, [f"{x/1000:.0f}" for x in labellist])
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    plt.legend(fontsize=11, framealpha=0.9)
    
    # Improve layout
    plt.tight_layout()
    filename = title.replace("\n", "").replace(" ", "_")
    plt.savefig(outdirectory + filename + ".png", dpi=300)
    plt.close()


def simul_for_setup(relocation: bool, buiseness_park_cords: str, props_of_interest: list, samplesize: int):
        #function to run the simulation for a given setup, it takes the same parameters as the main function and runs the simulation for the specified number of iterations, it also plots the results and saves them to the results directory. This function can be used to easily run the simulation for different setups by just calling this function with different parameters.
        
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
        props_normal_df = props_normal_df._append({"Type": "Buisness Park", "x": 1, "y": 12, "value": 240, "Cords": (1, 12)}, ignore_index=True)
        props_relocation_df = props_relocation_df._append({"Type": "Buisness Park", "x": 1, "y": 12, "value": 240, "Cords": (1, 12)}, ignore_index=True)
        #print("Buisness Park added at coordinates (1, 12) with value 240.")
        buiseness_park = True
    elif buiseness_park_cords == "B":
        props_normal_df = props_normal_df._append({"Type": "Buisness Park", "x": 5, "y": 6, "value": 60, "Cords": (5, 6)}, ignore_index=True)
        props_relocation_df = props_relocation_df._append({"Type": "Buisness Park", "x": 5, "y": 6, "value": 60, "Cords": (5, 6)}, ignore_index=True)
        #print("Buisness Park added at coordinates (5, 6) with value 60.")
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
        loss_mean_std.append(np.std(losses)/np.sqrt(len(losses)))
        p_exc = np.mean(np.array(losses) > 240)
        exceedence_probabilities.append(p_exc)
        std_exceedence_probabilities.append(np.sqrt(p_exc * (1 - p_exc) / len(losses)))
        if i % (samplesize // 10) == 0 and i > 0:
            print(f"Simulation progress for {'relocation' if relocation else 'normal'} scenario \n {f'with buiseness park in Location {buiseness_park_cords}' if buiseness_park else 'without buiseness park'}: {i/samplesize:.1%}")
    #title = f"Histogram of losses for {'relocation' if relocation else 'normal'} scenario {f'with buiseness park in Location {buiseness_park_cords}' if buiseness_park else 'without buiseness park'}"

    #plotting the results
    plot_histogram(losses, title=f"Histogram of losses for {'relocation' if relocation else 'normal'} scenario \n {f'with buiseness park in Location {buiseness_park_cords}' if buiseness_park else 'without buiseness park'}", keepzeros=True)
    plot_histogram(losses, title=f"Histogram of losses for {'relocation' if relocation else 'normal'} scenario \n {f'with buiseness park in Location {buiseness_park_cords}' if buiseness_park else 'without buiseness park'}", keepzeros=False)
    plot_convergence(loss_mean, loss_mean_std, title=f"Convergence of Average Annual Losses (AAL) for {'relocation' if relocation else 'normal'} scenario \n {f'with buiseness park in Location {buiseness_park_cords}' if buiseness_park else 'without buiseness park'}")
    plot_occurence_exceedence(losses, title=f"Occurrence exceedence for {'relocation' if relocation else 'normal'} scenario \n {f'with buiseness park in Location {buiseness_park_cords}' if buiseness_park else 'without buiseness park'}")
    andprob, orprob, houseprob = get_conditional_probability(losses, types_events, props_of_interest, cutoff=240, housecutoff=3)
    #print(f"Conditional probability of an event affecting {props_of_interest} given that there is a loss: {andprob:.4f}")
    analyse_conditional_probaility(losses, types_events, props_of_interest)
    plot_convergence_of_exceedence(exceedence_probabilities, std_exceedence_probabilities, title=f"Convergence of exceedence probability for {'relocation' if relocation else 'normal'} scenario \n {f'with buiseness park in Location {buiseness_park_cords}' if buiseness_park else 'without buiseness park'}")
    return losses, types_events, loss_mean, loss_mean_std, exceedence_probabilities, std_exceedence_probabilities

def simul_for_setup_multiplocessing(params):
    #function to run the simulation for a given setup, it takes the same parameters as the main function and runs the simulation for the specified number of iterations, it also plots the results and saves them to the results directory. This function can be used to easily run the simulation for different setups by just calling this function with different parameters.
    relocation, buiseness_park_cords, props_of_interest, samplesize = params
    results = simul_for_setup(relocation, buiseness_park_cords, props_of_interest, samplesize)
    losses, types_events, loss_mean, loss_mean_std, exceedence_probabilities, std_exceedence_probabilities = results
    summary_entry = {"Scenario": "Relocation" if relocation else "Normal", "Buiseness Park": f"Location {buiseness_park_cords}" if buiseness_park_cords else "No Buiseness Park", "Average Annual Loss": loss_mean[-1], "Std Loss": loss_mean_std[-1], "Exceedence Probability": exceedence_probabilities[-1], "Std Exceedence Probability": std_exceedence_probabilities[-1]}
    
    return summary_entry