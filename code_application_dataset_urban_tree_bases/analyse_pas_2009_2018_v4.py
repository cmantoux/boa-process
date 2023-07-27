import numpy as np
import matplotlib.pyplot as plt
import csv
import pickle

# Allows importing from neighbor folder
import sys
sys.path.append("..")

import src.estimation
import src.postprocessing


SAVE_ESTIMATION = False # If True, saves the MCMC chains in pickle files.
USE_SAVE = False # If True, skips the simulation and used the previously saved pickle file.

liste_especes_2009_2018 = ['Capsella bursa-pastoris', 'Chenopodium album', 'Conyza', 'Hordeum murinum', 'Lactuca serriola', 'Plantago lanceolata', 'Plantago major', 'Polygonum aviculare', 'Senecio inaequidens', 'Senecio vulgaris', 'Sisymbrium irio', 'Stellaria media', 'Taraxacum', 'Veronica persica'] 

#Step 1 - Generation of a visualization of the temporal dynamics of the metapopulation
for selected_plant in liste_especes_2009_2018 :
    print(selected_plant)
    # The output files will be saved in "output_folder/simulation_id[...]"
    output_folder = "output_analysis_pa_2009_2018"
    simulation_id = selected_plant

    np.random.seed(0) # Fix the random seed

    # Obtain the list of all the files starting with selected_plant
    f = open("ls.txt", "r")
    plant_files_raw = f.readlines()
    f.close()
    plant_files = [l[:-1] for l in plant_files_raw if l[:len(selected_plant)]==selected_plant]
    names_streets = [l[len(selected_plant)+1:len(selected_plant)+5] for l in plant_files_raw if l[:len(selected_plant)]==selected_plant]

    print(f"There are {len(plant_files)} distinct streets.\n")

    # Read the related observations
    observations = []
    for l in plant_files:
        O = np.loadtxt(f"donnees_occurence_2009_2018/{l}", delimiter=',')
        if len(O.shape)==1: # if there is only one patch
            O = O.reshape(-1, 1)
        observations.append(O)

    # Estimate the parameters of the noisy BOA model
    if USE_SAVE:
        f = open(f"{output_folder}/{simulation_id}_estimation.pkl", "rb")
        estimation_result = pickle.load(f)
        f.close() 
    else:
        estimation_result = src.estimation.estimate_multiple_streets(observations, H_max=10, niter=50)

    if SAVE_ESTIMATION:
        f = open(f"{output_folder}/{simulation_id}_estimation.pkl", "wb")
        pickle.dump(estimation_result, f)
        f.close()
    
    street_names = [l[len(selected_plant)+1:-5] for l in plant_files_raw if l[:len(selected_plant)]==selected_plant]
    print(f"List of street names: {street_names}\n")

    ############### Generation of images

    # Define the critical probability
    p_c = np.array([0.461, 0.652, 0.743,
                    0.796, 0.831, 0.856,
                    0.874, 0.888, 0.899,
                    0.909, 0.916])
    # p_c[h] = critical extinction probability for H=h


    # Display the complete result convergence

    src.postprocessing.plot_convergence(estimation_result)
    plt.savefig(f"{output_folder}/{simulation_id}_noisy_boa.png", bbox_inches="tight")


    # Print summary information

    # First, drop the first half of the MCMC iterations
    estimation_size = len(estimation_result["p_ext"])
    simplified_history = src.postprocessing.simplify_history(estimation_result, keep_length=estimation_size//2)

    # Next, compute and print the summary information
    summary = src.postprocessing.compute_summary(simplified_history, p_c, street_names=street_names)

    print("Summary:\n")
    for key in summary:
        print(key, ":", summary[key])
        

    # Create CSV with numerical results
    import pandas as pd
    df = pd.DataFrame(index=street_names, data={
        "GER": [summary["GER"][s] for s in street_names],
        "MaxGER": [summary["MaxGER"][s] for s in street_names],
        "p_ext_mean": [summary["p_ext_mean"][s] for s in street_names],
        "p_ext_std": [summary["p_ext_std"][s] for s in street_names],
        "log_lk": [summary["log_lk"][s] for s in street_names],
    })
    df.to_csv(f"{output_folder}/{simulation_id}_summary.csv")

    print(f"\nGER, MaxGER, p_ext_mean, p_ext_std, and log_lk were saved to {output_folder}/{simulation_id}_summary.csv")


    # Clean distribution plots for H and p_ext

    street_design = pd.read_csv("all_streets.csv", index_col="street")
    src.postprocessing.plot_posterior(simplified_history, summary,
                street_names, street_design,
                H_output=f"{output_folder}/{simulation_id}_H.pdf",
                p_ext_output=f"{output_folder}/{simulation_id}_p_ext.pdf",
               )    
