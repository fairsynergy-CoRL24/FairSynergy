from optimization import *
from tqdm import tqdm
from json import load as jload
import json # TODO delete this line
def feasibility_exp(num_robots=100, num_experiments = 20, seed_val=42):
    """
    Runs a Distributed Learning resource allocation simulation for increasing number of robots in the system with increasing cloud resource proportional
    to the number of robots so that limiting value of uniform allocation stays constant.

    Args:
    num_robots (int, optional): Number of robots. Default is 2.
    num_experiments (int, optional): Number of experiments initiated with randomized/different initial conditions.
    seed_val (int, optional): The seed to fix randomization for reproducability

    Returns:
    tuple: A tuple containing robot accuracies obtained by fairsynerg allocation, uniform allocation and random allocation.
    """
    np.random.seed(seed_val)
    # Lists to store results for different allocation methods
    fss = []  # Results for utility-based bilevel allocation
    rr = []   # Results for random allocation
    uu = []   # Results for uniform allocation

    # Looping through experiments
    for j in tqdm(range(num_experiments)):
        # Lists to store results for the current experiment
        fsResultsForIncreasingRobots = []  # Results for utility-based bilevel allocation
        uResultsForIncreasingRobots = []   # Results for uniform allocation
        rResultsForIncreasingRobots = []   # Results for random allocation
        
        # Generating random initial values for total tasks, total resources, alpha, rho, gamma1, and gamma2
        Tt = np.random.rand(1)[0]
        Ebar = np.random.rand(1)[0]
        alpha_initt = np.array([[np.random.rand(1)[0]/100]])
        rho_initt = np.array([[np.random.rand(1)[0]/100]])
        Gg_alpha = np.array([[np.random.rand(1)[0]]])
        Gg_rho = np.array([[np.random.rand(1)[0]]])
        
        # Looping through different numbers of robots
        for i in range(num_robots):
            irobots = i + 1
            
            # Incrementally updating total tasks, total resources, alpha, rho, gamma1, and gamma2
            if i>0:
                Tt += np.random.rand(1)[0]
                Ebar += np.random.rand(1)[0]
                alpha_initt = np.append(alpha_initt, [[np.random.rand(1)[0]/100]], axis=0)
                rho_initt = np.append(rho_initt, [[np.random.rand(1)[0]/100]], axis=0)
                Gg_alpha = np.append(Gg_alpha, [[np.random.rand(1)[0]]], axis=0)
                Gg_rho = np.append(Gg_rho, [[np.random.rand(1)[0]]], axis=0)
            
            # Creating configuration dictionary with updated values
            con = {
                "E_bar": Ebar,
                "T": Tt,
                "alpha_init_vec": alpha_initt,
                "rho_init_vec": rho_initt,
                "gamma2": Gg_alpha,
                "gamma1": Gg_rho,
                "optforrho": True,
                "optforalpha": True
            }
            # Running fairsynergy allocation, uniform allocation, and random allocation
            _, _, fs = fairSynergy2D(con, irobots)
            _, _, u = uniformAssignment(con, irobots)
            _, _, r = randomAssignment(con, irobots)
            
            # Storing results for the current number of robots
            fsResultsForIncreasingRobots.append(fs)
            uResultsForIncreasingRobots.append(u)
            rResultsForIncreasingRobots.append(r)
        
        # Storing results for the current experiment
        fss.append(fsResultsForIncreasingRobots)
        uu.append(uResultsForIncreasingRobots)
        rr.append(rResultsForIncreasingRobots)
    return fss, uu, rr

def feasibility_exp_EEN(num_robots=100, num_experiments = 20, seed_val=42):
    """
    Runs an Early Exit Inference resource allocation simulation for increasing number of robots in the system with increasing cloud resource proportional
    to the number of robots so that limiting value of uniform allocation stays constant.

    Args:
    num_robots (int, optional): Number of robots. Default is 2.
    num_experiments (int, optional): Number of experiments initiated with randomized/different initial conditions.
    seed_val (int, optional): The seed to fix randomization for reproducability

    Returns:
    tuple: A tuple containing robot accuracies obtained by fairsynerg allocation, uniform allocation and random allocation.
    """
    np.random.seed(seed_val)
    
    # Lists to store results for different allocation methods
    fss = []  # Results for utility-based bilevel allocation
    rr = []   # Results for random allocation
    uu = []   # Results for uniform allocation
    
    # Looping through experiments
    for j in tqdm(range(num_experiments)):
        
        # Lists to store results for the current experiment
        fsResultsForIncreasingRobots = []  # Results for utility-based bilevel allocation
        uResultsForIncreasingRobots = []   # Results for uniform allocation
        rResultsForIncreasingRobots = []   # Results for random allocation
        
        # Generating random initial values for total resource, initial number of EE layers, gamma values
        Tt = np.random.rand(1)[0]
        Ee = np.array([[np.random.rand(1)[0]/10]])
        Gg = np.array([[np.random.rand(1)[0]]])
        
        # Looping through different numbers of robots
        for i in range(num_robots):
            irobots = i+1
            
            # Incrementally updating total resource, initial number of EE layers, gamma values
            if i>0:
                Tt += np.random.rand(1)[0]
                Ee = np.append(Ee,[[np.random.rand(1)[0]/10]],axis=0)
                Gg = np.append(Gg,[[np.random.rand(1)[0]]],axis=0)
                
            # Creating configuration dictionary with updated values
            con = {"T":Tt, "E_init":Ee, "gamma_vec":Gg}
            
            # Running utility-based bilevel allocation, uniform allocation, and random allocation
            _, fs = fairSynergy1D(con, irobots)
            _, u = randomAssignment1D(con, irobots)
            _, r = uniformAssignment1D(con, irobots)
            
            # Storing results for the current number of robots
            fsResultsForIncreasingRobots.append(fs)
            uResultsForIncreasingRobots.append(u)
            rResultsForIncreasingRobots.append(r)
            
        # Storing results for the current experiment
        fss.append(fsResultsForIncreasingRobots)
        uu.append(uResultsForIncreasingRobots)
        rr.append(rResultsForIncreasingRobots)
    return fss, uu, rr


#TODO generalize the functions to indefinite degrees of freedom

def boxplot_experiments(configs_file = "experimental_data/DL_configs.json", seed=42, isSaving=False):
    """
    Runs a simulation with different configurations specified in a JSON file and compares the performance of different allocation methods.

    Args:
        configs_file (str, optional): Path to the JSON file containing simulation configurations. Default is "experimental_data/DL_configs.json".
        seed (int, optional): Seed value for randomization. Default is 42.
        isSaving (bool, optional): Flag indicating whether to save the results. Default is False.

    Returns:
        tuple: A tuple containing lists of accuracies obtained by random allocation, uniform allocation, and fair synergy allocation.
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    # Load configurations from the JSON file
    with open(configs_file, 'r') as json_file:
    # Load the JSON data from the file
        configs = jload(json_file)
    #configs = {"E_bar_vec":E_bar_vec, "T_vec":T_vec, "alpha_init_1":alpha_init_1, "alpha_init_2":alpha_init_2, "rho_init_1":rho_init_1, "rho_init_2":rho_init_2, "gamma2_vec":gamma2_vec, "gamma1_vec":gamma1_vec}
    # Initialize result lists for different allocation methods
    results_fs = []
    results_uni = []
    results_rand = []
    # Iterate over different configurations and perform simulations
    for conf in tqdm(range(len(configs["E_bar_vec"]))):
        # Extract configuration parameters
        config = {"E_bar":configs["E_bar_vec"][conf],
                "T":configs["T_vec"][conf],
                "alpha_init_vec":np.asarray([[configs["alpha_init_1"][conf]],[configs["alpha_init_2"][conf]]]),
                "rho_init_vec":np.asarray([[configs["rho_init_1"][conf]],[configs["rho_init_2"][conf]]]),
                "gamma2":np.asarray([[configs["gamma2_vec"][conf]], [configs["gamma2_vec"][conf]]]),
                "gamma1":np.asarray([[configs["gamma1_vec"][conf]], [configs["gamma1_vec"][conf]]]),
                "optforrho":True,
                "optforalpha":True
                }
        # Perform simulations with different allocation methods
        _,_, fs_res = fairSynergy2D(config)
        _,_, randres = randomAssignment(config)
        _,_, unires = uniformAssignment(config)
        # Store results in respective lists
        results_fs.append(fs_res)
        results_uni.append(unires)
        results_rand.append(randres)
    # Save results if isSaving flag is True       
    if isSaving:
        np.save("./experimental_data/DL_BoxPlot_Results.npy", np.array([results_rand, results_uni, results_fs]))
        
    return results_fs, results_uni, results_rand

