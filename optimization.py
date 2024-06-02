import numpy as np
import cvxpy as cp

## 2D Allocation mechanisms are provided below:

# Function for uniform allocation of resources to robots
def uniformAssignment(config, num_robots=2, isPrint=False):
    """
    Allocates resources uniformly to robots and calculates accuracies.

    Args:
    config (dict): A dictionary containing configuration parameters including 'E_bar' (total resources),
                   'T' (total tasks), 'alpha_init_vec' (initial alpha values for robots),
                   'rho_init_vec' (initial rho values for robots), 'gamma1', and 'gamma2'.
    num_robots (int, optional): Number of robots. Default is 2.
    isPrint (bool, optional): Whether to print the results. Default is False.

    Returns:
    tuple: A tuple containing allocated alpha, allocated rho, and the mean accuracy.
    """
    np.random.seed(42)
    # Extracting values from the configuration dictionary
    B = config['E_bar']  # Total resources available
    T = config['T']      # Total tasks to be performed
    alpha_init = config['alpha_init_vec']  # Initial alpha values for robots
    rho_init = config['rho_init_vec']      # Initial rho values for robots
    gamma1 = config['gamma1']               # Weight parameter for rho in accuracy calculation
    gamma2 = config['gamma2']               # Weight parameter for alpha in accuracy calculation

    # Calculating resource allocation per robot
    alpha_alloc = (T/num_robots)*np.ones(np.shape(alpha_init))
    rho_alloc = (B/num_robots)*np.ones(np.shape(rho_init))

    # Updating alpha and rho values after allocation
    alpha = alpha_init + alpha_alloc
    rho = rho_init + rho_alloc

    # Calculating accuracies based on allocated resources
    accuracies = calcAcc(alpha, rho, gamma1, gamma2)

    # Calculating the mean accuracy as the result
    result = np.mean(accuracies)

    # Printing results if isPrint is True
    if isPrint:
        print("Uniform Allocation Result: ")
        print(result)
        print("Allocations")
        print("Allocated rho:")
        print(rho_alloc)
        print("Final rho:")
        print(rho)
        print("Allocated alpha:")
        print(alpha_alloc)
        print("Final alpha:")
        print(alpha)
        print("Accuracies")
        print(accuracies)
        print("=============================")

    return alpha_alloc, rho_alloc, result


# Function for random allocation of resources to robots
def randomAssignment(config, num_robots=2, isPrint=False):
    """
    Allocates resources randomly to robots and calculates accuracies.

    Args:
    config (dict): A dictionary containing configuration parameters including 'E_bar' (total resources),
                   'T' (total tasks), 'alpha_init_vec' (initial alpha values for robots),
                   'rho_init_vec' (initial rho values for robots), 'gamma1', and 'gamma2'.
    num_robots (int, optional): Number of robots. Default is 2.
    isPrint (bool, optional): Whether to print the results. Default is False.

    Returns:
    tuple: A tuple containing allocated alpha, allocated rho, and the mean accuracy.
    """
    np.random.seed(42)
    # Extracting values from the configuration dictionary
    B = config['E_bar']  # Total resources available
    T = config['T']      # Total tasks to be performed
    alpha_init = config['alpha_init_vec']  # Initial alpha values for robots
    rho_init = config['rho_init_vec']      # Initial rho values for robots
    gamma1 = config['gamma1']               # Weight parameter for rho in accuracy calculation
    gamma2 = config['gamma2']               # Weight parameter for alpha in accuracy calculation
    
    # Generating random allocations for alpha and rho based on Dirichlet distribution
    rho_alloc = np.transpose(np.random.dirichlet(np.ones(num_robots), size=1) * B)
    alpha_alloc = np.transpose(np.random.dirichlet(np.ones(num_robots), size=1) * T)
    
    # Updating alpha and rho values after random allocation
    alpha = alpha_init + alpha_alloc
    rho = rho_init + rho_alloc

    # Calculating accuracies based on allocated resources
    accuracies = calcAcc(alpha, rho, gamma1, gamma2)

    # Calculating the mean accuracy as the result
    result = np.mean(accuracies)

    # Printing results if isPrint is True
    if isPrint:
        print("Random Allocation Result: ")
        print(result)
        print("Allocations")
        print("Allocated rho:")
        print(rho_alloc)
        print("Final rho:")
        print(rho)
        print("Allocated alpha:")
        print(alpha_alloc)
        print("Final alpha:")
        print(alpha)
        print("Accuracies")
        print(accuracies)
        print("=============================")

    return alpha_alloc, rho_alloc, result


def calcAcc(rho, alpha, gamma1, gamma2):
    """
    Calculates accuracy based on alpha, rho, gamma1, and gamma2.

    Args:
    rho (numpy.ndarray): Rho values for robots.
    alpha (numpy.ndarray): Alpha values for robots.
    gamma1 (float): Weight parameter for rho.
    gamma2 (float): Weight parameter for alpha.

    Returns:
    numpy.ndarray: Calculated accuracies.
    """
    return np.power(rho,gamma1)*np.power(alpha,gamma2)

def calcAggregateAcc(acc, num_robots=2):
    """
    Calculates aggregate accuracy for all robots

    Args:
    acc (numpy.ndarray): The accuracies for individual robots.
    
    Returns:
    numpy.ndarray: Calculated aggregate accuracy.
    """
    return np.sum(acc)/num_robots


# Function for utility-based bilevel resource allocation
def fairSynergy2D(config, num_robots=2):
    """
    Optimizes resource allocation using utility-based bilevel optimization.

    Args:
    config (dict): A dictionary containing configuration parameters including 'E_bar' (total resources),
                   'T' (total tasks), 'alpha_init_vec' (initial alpha values for robots),
                   'rho_init_vec' (initial rho values for robots), 'gamma1', 'gamma2', 'optforrho', and 'optforalpha'.
    num_robots (int, optional): Number of robots. Default is 2.

    Returns:
    tuple: A tuple containing optimized rho and alpha vectors.
    """
    np.random.seed(42)
    # Default input values
    E_bar = 0.5
    T = 0.5
    alpha_init_vec = np.asarray([[0.1], [0.9]])
    rho_init_vec = np.asarray([[0.5], [0.5]])
    gamma1 = 0.7
    gamma2 = 0.2
    optforrho = True
    optforalpha = True

    # Updating input values if provided in the configuration dictionary
    if len(config):
        E_bar = config["E_bar"]
        T = config["T"]
        alpha_init_vec = np.asarray(config["alpha_init_vec"])
        rho_init_vec = np.asarray(config["rho_init_vec"])
        gamma1 = config["gamma1"]
        gamma2 = config["gamma2"]
        optforrho = config["optforrho"]
        optforalpha = config["optforalpha"]

    # Variables for optimization (rho and alpha vectors)
    rho_vec = cp.Variable((num_robots, 1))
    alpha_vec = cp.Variable((num_robots, 1))
    
    # Fixed versions of the variables (initialized with zeros)
    fixed_rho_vec = np.zeros((num_robots, 1))
    fixed_alpha_vec = np.zeros((num_robots, 1))

    # Optimization for rho if optforrho is True
    if optforrho:    
        constraints = []
        A_mat_rho = np.concatenate((np.ones((1, num_robots)), -np.eye(num_robots)), axis=0)
        b_vec_rho = np.concatenate((np.array([[E_bar]]), np.zeros((num_robots, 1))), axis=0)
        constraints += [A_mat_rho @ rho_vec <= b_vec_rho]
        objective_expr = 0
        for i in range(num_robots):
            objective_expr += cp.multiply(cp.power((rho_init_vec[i, 0] + rho_vec[i, 0]), gamma1[i, 0]),
                                          cp.power(alpha_init_vec[i, 0] + fixed_alpha_vec[i, 0], gamma2[i, 0]))
        objective = cp.Minimize(-objective_expr)

        # Solving the optimization problem for rho
        problem = cp.Problem(objective, constraints)
        problem.solve()
        fixed_rho_vec = rho_vec.value
        
    # Optimization for alpha if optforalpha is True
    if optforalpha:
        constraints = []
        A_mat_alpha = np.concatenate((np.ones((1, num_robots)), -np.eye(num_robots)), axis=0)
        b_vec_alpha = np.concatenate((np.array([[T]]), np.zeros((num_robots, 1))), axis=0)
        constraints += [A_mat_alpha @ alpha_vec <= b_vec_alpha]
        objective_expr = 0
        for i in range(num_robots):
            objective_expr += cp.multiply(cp.power((rho_init_vec[i, 0] + fixed_rho_vec[i, 0]), gamma1[i, 0]),
                                          cp.power(alpha_init_vec[i, 0] + alpha_vec[i, 0], gamma2[i, 0]))
        objective = cp.Minimize(-objective_expr)

        # Solving the optimization problem for alpha
        problem = cp.Problem(objective, constraints)
        problem.solve()
        fixed_alpha_vec = alpha_vec.value
    
    # calculate the mean accuracy for the resulting allocation
    final_acc = np.sum(calcAcc(fixed_rho_vec+rho_init_vec, fixed_alpha_vec+alpha_init_vec, gamma1, gamma2))/num_robots
    
    # Returning optimized fixed_rho_vec and fixed_alpha_vec
    return fixed_rho_vec, fixed_alpha_vec, final_acc

# Function for univariate NUM like resource allocation
def NUM2D(config, num_robots=2):
    """
    Allocates 2D resources to robots based on given constraints.

    Args:
    config (dict): A dictionary containing configuration parameters including 'E_bar' (total resources),
                   'T' (total tasks), 'alpha_init_vec' (initial alpha values for robots),
                   'rho_init_vec' (initial rho values for robots), 'gamma1', 'gamma2', 'optforrho', and 'optforalpha'.
    num_robots (int, optional): Number of robots. Default is 2.

    Returns:
    tuple: A tuple containing optimized rho and alpha vectors.
    """
    # Default input values
    E_bar = 0.5
    T = 0.5
    alpha_init_vec = np.asarray([[0.1], [0.9]])
    rho_init_vec = np.asarray([[0.5], [0.5]])
    gamma1 = 0.7
    gamma2 = 0.2
    optforrho = True
    optforalpha = True

    # Updating input values if provided in the configuration dictionary
    if len(config):
        E_bar = config["E_bar"]
        T = config["T"]
        alpha_init_vec = np.asarray(config["alpha_init_vec"])
        rho_init_vec = np.asarray(config["rho_init_vec"])
        gamma1 = config["gamma1"]
        gamma2 = config["gamma2"]
        optforrho = config["optforrho"]
        optforalpha = config["optforalpha"]
    
    # Calculating fixed allocations for rho and alpha based on initial values and constraints
    fixed_rho_vec = np.ones((num_robots, 1)) * (
            (E_bar + np.sum(rho_init_vec)) / num_robots > rho_init_vec) * (
                            (E_bar + np.sum(rho_init_vec)) / num_robots - rho_init_vec)
    fixed_rho_vec = fixed_rho_vec / np.sum(fixed_rho_vec) * E_bar  # Normalizing to total resources

    fixed_alpha_vec = np.ones((num_robots, 1)) * (
            (T + np.sum(alpha_init_vec)) / num_robots > alpha_init_vec) * (
                             (T + np.sum(alpha_init_vec)) / num_robots - alpha_init_vec)
    fixed_alpha_vec = fixed_alpha_vec / np.sum(fixed_alpha_vec) * T  # Normalizing to total tasks

    return fixed_rho_vec, fixed_alpha_vec

## 1D Allocation mechanisms are provided below:

def uniformAssignment1D(config, num_robots=2):
    """
    Performs uniform resource allocation for 1D scenario.

    Args:
        config (dict): Configuration parameters including 'T', 'gamma_vec', and 'E_init'.
        num_robots (int, optional): Number of robots. Default is 2.

    Returns:
        tuple: A tuple containing the allocated resource and the mean accuracy obtained.
    """
    # Inputs
    T = config["T"]
    # initializtion
    gamma_vec = config["gamma_vec"]
    E_init = config["E_init"]
    E_alloc = np.ones((num_robots,1))*(T/num_robots)
    final_acc = calcAcc1D(E_alloc+E_init, gamma_vec)
    return E_alloc, np.mean(final_acc)

def randomAssignment1D(config, num_robots=2):
    """
    Performs random resource allocation for 1D scenario.

    Args:
        config (dict): Configuration parameters including 'T', 'gamma_vec', and 'E_init'.
        num_robots (int, optional): Number of robots. Default is 2.

    Returns:
        tuple: A tuple containing the allocated resource and the mean accuracy obtained.
    """
    # Inputs
    T = config["T"]
    # initializtion
    gamma_vec = config["gamma_vec"]
    E_init = config["E_init"]
    E_alloc = np.transpose(np.random.dirichlet(np.ones(num_robots),size=1)*T)
    final_acc = calcAcc1D(E_alloc+E_init, gamma_vec)
    return E_alloc, np.mean(final_acc)


def fairSynergy1D(config, num_robots=2):
    """ A function to realize fair resource allocation mechanism
    """
    # Inputs
    T = config["T"]

    # initializtion
    gamma_vec = config["gamma_vec"]
    E_init = config["E_init"]

    # Variables
    E_bar = cp.Variable((num_robots,1))
    
    # Constraints
    constraints = []
    A_mat = np.concatenate((np.ones((1,num_robots)), -np.eye(num_robots)),axis=0)
    b_vec = np.concatenate((np.array([[T]]), np.zeros((num_robots,1))),axis=0)
    constraints += [A_mat@E_bar <= b_vec]
    
    # Define the problem and solve it
    objective_expr = 0
    for i in range(num_robots):
        objective_expr +=cp.power((E_bar[i,0]+E_init[i,0]),gamma_vec[i,0])
    objective = cp.Minimize(-objective_expr)


    problem = cp.Problem(objective, constraints)
    problem.solve()
    E_alloc = E_bar.value
    final_acc = calcAcc1D(E_alloc+E_init, gamma_vec)
    return E_alloc, np.sum(final_acc)/num_robots

def calcAcc1D(E, gamma):
    """
    Calculates accuracy based on allocated energy and gamma values.

    Args:
        E (array): Allocated resource values.
        gamma (array): Gamma values.

    Returns:
        array: Accuracy values.
    """
    return np.power(E,gamma)

def calcDualForBenchmarks(E_bar, con):
    """
    Calculates dual variables from the paper's dual problem's equality constraint used for agent's bid.

    Args:
        E_bar (array): Allocated resource values.
        con (dict): Configuration parameters including 'gamma_vec' and 'E_init'.

    Returns:
        array: Dual variables.
    """
    gamma_var = con['gamma_vec'] 
    return gamma_var/np.power(E_bar + con['E_init'], np.ones(np.shape(gamma_var)) - gamma_var)

    

# def calculateAcc1D(config, E_bar, num_robots=2): # TODO generalize it to multi dimensional case to be used
#     # initializtion
#     # TODO add the degree of freedom as configuration information or extract it from there.
#     gamma_vec = config["gamma_vec"]
#     E_init = config["E_init"]
#     #print(np.power(E_bar+E_init, gamma_vec))
#     return np.sum(np.power(E_bar+E_init, gamma_vec))/num_robots

# TODO check the functions below for the generalizability/ where to use

# def calcDualForBenchmarks(E_bar, con):
#     gamma_var = con['gamma_vec'] 
#     return gamma_var/np.power(E_bar + con['E_init'], np.ones(np.shape(gamma_var)) - gamma_var)
