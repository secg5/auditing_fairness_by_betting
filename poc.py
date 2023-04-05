"""The following script executes a proof of concept for auditing fairness by betting.
The idea is using a notion of fairness to compute the ate?
"""
import numpy as np 
from scipy.stats import bernoulli
from confseq.betting import hedged_cs
import matplotlib.pyplot as plt 
from scipy.optimize import minimize
import scipy


DATASE_SIZE = 1000
TIME_STEPS = 500
p_x_0 = [0.5, 0.3, 0.2]

def simulate_data_different_means(dataset_size: int, time_steps:int):
    """Simulates a simple dataset with a random treatment 
    and a confounding variable.

    Args:
        dataset_size (int): _description_
        time_steps (int): _description_

    Returns:
       Tuple with both sequences of means.
    """
    means_1 = []
    means_0 = []
    p_x_0 = [0.5, 0.3, 0.2]
    for t in range(time_steps):
        X = np.random.choice(a=[0, 1, 2], size=dataset_size, p=p_x_0)

        pi_A = scipy.special.expit(X)
        A = 1*(pi_A > np.random.uniform(size=dataset_size))

        mu = scipy.special.expit(2*A -  X)
        y = 1*(mu > np.random.uniform(size=dataset_size))
        
        mean_1 = y[A == 1].mean()
        mean_0 = y[A == 0].mean()
        
        means_1.append(mean_1)
        means_0.append(mean_0)

    return means_1, means_0

def simulate_data_same_means(dataset_size: int, time_steps:int):
    """Simulates a simple dataset with a random treatment 
    and a confounding variable.

    Args:
        dataset_size (int): _description_
        time_steps (int): _description_

    Returns:
       Tuple with both sequences of means.
    """
    means_1 = []
    means_0 = []
    
    for t in range(time_steps):
        X = np.random.choice(a=[0, 1, 2], size=dataset_size, p=p_x_0)

        pi_A =  np.random.choice(a=[0,1], size=dataset_size, p=[0.3, 0.7])
        A = 1*(pi_A > np.random.uniform(size=dataset_size))

        mu = scipy.special.expit(2*X)
        y = 1*(mu > np.random.uniform(size=dataset_size))
                
        mean_1 = y[A == 1].mean()
        mean_0 = y[A == 0].mean()
        
        means_1.append(mean_1)
        means_0.append(mean_0)

    return means_1, means_0

if __name__ == '__main__':

    means_1, means_0 = simulate_data_different_means(DATASE_SIZE, TIME_STEPS)

    lb_0, ub_0 = hedged_cs(means_0)
    lb_1, ub_1 = hedged_cs(means_1)

    E_0 = 0
    E_1 = 0
    for x_0 in range(3):
        E_0 += scipy.special.expit(0 - x_0)*(p_x_0[x_0])
        E_1 += scipy.special.expit(2 - x_0)*(p_x_0[x_0])

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(np.arange(TIME_STEPS), lb_0, c='navy', label='A = 0')
    axs[0].plot(np.arange(TIME_STEPS), ub_0, c='navy')
    axs[0].plot(np.arange(TIME_STEPS), lb_1, c='tab:olive', label='A = 1')
    axs[0].plot(np.arange(TIME_STEPS), ub_1, c='tab:olive')
    # plt.axhline(E_0, ls='--', c='k')
    # plt.axhline(E_1, ls='--', c='r')
    fig.suptitle("Confidence sequence for E[y|A]_t")
    axs[0].legend()

    s_means_1, s_means_0 = simulate_data_same_means(DATASE_SIZE, TIME_STEPS)

    lb_0, ub_0 = hedged_cs(s_means_0)
    lb_1, ub_1 = hedged_cs(s_means_1)

    axs[1].plot(np.arange(TIME_STEPS), lb_0, c='navy', label='A = 0')
    axs[1].plot(np.arange(TIME_STEPS), ub_0, c='navy')
    axs[1].plot(np.arange(TIME_STEPS), lb_1, c='tab:olive', label='A = 1')
    axs[1].plot(np.arange(TIME_STEPS), ub_1, c='tab:olive')
    #plt.axhline(E_0, ls='--', c='k')
    #plt.axhline(E_1, ls='--', c='r')
    axs[1].legend()
    fig.savefig("confidence_sequence")

    plt.figure(1)
    plt.plot(np.abs(np.array(lb_0) - np.array(ub_1)))
    plt.title("Absolute value of difference in means")
    plt.savefig("means_diference")

    
    plt.figure(2)
    plt.plot(np.abs(np.array(lb_0) - np.array(ub_1)))
    plt.title("Absolute value of difference in bounds")
    plt.savefig("means_diference_2")