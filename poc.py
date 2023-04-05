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


def simulate_data(dataset_size: int, time_steps:int):
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
    counterfactual_1 = []
    p_x_0 = [0.5, 0.3, 0.2]
    for t in range(time_steps):
        X = np.random.choice(a=[0, 1, 2], size=dataset_size, p=p_x_0)

        pi_A = scipy.special.expit(X)
        A = 1*(pi_A > np.random.uniform(size=dataset_size))

        mu = scipy.special.expit(2*A -  X)
        y = 1*(mu > np.random.uniform(size=dataset_size))
        
        mu_1 = scipy.special.expit(2 - X)
        y_1 = 1*(mu > np.random.uniform(size=dataset_size))
        
        mean_1 = y[A == 1].mean()
        mean_0 = y[A == 0].mean()
        
        counterfactual_1.append(y_1.mean())
        means_1.append(mean_1)
        means_0.append(mean_0)

    return means_1, means_0

if __name__ == '__main__':

    means_1, means_0 = simulate_data(DATASE_SIZE, TIME_STEPS)

    lb_0, ub_0 = hedged_cs(means_1)
    lb_1, ub_1 = hedged_cs(means_0)

    plt.plot(np.arange(TIME_STEPS), lb_0, c='navy', label='A = 0')
    plt.plot(np.arange(TIME_STEPS), ub_0, c='navy')
    plt.plot(np.arange(TIME_STEPS), lb_1, c='tab:olive', label='A = 1')
    plt.plot(np.arange(TIME_STEPS), ub_1, c='tab:olive')
    #plt.axhline(E_0, ls='--', c='k')
    #plt.axhline(E_1, ls='--', c='r')
    plt.title("Confidence sequence for E[y|A]_t")
    plt.legend()
    plt.savefig("confidence_sequence")