import numpy as np 
import os 
import pickle
from scipy.stats import permutation_test

def test_by_betting(seq1, seq2, alpha=0.05): 
    # Construct wealth process from sequence X using ONS 
    
    wealth = 1
    wealth_hist = [1] 
    const = 2 / (2 - np.log(3))
    lambd = 0 
    zt2 = 0 
    for t in range(1,min(len(seq1), len(seq2))): 
                
        St = 1 - lambd*(seq1[t] - seq2[t])
        wealth = wealth * St 
        wealth_hist.append(wealth)
        if wealth > 1/alpha: 
            #print(f"Reject at time {t}")
            return wealth_hist, 'reject'
        
        # Update lambda via ONS  
        g = seq1[t] - seq2[t]
        z = g / (1 - lambd*g)
        zt2 += z**2
        lambd = max(min(lambd - const*z/(1 + zt2), 1/2), -1/2)
        
    U = np.random.uniform()
    if wealth > U/alpha: 
        return wealth_hist, 'reject'
    
    return wealth_hist, 'sustain'

def betting_experiment(seq1, seq2, alphas, iters=10): 

    results = []
    for _ in range(iters): 
        np.random.shuffle(seq1)
        np.random.shuffle(seq2)
        taus = []
        for alpha in alphas: 
            wealth, _ = test_by_betting(seq1, seq2, alpha=alpha)
            taus.append(len(wealth))
        results.append(taus)
        
    return results
            
def test_stat(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

def perm_test(seq1, seq2, p):    
    
    # because our statistic is vectorized, we pass `vectorized=True`
    # `n_resamples=np.inf` indicates that an exact test is to be performed
    res = permutation_test((seq1, seq2), test_stat, vectorized=True,
                           n_resamples=2000, alternative='two-sided')
    # print(res.pvalue)
    if res.pvalue <= p: 
        return True 
    return False
    
def seq_perm_test(seq1, seq2, p=0.05, k=100, bonferroni=False): 
    
    l = min(len(seq1), len(seq2))
    for i in range(int(l/k)): 
        pi = p / 2**(i+1) if bonferroni else p
        if perm_test(seq1[i*k:k*(i+1)], seq2[i*k:k*(i+1)], pi): 
            # print('Reject')
            return k*(i+1), 'reject'
    return l, 'sustain'

def seq_perm_test_experiment(seq1, seq2, alphas, iters=10, k=100, bonferroni=False): 
    
    results = []
    for _ in range(iters): 
        taus = []
        np.random.shuffle(seq1)
        np.random.shuffle(seq2)
        for alpha in alphas: 
            steps, _ = seq_perm_test(seq1, seq2, p=alpha, k=k, bonferroni=bonferroni)
            taus.append(steps)
        results.append(taus)
        
    return results
    

def get_mean_std(arr): 
    return np.mean(arr, axis=0), np.std(arr, axis=0)

def plt_mean_std(ax, arr, alphas, label, color='navy', plot_std=True): 
    mean, std = get_mean_std(arr)
    ax.plot(alphas, mean, lw=2, label=label, c=color)
    if plot_std: 
        ax.fill_between(alphas, mean-std, mean+std, alpha=0.05, color=color)

def save_results(name, data): 
    
    n = np.sum([x.startswith(name) for x in os.listdir('data/')])
    with open(f'data/{name}-{n}.p', 'wb') as f: 
        pickle.dump(data, f)
              