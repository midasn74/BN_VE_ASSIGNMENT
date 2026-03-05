"""
For EM I didn't use the endorisk data,
as this took 20+ minutes for one iteration on my machine,
more in this in the report
"""

import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from read_bayesnet import BayesNet
from expectation_max import ExpectationMaximization 

def generate_earthquake_data(n_samples=1000):
    """
    Generates a dataset without the hidden variables for the given actual probabilities.
    """
    burglary = np.random.choice(['True', 'False'], size=n_samples, p=[0.01, 0.99])
    earthquake = np.random.choice(['True', 'False'], size=n_samples, p=[0.02, 0.98])
    
    alarm = []
    john_calls = []
    mary_calls = []

    for i in range(n_samples):
        b = burglary[i]
        e = earthquake[i]
        
        if b == 'True' and e == 'True':
            p_alarm = 0.95
        elif b == 'False' and e == 'True':
            p_alarm = 0.29
        elif b == 'True' and e == 'False':
            p_alarm = 0.94
        else: # False, False
            p_alarm = 0.001
            
        a = np.random.choice(['True', 'False'], p=[p_alarm, 1 - p_alarm])
        alarm.append(a)
        
        p_john = 0.90 if a == 'True' else 0.05
        john_calls.append(np.random.choice(['True', 'False'], p=[p_john, 1 - p_john]))
        
        p_mary = 0.70 if a == 'True' else 0.01
        mary_calls.append(np.random.choice(['True', 'False'], p=[p_mary, 1 - p_mary]))

    df = pd.DataFrame({
        'Burglary': burglary,
        'Earthquake': earthquake,
        'Alarm': alarm,
        'JohnCalls': john_calls,
        'MaryCalls': mary_calls
    })

    # Hide Burglary and Earthquake
    df_hidden = df.copy()
    df_hidden['Burglary'] = np.nan
    df_hidden['Earthquake'] = np.nan
    
    return df_hidden

def randomize_hidden_cpts(net, hidden_vars):
    """Initializes CPTs of hidden variables with random values."""
    for var in hidden_vars:
        df = net.probabilities[var]
        if len(net.parents[var]) == 0:
            rand_vals = np.random.rand(len(df))
            df['prob'] = rand_vals / rand_vals.sum()
        else:
            df['prob'] = np.random.rand(len(df))
            parents = net.parents[var]
            df['prob'] = df.groupby(parents)['prob'].transform(lambda x: x / x.sum())
    return net


if __name__ == '__main__':
    # # Generate data
    # data = generate_earthquake_data()
    # data.to_csv('data/earthquake_test_data.csv', index=False)
    # print("Dataset generated")

    # Settings - run EM
    NETWORK_FILE = 'networks/earthquake.bif'
    DATA_FILE = 'data/earthquake_test_data.csv'
    HIDDEN_NODES = ['Burglary', 'Earthquake']
    
    data = pd.read_csv(DATA_FILE)
    
    data_clean = data.drop(columns=HIDDEN_NODES)
    
    for col in data_clean.columns:
        data_clean[col] = data_clean[col].astype(str).str.capitalize()

    np.random.seed(42)
    for i in range(5):
        print(f"\n--- TRIAL {i+1}: Random Initialization ---")
        
        net = BayesNet(NETWORK_FILE)
        net = randomize_hidden_cpts(net, HIDDEN_NODES)
        
        em = ExpectationMaximization(net, HIDDEN_NODES)
        
        learned_net = em.run(data_clean, max_iterations=30, tolerance=1e-4)
        
        # Bonus point 😎
        em.to_bif(f'earthquake_learned_trial_{i+1}.bif')