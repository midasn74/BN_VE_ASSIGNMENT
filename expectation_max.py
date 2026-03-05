import pandas as pd
import numpy as np
import datetime
from factor import Factor
from variable_elim import VariableElimination, minimum_neighbors_heuristic, create_logger

class ExpectationMaximization:
    """
    Class for Expectation Maximization
    """
    def __init__(self, network, hidden_vars):
        """
        Initialize EM algorithm given the specified network and hidden variables.
        """
        self.network = network
        self.hidden_vars = hidden_vars
        self.ve = VariableElimination(network)
        self.logger = create_logger("expectation_maximization", "em_run")

    def run(self, data, max_iterations=10, tolerance=1e-4):
        """
        Use the expectation maximization algorithm to find parameters of the specified network

        Input:
            data:           Observed data, might contain missing data for the hidden variables
            max_iterations: Max amount of full iterations to complete, 
                            ignoring the tolerance once this amount is reached
            tolerance:      Convergence treshold, if max change in probabilities is < tolerance
                            we consider it to have converged

        Output:             The resulting network with the estimated and inferred probabilities
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        def log(message):
            self.logger.info(message)

        log(f"--- Expectation Maximization Run: {timestamp} ---")
        log(f"Hidden Variables: {self.hidden_vars}")
        log(f"Max Iterations: {max_iterations}, Tolerance: {tolerance}")

        for i in range(max_iterations):
            log(f"\n--- Starting Iteration {i+1} ---")
            
            # E-step
            expected_counts = self._compute_expected_counts(data, log)
            
            # M-step
            new_probabilities = self._update_parameters(expected_counts)
            
            # Convergence
            converged, max_diff = self._has_converged(self.network.probabilities, new_probabilities, tolerance)
            log(f"Iteration {i+1} Max probability change: {max_diff:.6f}")
            
            self.network.probabilities = new_probabilities
            
            if converged:
                log(f"Converged at iteration {i+1}")
                break
        
        log("--- EM Run Complete ---")
        return self.network

    def _compute_expected_counts(self, data, log_func):
        counts = {node: self.network.probabilities[node].copy() for node in self.network.nodes}
        for node in counts:
            counts[node]['prob'] = 0.0

        unique_data = data.groupby(list(data.columns)).size().reset_index(name='row_count')
        total_patterns = len(unique_data)
        global_cache = {}

        log_func(f"Processing {total_patterns} unique data patterns...")

        for idx, row in unique_data.iterrows():
            weight = row['row_count']
            evidence_dict = row.drop('row_count').to_dict()
            evidence_items = frozenset(evidence_dict.items())

            for node in self.network.nodes:
                parents = self.network.parents[node]
                family = tuple(sorted([node] + parents))
                cache_key = (family, evidence_items)
                
                if cache_key not in global_cache:
                    relevant_evidence = {k: v for k, v in evidence_dict.items() if k not in family}
                    res = self.ve.run(list(family), relevant_evidence, minimum_neighbors_heuristic)
                    global_cache[cache_key] = res
                
                joint_prob_df = global_cache[cache_key]

                if not joint_prob_df.empty:
                    temp_merge = counts[node].merge(
                        joint_prob_df, on=list(family), how='left', suffixes=('', '_new')
                    )
                    counts[node]['prob'] += temp_merge['prob_new'].fillna(0) * weight

        return counts

    def _update_parameters(self, counts):
        new_probs = {}
        for node, df in counts.items():
            parents = self.network.parents[node]
            if not parents:
                total = df['prob'].sum()
                df['prob'] = df['prob'] / total if total > 0 else 1.0 / len(df)
            else:
                sum_per_parent = df.groupby(parents)['prob'].transform('sum')
                df['prob'] = df['prob'] / sum_per_parent.replace(0, np.nan)
                df['prob'] = df['prob'].fillna(1.0 / len(df[node].unique()))
                
            new_probs[node] = df
        return new_probs

    def _has_converged(self, old_probs, new_probs, tol):
        max_diff = 0
        for node in old_probs:
            diff = np.abs(old_probs[node]['prob'].values - new_probs[node]['prob'].values).max()
            max_diff = max(max_diff, diff)
        return max_diff < tol, max_diff

    def to_bif(self, filename):
        """
        Exports network to a bif file for bonus point 
        """
        self.logger.info(f"Exporting network to {filename}")
        with open(filename, 'w') as f:
            f.write(f"network unknown {{\n}}\n")

            for node in self.network.nodes:
                states = self.network.probabilities[node][node].unique()
                states_str = ", ".join([str(s) for s in states])
                f.write(f"variable {node} {{\n")
                f.write(f"  type discrete [ {len(states)} ] {{ {states_str} }};\n")
                f.write(f"}}\n")

            for node in self.network.nodes:
                parents = self.network.parents[node]
                df = self.network.probabilities[node]
                
                if not parents:
                    probs = ", ".join([f"{p:.6f}" for p in df['prob']])
                    f.write(f"probability ( {node} ) {{\n")
                    f.write(f"  table {probs};\n")
                    f.write(f"}}\n")
                else:
                    parent_str = ", ".join(parents)
                    f.write(f"probability ( {node} | {parent_str} ) {{\n")
                    for combo, group in df.groupby(parents):
                        combo_str = str(combo) if isinstance(combo, tuple) else f"({combo})"
                        probs = ", ".join([f"{p:.6f}" for p in group['prob']])
                        f.write(f"  {combo_str} {probs};\n")
                    f.write(f"}}\n")
        self.logger.info("BIF export complete.")