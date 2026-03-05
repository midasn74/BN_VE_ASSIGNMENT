"""
@Author: Joris van Vugt, Moira Berens, Leonieke van den Bulk

Class for the implementation of the variable elimination algorithm.

"""
import pandas as pd
import numpy as np
import logging
import datetime
from factor import Factor
from logger import create_logger

# Bonus point 😎
def least_incoming_arcs_heuristic(network):
    return sorted(network.nodes, key=lambda n: len(network.parents[n]))

# Bonus point 😎
def minimum_neighbors_heuristic(network):
    neighbor_counts = {node: len(network.parents[node]) for node in network.nodes}
    
    for node, parents in network.parents.items():
        for parent in parents:
            neighbor_counts[parent] += 1
            
    return sorted(network.nodes, key=lambda n: neighbor_counts[n])

class VariableElimination():
    """
    Class for Regular variable elimination
    """

    def __init__(self, network):
        """
        Initialize variable elimination algorithm given the specified network.
        """
        self.network = network
        self.logger = create_logger("variable_elimination", "ve_run")

    def run(self, query, observed, elim_order):
        """
        Use the variable elimination algorithm to find out the probability
        distribution of the query variable given the observed variables

        Input:
            query:      The query variable
            observed:   A dictionary of the observed variables {variable: value}
            elim_order: Either a list specifying the elimination ordering
                        or a function that will determine an elimination ordering
                        given the network during the run

        Output: A variable holding the probability distribution
                for the query variable

        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        def log(message):
            self.logger.info(message)

        query_vars = [query] if isinstance(query, str) else query

        log(f"--- Variable Elimination Run: {timestamp} ---")
        log(f"Query: {query_vars}")
        log(f"Observed: {observed}")

        factors = []
        for node in self.network.nodes:
            cpt = self.network.probabilities[node]
            factors.append(Factor([c for c in cpt.columns if c != 'prob'], cpt))
        
        log(f"\nInitial Factors: {len(factors)}")
        for i, f in enumerate(factors):
            log(f"Factor {i} vars: {f.variables}")

        for var, val in observed.items():
            log(f"\n--- Processing Evidence: {var} = {val} ---")
            for i in range(len(factors)):
                if var in factors[i].variables:
                    factors[i] = factors[i].reduce(var, val)
                    log(f"Reduced factor {i}. New CPT head:\n{factors[i].cpt.head().to_string()}")

        if callable(elim_order):
            elim_order = elim_order(self.network)
        
        log(f"\nElimination Order: {elim_order}")

        for var in elim_order:
            if var in query_vars or var in observed:
                continue
            
            log(f"\n--- Eliminating variable: {var} ---")
            
            relevant_factors = []
            other_factors = []
            
            for f in factors:
                if var in f.variables:
                    relevant_factors.append(f)
                else:
                    other_factors.append(f)
            
            log(f"Factors containing {var}: {len(relevant_factors)}")
            
            if relevant_factors:
                product_factor = relevant_factors[0]
                for f in relevant_factors[1:]:
                    product_factor = product_factor.multiply(f)
                
                log(f"Product factor CPT head (pre-sum):\n{product_factor.cpt.head().to_string()}")
                
                summed_factor = product_factor.sum_out(var)
                log(f"Factor after summing out {var}:\n{summed_factor.cpt.to_string()}")
                
                factors = other_factors + [summed_factor]

        log("\n--- Finalizing Joint Query ---")
        if not factors:
            return pd.DataFrame()

        final_factor = factors[0]
        for f in factors[1:]:
            final_factor = final_factor.multiply(f)
        
        total_prob = final_factor.cpt['prob'].sum()
        if total_prob > 0:
            final_factor.cpt['prob'] = final_factor.cpt['prob'] / total_prob
        
        return final_factor.cpt


class MAPVariableElimination():
    """
    Class for MAP variable elimination
    """

    def __init__(self, network):
        """
        Initialize MAP variable elimination algorithm given the specified network.
        """
        self.network = network
        self.logger = create_logger("variable_elimination", "ve_run")

    def run_map(self, query_vars, observed, elim_order):
        """
        Use the variable elimination algorithm to compute the MAP (Maximum a Posteriori)

        Input:
            query_vars: A list of query variables, (hypotheses)
            observed:   A dictionary of the observed variables {variable: value}
            elim_order: Either a tuple of lists given in this ordering (sum_order, max_order)
                            or a function that will determine an elimination ordering

        Output: A tuple containing:
                1. The maximum joint probability
                2. A dictionary with the MAP assignment {variable: value}
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        def log(message):
            self.logger.info(message)

        if callable(elim_order):
            # Create order given the heuristic function for sum and max
            full_order = elim_order(self.network)
            sum_order = [v for v in full_order if v not in query_vars and v not in observed]
            max_order = [v for v in full_order if v in query_vars]
        else:
            # If a list is passed, use the list
            sum_order, max_order = elim_order

        log(f"--- MAP Inference Run: {timestamp} ---")
        log(f"Query Variables (H): {query_vars}")
        log(f"Observed Evidence (E): {observed}")
        log(f"Sum Ordering (I): {sum_order}")
        log(f"Max Ordering: {max_order}")

        factors = []
        for node in self.network.nodes:
            cpt = self.network.probabilities[node]
            factors.append(Factor([c for c in cpt.columns if c != 'prob'], cpt))
        
        for var, val in observed.items():
            log(f"\n--- Processing Evidence: {var} = {val} ---")
            for i in range(len(factors)):
                if var in factors[i].variables:
                    factors[i] = factors[i].reduce(var, val)
                    log(f"Reduced factor {i}. New CPT head:\n{factors[i].cpt.head().to_string()}")

        for var in sum_order:
            if var in observed or var in query_vars:
                continue
            
            log(f"\n--- Summing out intermediate variable: {var} ---")
            relevant_factors, other_factors = [], []
            
            for f in factors:
                if var in f.variables:
                    relevant_factors.append(f)
                else:
                    other_factors.append(f)
            
            if relevant_factors:
                product_factor = relevant_factors[0]
                for f in relevant_factors[1:]:
                    product_factor = product_factor.multiply(f)
                
                summed_factor = product_factor.sum_out(var)
                log(f"Factor after summing out {var}:\n{summed_factor.cpt.to_string()}")
                factors = other_factors + [summed_factor]

        traceback_list = []
        for var in max_order:
            if var in observed:
                continue
            
            log(f"\n--- Maximizing out query variable: {var} ---")
            relevant_factors, other_factors = [], []
            
            for f in factors:
                if var in f.variables:
                    relevant_factors.append(f)
                else:
                    other_factors.append(f)
            
            if relevant_factors:
                product_factor = relevant_factors[0]
                for f in relevant_factors[1:]:
                    product_factor = product_factor.multiply(f)
                
                maximized_factor, argmax_trace = product_factor.maximize(var)
                
                traceback_list.append(argmax_trace)
                
                log(f"Factor after maximizing out {var}:\n{maximized_factor.cpt.to_string()}")
                log(f"Argmax trace for {var} saved.")
                
                factors = other_factors + [maximized_factor]

        log("\n--- Finalizing MAP Probability ---")
        final_factor = factors[0]
        for f in factors[1:]:
            final_factor = final_factor.multiply(f)
        
        map_probability = final_factor.cpt['prob'].iloc[0]
        log(f"Maximum unnormalized joint probability: {map_probability}")

        log("\n--- Executing Traceback Phase ---")
        map_assignment = {}
        
        for argmax_trace in reversed(traceback_list):
            # The maximized variable we keep as the last value
            maximized_var = argmax_trace.columns[-1]
            remaining_vars = list(argmax_trace.columns[:-1])
            
            if not remaining_vars:
                best_val = argmax_trace[maximized_var].iloc[0]
            else:
                condition = pd.Series([True] * len(argmax_trace))
                for r_var in remaining_vars:
                    if r_var in map_assignment:
                        condition = condition & (argmax_trace[r_var] == map_assignment[r_var])
                
                best_val = argmax_trace.loc[condition, maximized_var].iloc[0]
                
            map_assignment[maximized_var] = best_val
            log(f"Traced back {maximized_var} = {best_val}")

        log(f"\nFinal MAP Assignment: {map_assignment}")
        
        return map_probability, map_assignment

    def mock_run(self, query_vars, observed, elim_order):
        """
        Simulates the MAP Variable Elimination to assess complexity metrics
        """
        import time
        start_time = time.time()
        
        if callable(elim_order):
            full_order = elim_order(self.network)
            sum_order = [v for v in full_order if v not in query_vars and v not in observed]
            max_order = [v for v in full_order if v in query_vars]
        else:
            sum_order, max_order = elim_order
            
        cardinalities = {}
        for node in self.network.nodes:
            cardinalities[node] = len(self.network.probabilities[node][node].unique())
            
        factors = []
        for node in self.network.nodes:
            factor_vars = set(self.network.probabilities[node].columns) - {'prob'}
            
            factor_vars = factor_vars - set(observed.keys())
            factors.append(factor_vars)
            
        max_factor_rows = 0
        total_multiplications = 0
        
        def calculate_factor_rows(factor_vars):
            size = 1
            for var in factor_vars:
                size *= cardinalities[var]
            return size
            
        def simulate_elimination(var, phase_name):
            nonlocal factors, max_factor_rows, total_multiplications
            relevant_factors = [f for f in factors if var in f]
            other_factors = [f for f in factors if var not in f]
            
            if relevant_factors:
                total_multiplications += len(relevant_factors) - 1
                
                new_factor_vars = set().union(*relevant_factors)
                
                intermediate_rows = calculate_factor_rows(new_factor_vars)
                if intermediate_rows > max_factor_rows:
                    max_factor_rows = intermediate_rows
                    
                new_factor_vars.remove(var)
                
                factors = other_factors + [new_factor_vars]

        for var in sum_order:
            simulate_elimination(var, "Sum")
            
        for var in max_order:
            simulate_elimination(var, "Max")
            
        end_time = time.time()

        self.logger.info(f"--- Mock Run Metrics ---")
        self.logger.info(f"Max factor size (rows): {max_factor_rows}")
        self.logger.info(f"Total simulated multiplications: {total_multiplications}")
        self.logger.info(f"Sum order: {sum_order}")
        self.logger.info(f"Max order: {max_order}")
        self.logger.info(f"Simulation time (ms): {(end_time - start_time) * 1000:.2f}")
        
        return {
            'max_factor_size_rows': max_factor_rows,
            'total_simulated_multiplications': total_multiplications,
            'sum_order': sum_order,
            'max_order': max_order,
            'simulation_time_ms': (end_time - start_time) * 1000
        }
    