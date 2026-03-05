"""
@Author: Joris van Vugt, Moira Berens, Leonieke van den Bulk

Representation of a Bayesian network read in from a .bif file.

"""

import pandas as pd
import itertools

class BayesNet():
    """
    This class represents a Bayesian network.
    It can read files in a .bif format (if the formatting is
    along the lines of http://www.bnlearn.com/bnrepository/)

    Uses pandas DataFrames for representing conditional probability tables
    """

    # Possible values per variable
    values = {}

    # Probability distributions per variable
    probabilities = {}

    # Parents per variable
    parents = {}

    def __init__(self, filename):
        """
        Construct a bayesian network from a .bif file

        """
        with open(filename, 'r') as file:
            line_number = 0
            for line in file:
                if line.startswith('network'):
                    self.name = ' '.join(line.split()[1:-1])
                elif line.startswith('variable'):
                    self.parse_variable(line_number, filename)
                elif line.startswith('probability'):
                    self.parse_probability(line_number, filename)
                line_number = line_number + 1


    def parse_probability(self, line_number, filename):
        line = open(filename, 'r').readlines()[line_number]

        variable, parents = self.parse_parents(line)
        next_line = open(filename, 'r').readlines()[line_number + 1].strip()

        if next_line.startswith('table'):
            comma_sep_probs = next_line.split('table')[1].split(';')[0].strip()
            probs = [float(p) for p in comma_sep_probs.replace(',', ' ').split()]
            
            if not parents:
                df = pd.DataFrame(columns=[variable, 'prob'])
                for value, p in zip(self.values[variable], probs):
                    df.loc[len(df)] = [value, p]
                self.probabilities[variable] = df
            else:
                parent_values = [self.values[p] for p in parents]
                
                parent_combos = list(itertools.product(*parent_values))
                
                data = []
                prob_idx = 0
                
                for combo in parent_combos:
                    for val_child in self.values[variable]:
                        if prob_idx < len(probs):
                            p = probs[prob_idx]
                            prob_idx += 1
                        else:
                            p = 0.0 # Default fallback
                            
                        row = [val_child] + list(combo) + [p]
                        data.append(row)
                
                # Create the DataFrame in one go
                df = pd.DataFrame(data, columns=[variable] + parents + ['prob'])
                self.probabilities[variable] = df

        else:
            df = pd.DataFrame(columns=[variable] + parents + ['prob'])

            with open(filename, 'r') as file:
                for i in range(line_number + 1):
                    file.readline()
                for line in file:
                    if '}' in line:
                        break
                    
                    comma_sep_values = line.split('(')[1].split(')')[0]
                    values = [v.strip() for v in comma_sep_values.replace(',', ' ').split()]

                    comma_sep_probs = line.split(')')[1].split(';')[0].strip()
                    probs = [float(p) for p in comma_sep_probs.replace(',', ' ').split()]

                    for value, p in zip(self.values[variable], probs):
                        df.loc[len(df)] = [value] + values + [p]

            self.probabilities[variable] = df

    def parse_variable(self, line_number, filename):
        """
        Parse the name of a variable and its possible values
        """
        # FIX: Remove " chars from the variable name
        variable = open(filename, 'r').readlines()[line_number].split()[1].replace('"', '') 
        
        line = open(filename, 'r').readlines()[line_number+1]
        start = line.find('{') + 1
        end = line.find('}')
        # FIX: Remove " chars from values
        values = [value.strip().replace('"', '') for value in line[start:end].replace(',', ' ').split()]
        self.values[variable] = values
    
    def parse_parents(self, line):
        """
        Find out what variables are the parents
        Returns the variable and its parents
        """
        start = line.find('(') + 1
        end = line.find(')')
        content = line[start:end].replace(',', ' ')
        
        # FIX: Remove " chars from everything in the content string first
        content = content.replace('"', '')
        
        if '|' in content:
            child, parents = content.split('|')
        else:
            # Format: Child Parent1 Parent2 (first item is child)
            tokens = content.split()
            child = tokens[0]
            parents = ' '.join(tokens[1:])
            
        variable = child.strip()
        self.parents[variable] = [v.strip() for v in parents.split()]
        
        return variable, self.parents[variable]

    @property
    def nodes(self):
        """Returns the names of the variables in the network"""
        return list(self.values.keys())
