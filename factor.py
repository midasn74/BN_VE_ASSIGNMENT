import pandas as pd

class Factor:
    action_calls = {
        'multiply': 0,
        'sum_out': 0,
        'reduce': 0,
        'maximize': 0
    }

    def __init__(self, variables, cpt):
        self.variables = variables
        self.cpt = cpt

    def multiply(self, other):
        common_vars = list(set(self.variables).intersection(set(other.variables)))
        
        if not common_vars:
            new_cpt = self.cpt.assign(key=1).merge(other.cpt.assign(key=1), on='key').drop('key', axis=1)
        else:
            new_cpt = self.cpt.merge(other.cpt, on=common_vars)
        
        new_cpt['prob'] = new_cpt['prob_x'] * new_cpt['prob_y']
        new_cpt = new_cpt.drop(['prob_x', 'prob_y'], axis=1)
        
        return Factor(list(set(self.variables).union(set(other.variables))), new_cpt)
    
    def sum_out(self, variable):
        Factor.action_calls['sum_out'] += 1
        remaining_vars = [v for v in self.variables if v != variable]
        
        if not remaining_vars:
            new_cpt = pd.DataFrame({'prob': [self.cpt['prob'].sum()]})
        else:
            new_cpt = self.cpt.groupby(remaining_vars, as_index=False)['prob'].sum()
            
        return Factor(remaining_vars, new_cpt)

    def reduce(self, variable, value):
        Factor.action_calls['reduce'] += 1
        if variable not in self.variables:
            return self
        
        new_cpt = self.cpt[self.cpt[variable] == value].copy()
        new_cpt = new_cpt.drop(columns=[variable])
        new_vars = [v for v in self.variables if v != variable]
        
        return Factor(new_vars, new_cpt)
    
    def maximize(self, variable):
        Factor.action_calls['maximize'] += 1
        
        remaining_vars = [v for v in self.variables if v != variable]
        
        if not remaining_vars:
            max_idx = self.cpt['prob'].idxmax()
            max_prob = self.cpt.loc[max_idx, 'prob']
            best_val = self.cpt.loc[max_idx, variable]
            
            new_cpt = pd.DataFrame({'prob': [max_prob]})
            argmax_df = pd.DataFrame({variable: [best_val]})
            
            return Factor(remaining_vars, new_cpt), argmax_df
            
        else:
            max_indices = self.cpt.groupby(remaining_vars)['prob'].idxmax()
            
            max_rows = self.cpt.loc[max_indices].reset_index(drop=True)
            
            new_cpt = max_rows[remaining_vars + ['prob']].copy()
            
            argmax_df = max_rows[remaining_vars + [variable]].copy()
            
            return Factor(remaining_vars, new_cpt), argmax_df

