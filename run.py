from read_bayesnet import BayesNet
from variable_elim import *

if __name__ == '__main__':
    NETWORK_FILE = 'networks/earthquake.bif'

    ### Evidence and heuristics, used for both VE and MAP ###
    # Simple example
    EVIDENCE = {'Earthquake': 'False', 'Burglary': 'True'}
    # More complex example, used to test limits of MAP for the bonus point
    # EVIDENCE = {'MedCost': 'Thousand', 'ThisCarDam': 'Severe', 'OtherCarCost': 'Thousand'}

    heuristics = {
        'Least incoming arcs': least_incoming_arcs_heuristic,
        'Minimum neighbors': minimum_neighbors_heuristic,
        'Nodes in given order': None
    }

    ### Variable Elimination ###
    RUN_VE = True
    VALIDATE_VE = True
    # Simple example
    QUERY_VE = 'JohnCalls'
    # More complex example
    # QUERY_VE = 'Accident'

    ### Maximum a Posteriori ###
    RUN_MAP = True
    RUN_MOCK_ONLY = False
    VALIDATE_MAP = True
    # Simple example
    QUERY_MAP = ['JohnCalls', 'MaryCalls']
    # More complex example, used to test limits of MAP for the bonus point
    # QUERY_MAP = ['Accident', 'DrivingSkill']


    if RUN_VE:
        print("\n" + "="*40)
        print("VE")
        print("="*40)
        for name, heuristic in heuristics.items():
            print("\n" + "="*40)
            print(f"VE with heuristic: {name}")
            print("="*40)
            net = BayesNet(NETWORK_FILE) 
            
            ve = VariableElimination(net)

            if (name == "Nodes in given order"):
                elim_order = net.nodes
            else:
                elim_order = heuristic(net)

            print(ve.run(QUERY_VE, EVIDENCE, elim_order))

            print(f"\nFactor action calls for heuristics {name}: {Factor.action_calls}")

        if VALIDATE_VE:
            print("\n" + "="*40)
            print("VE VALIDATION")
            print("="*40)

            from pgmpy.readwrite import BIFReader
            from pgmpy.inference import VariableElimination as PgmpyVE

            reader = BIFReader(NETWORK_FILE)
            pgmpy_model = reader.get_model()

            pgmpy_infer = PgmpyVE(pgmpy_model)

            pgmpy_result = pgmpy_infer.query(variables=[QUERY_VE], evidence=EVIDENCE)

            print(pgmpy_result)


    if RUN_MAP:
        print("\n" + "="*40)
        print("MAP")
        print("="*40)
        for name, heuristic in heuristics.items():

            print("\n" + "="*40)
            print(f"MAP with heuristic: {name}")
            print("="*40)
            net = BayesNet(NETWORK_FILE) 
            ve = MAPVariableElimination(net)

            Factor.action_calls = {'multiply': 0, 'sum_out': 0, 'reduce': 0, 'maximize': 0}

            if (name == "Nodes in given order"):
                full_order = net.nodes
            else:
                full_order = heuristic(net)

            sum_order = [v for v in full_order if v not in QUERY_MAP and v not in EVIDENCE]
            max_order = [v for v in full_order if v in QUERY_MAP]

            if not RUN_MOCK_ONLY:
                map_prob, map_assignment = ve.run_map(QUERY_MAP, EVIDENCE, (sum_order, max_order))

                print(f"\n--- Results for Heuristic: {name} ---")
                print(f"MAP Probability: {map_prob}")
                print(f"MAP Assignment:  {map_assignment}")
                print(f"Factor action calls: {Factor.action_calls}")

            metrics_lia = ve.mock_run(QUERY_MAP, EVIDENCE, (sum_order, max_order))
            print(f"Mock run metrics: {metrics_lia}")

        if VALIDATE_MAP:
            print("\n" + "="*40)
            print("MAP VALIDATION")
            print("="*40)

            from pgmpy.readwrite import BIFReader
            from pgmpy.inference import VariableElimination as PgmpyVE

            reader = BIFReader(NETWORK_FILE)
            pgmpy_model = reader.get_model()

            pgmpy_infer = PgmpyVE(pgmpy_model)

            pgmpy_result = pgmpy_infer.map_query(variables=QUERY_MAP, evidence=EVIDENCE)

            print(f"pgmpy MAP Assignment: {pgmpy_result}")
