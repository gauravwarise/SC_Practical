import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# pip install networkx
# pip install scipy
# pip install scikit-fuzzy



# Define fuzzy variables
A = ctrl.Antecedent(np.arange(0, 11, 1), 'A')
B = ctrl.Antecedent(np.arange(0, 11, 1), 'B')
ratio = ctrl.Consequent(np.arange(0, 11, 1), 'ratio')

# Define fuzzy membership functions
A['low'] = fuzz.trimf(A.universe, [0, 0, 5])
A['high'] = fuzz.trimf(A.universe, [5, 10, 10])
B['low'] = fuzz.trimf(B.universe, [0, 0, 5])
B['high'] = fuzz.trimf(B.universe, [5, 10, 10])
ratio['small'] = fuzz.trimf(ratio.universe, [0, 0, 5])
ratio['large'] = fuzz.trimf(ratio.universe, [5, 10, 10])

# Define fuzzy rules
rule1 = ctrl.Rule(A['low'] & B['high'], ratio['small'])
rule2 = ctrl.Rule(A['high'] & B['low'], ratio['large'])

# Control system
ratio_ctrl = ctrl.ControlSystem([rule1, rule2])
ratio_sim = ctrl.ControlSystemSimulation(ratio_ctrl)

# Input values
A_value = 7
B_value = 3
ratio_sim.input['A'] = A_value
ratio_sim.input['B'] = B_value

# Compute and output the result
ratio_sim.compute()
print(f"Ratio for A = {A_value} and B = {B_value} is: {ratio_sim.output['ratio']:.2f}")
