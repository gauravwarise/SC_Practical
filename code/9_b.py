import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Fuzzy variables
service_quality = ctrl.Antecedent(np.arange(0, 11, 1), 'service_quality')
bill_amount = ctrl.Antecedent(np.arange(0, 101, 1), 'bill_amount')
tip_amount = ctrl.Consequent(np.arange(0, 21, 1), 'tip_amount')

# Membership functions
service_quality['poor'] = fuzz.trimf(service_quality.universe, [0, 0, 5])
service_quality['excellent'] = fuzz.trimf(service_quality.universe, [5, 10, 10])
bill_amount['low'] = fuzz.trimf(bill_amount.universe, [0, 0, 50])
bill_amount['high'] = fuzz.trimf(bill_amount.universe, [50, 100, 100])
tip_amount['small'] = fuzz.trimf(tip_amount.universe, [0, 0, 5])
tip_amount['large'] = fuzz.trimf(tip_amount.universe, [5, 10, 20])

# Fuzzy rules
rule1 = ctrl.Rule(service_quality['poor'] & bill_amount['low'], tip_amount['small'])
rule2 = ctrl.Rule(service_quality['excellent'] & bill_amount['high'], tip_amount['large'])

# Control system
tip_ctrl = ctrl.ControlSystem([rule1, rule2])
tip_sim = ctrl.ControlSystemSimulation(tip_ctrl)

# Inputs
tip_sim.input['service_quality'] = 7  # Example: excellent service
tip_sim.input['bill_amount'] = 60     # Example: moderate bill amount

# Compute the output
tip_sim.compute()

# Output the result
print(f"Suggested Tip Amount: {tip_sim.output['tip_amount']:.2f}")
