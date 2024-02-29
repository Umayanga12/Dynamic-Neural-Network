import pandas as pd

data = pd.read_csv("./machine_components_data.csv")
#print(data.columns)

# Create a list of columns to drop for expected_input
columns_to_drop_output = ['Capacity', 'Failure_Rate', 'Setup_Time', 'Quality_Parameter']
expected_output = data.drop(columns_to_drop_output, axis=1)

print(expected_output.columns)

# Create a list of columns to drop for expected_output
columns_to_drop_input = ['Processing_Time', 'Maintenance_Interval', 'Maintenance_Duration', 
                          'Failure_Rate', 'Energy_Consumption', 'Availability']
expected_input = data.drop(columns_to_drop_input, axis=1)

print(expected_input.columns)
