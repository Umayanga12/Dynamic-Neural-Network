import optuna
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.losses import categorical_crossentropy
from tensorflow.keras.models import load_model
# importing the data set
data_url = "./machine_components_data.csv"
data_set = pd.read_csv(data_url)

data = pd.read_csv("./machine_components_data.csv")
#print(data.columns)

columns_to_drop_output = ['Capacity', 'Failure_Rate', 'Setup_Time', 'Quality_Parameter']
expected_output = data.drop(columns_to_drop_output, axis=1)

#print(expected_output.columns)

columns_to_drop_input = ['Processing_Time', 'Maintenance_Interval', 'Maintenance_Duration', 
                          'Failure_Rate', 'Energy_Consumption', 'Availability']
expected_input = data.drop(columns_to_drop_input, axis=1)

#print(expected_input.columns)

train_input, test_input, train_output, test_output = train_test_split(expected_input, expected_output,
                                                                      test_size=.2, random_state=42)
def objective(trial):
    num_layers = trial.suggest_int('num_layers', 1, 10)
    num_nodes = trial.suggest_int('num_nodes', 16, 128)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

    model = Sequential()
    model.add(Input(shape=(len(expected_input.columns),)))
    for _ in range(num_layers):
        model.add(Dense(num_nodes, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='linear'))  # Linear activation for regression

    model.compile(optimizer=Adam(learning_rate),
                  loss='mean_squared_error',  # Use appropriate loss for regression
                  metrics=['mean_absolute_error'])  # MAE as metric for regression
   # model.summary()
    model.fit(train_input, train_output, epochs=5, validation_split=0.2, verbose=0)
    # Get the validation loss and MAE
    val_loss, val_mae = model.evaluate(test_input, test_output, verbose=0)
    return val_mae  # Return MAE as the objective to minimize


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials = 10)

best_params = study.best_params
print("Best hyper-parameters : ", best_params)

# Define the input layer with appropriate input shape
input_layer = Input(shape=(len(expected_input.columns),))

best_model = Sequential()
best_model.add(Input(shape=(len(expected_input.columns),)))
for _ in range(best_params['num_layers']):
    best_model.add(Dense(best_params['num_nodes'], activation='relu'))
    best_model.add(Dropout(best_params['dropout_rate']))
best_model.add(Dense(len(expected_output.columns), activation='softmax'))
# best_model.compile(optimizer=Adam(best_params['learning_rate']),
#                    loss='sparse_categorical_crossentropy',
#                    metrics=['accuracy'])
best_model.compile(optimizer ='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#train_output_encoded = LabelEncoder.fit_transform(train_output)
best_model.fit(train_input, train_output, epochs=10, validation_split=0.2)
best_model.summary()
best_model.save('best_model.h5')
test_loss, test_acc = best_model.evaluate(test_input, test_output)

print("Test accuracy:", test_acc)
print(best_model.predict(test_input))

#loaded_model = load_model('best_model.h5')

# predictions = loaded_model.predict()

# print(predictions)