import optuna
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# importing the data set
data_url = ""
data_set = pd.read_csv(data_url)

# expected inputs and outputs
expected_inputs = ""
expected_output = ""

# make the test and training data
(test_input, test_output), (train_input, train_output) = train_test_split(expected_inputs, expected_output,
                                                                          test_size=.2, random_state=42)
# making the ANN models for finding the regression
def objective(trail):
    num_layers = trail.suggest_int('num_layers', 1, 10)
    num_nodes = trail.suggest_int('num_nodes', 16, 128)
    dropout_rate = trail.suggest_int('dropout_rate', 0.0, 0.5)
    learning_rate = trail.suggest_logunifrom('learning_rate', 1e-5, 1e2)

    model = Sequential()
    model.add(len(expected_inputs))
    for _ in range(num_layers):
        model.add(Dense(num_nodes, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(expected_output, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_input, train_output, epochs=5, validation_split=0.2, verbose=0)
    model.summary()
    val_loss, val_acc = model.evaluate(test_input, test_output, verbose=0)
    plt.plot(val_loss)
    plt.plot(val_acc)
    plt.show()
    return val_acc


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials = 10)

best_params = study.best_params
print("Best hyper-parameters : ", best_params)

best_model = Sequential()
best_model.add(len(expected_inputs))
for _ in range(best_params['num_layers']):
    best_model.add(Dense(best_params['num_nodes'], activation='relu'))
    best_model.add(Dropout(best_params['dropout_rate']))
best_model.add(Dense(expected_output, activation='softmax'))
best_model.compile(optimizer=Adam(best_params['learning_rate']),
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
best_model.fit(train_input, train_output, epochs=10, validation_split=0.2)
best_model.summary()
best_model.save('best_model.h5')
test_loss, test_acc = best_model.evaluate(test_input, test_output)

plt.plot(test_loss)
plt.plot(test_acc)
plt.show()

print("Test accuracy:", test_acc)

best_model.Predict(test_input)

