# Radiation Source Identification with Deep Learning

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

# Getting the data ready
data_source_1 = np.transpose(np.loadtxt("source0.txt"))
data_source_2 = np.transpose(np.loadtxt("source1.txt"))

# Plot 3 curves for each source
time = range(0, 300)
first_curve = 12
second_curve = 250
third_curve = 480

plt.figure(dpi=1000)
plt.figure(1)
plt.plot(time, data_source_1[first_curve-1, :], "b-", label="Source 1")
plt.plot(time, data_source_2[first_curve-1, :], "r-", label="Source 2")
plt.xlabel("Time")
plt.ylabel("Electric field [V/m]")
plt.title("Measured Electric field for Source 1 and Source 2 ($%s^{th}$ curves)" % first_curve)
plt.legend()

plt.figure(dpi=1000)
plt.figure(2)
plt.plot(time, data_source_1[second_curve-1, :], "b-", label="Source 1")
plt.plot(time, data_source_2[second_curve-1, :], "r-", label="Source 2")
plt.xlabel("Time")
plt.ylabel("Electric field [V/m]")
plt.title("Measured Electric field for Source 1 and Source 2 ($%s^{th}$ curves)" % second_curve)
plt.legend()

plt.figure(dpi=1000)
plt.figure(3)
plt.plot(time, data_source_1[third_curve-1, :], "b-", label="Source 1")
plt.plot(time, data_source_2[third_curve-1, :], "r-", label="Source 2")
plt.xlabel("Time")
plt.ylabel("Electric field [V/m]")
plt.title("Measured Electric field for Source 1 and Source 2 ($%s^{th}$ curves)" % third_curve)
plt.legend()

# Add the output "0" to "data_source_1"
data_source_1 = np.c_[data_source_1, np.zeros(500)]

# Add the output "1" to "data_source_2"
data_source_2 = np.c_[data_source_2, np.ones(500)]

# Concatenate "data_source_1" and "data_source_2"
data_sources_1_2 = np.concatenate((data_source_1, data_source_2), axis=0)

# Shuffle lines of "data_sources_1_2"
data_sources_1_2_shuffled = np.random.permutation(data_sources_1_2)

# Create input matrix
X = data_sources_1_2_shuffled[:, :300]  # jusqu'à l'avant-dernière colonne

# Create output matrix
Y = data_sources_1_2_shuffled[:, 300:]  # dernière colonne

# Scale input data
X = X*1000

# Split training data and validation data
X_train = X[:700, :]  # 70 %
X_val = X[700:, :]  # 30 %

Y_train = Y[:700, :]  # 70 %
Y_val = Y[700:, :]  # 30 %

# Design the network
model = Sequential()
model.add(Dense(32, input_dim=300, kernel_initializer="uniform", activation="relu"))
model.add(Dense(32, kernel_initializer="uniform", activation="relu"))
model.add(Dense(32, kernel_initializer="uniform", activation="relu"))
model.add(Dense(1, kernel_initializer="uniform", activation="sigmoid")) # sortie de dimension 1, car c'est un booléen ("0" ou "1")
model.summary()

# Compile and fit the model
epo = 50
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
hist = model.fit(X_train, Y_train, epochs=epo, batch_size=10, validation_data=(X_val, Y_val))
train_loss = hist.history["loss"]
valid_loss = hist.history["val_loss"]
ep = range(1, epo+1)

plt.figure(dpi=1000)
plt.figure(4)
plt.plot(ep, train_loss, "b-", label="Training error")
plt.plot(ep, valid_loss, "r-", label="Validation error")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.title("Training and validation losses")
plt.legend()

# Evaluate the model
Y_pred = model.predict(X_val)

for i in range(0, np.size(Y_pred)):
    if Y_pred[i, 0] >= 0.5:
        Y_pred[i, 0] = 1
    elif Y_pred[i, 0] < 0.5:
        Y_pred[i, 0] = 0

# Calculate success rate
success = 0

for i in range(0, np.size(Y_pred)):
    if Y_pred[i, 0] == Y_val[i, 0]:
        success += 1

accuracy = (success/300)*100
print(accuracy)

# Show figures
plt.show()
