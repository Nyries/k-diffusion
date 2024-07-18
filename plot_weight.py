import matplotlib.pyplot as plt
import numpy as np

weight_array = np.load('weight_array.npy')
activation_array = np.load('activation_array.npy')
fig, ax = plt.subplots(1,2)
ax[0].plot(weight_array)
ax[0].set_title('Weights')
ax[0].set_xlabel('Steps')
ax[0].set_ylabel('Magnitude')
ax[1].plot(activation_array)
ax[1].set_title('Activation')
ax[1].set_xlabel('Steps')
ax[1].set_ylabel('Magnitude')

plt.show()