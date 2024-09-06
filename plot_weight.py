import matplotlib.pyplot as plt
import numpy as np

"""Using the weight, activation and loss array obtained during training to plot them."""

weight_array = np.load('weight_array.npy')
activation_array = np.load('activation_array.npy')
loss_array = np.load('loss_array.npy')
fig, ax = plt.subplots(1,2)

ax[0].plot(weight_array)
ax[0].set_title('Weights')
ax[0].set_xlabel('Steps')
ax[0].set_ylabel('Magnitude')
ax[1].plot(activation_array)
ax[1].set_title('Activations')
ax[1].set_xlabel('Steps')
ax[1].set_ylabel('Magnitude')
# ax[2].plot(loss_array)
# ax[2].set_title('Loss')
# ax[2].set_xlabel('Steps')
# ax[2].set_ylabel('Magnitude')

plt.show()
