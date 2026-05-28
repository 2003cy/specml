import numpy as np
import matplotlib.pyplot as plt

# Load the loss
loss = np.load('loss_curve 20260527 n8000 lr1e-4.npy')

# Plot loss

plt.title('Test 1: Loss Over 8000 Steps with a LR of 1e-4')
plt.xlabel('Number of Steps')
plt.ylabel('Loss')
plt.plot(loss, color = 'pink')
plt.savefig('Test 1 20260527 n8000 lr1e-4')
plt.show()



