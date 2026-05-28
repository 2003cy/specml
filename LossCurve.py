import numpy as np
import matplotlib.pyplot as plt

# Load the loss
loss = np.load('loss_curve 20260528 n16000 lr5e-4.npy')

# Plot loss

plt.title('Test 2: Loss Over 16000 Steps with a LR of 5e-4')
plt.xlabel('Number of Steps')
plt.ylabel('Loss')
plt.plot(loss, color = 'pink')
plt.savefig('Test 2 20260528 n16000 lr5e-4')
plt.show()



