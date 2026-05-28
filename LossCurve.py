import numpy as np
import matplotlib.pyplot as plt

# Load the loss
loss = np.load('loss_curve 20260528 n16000 lr1e-4 2restarts.npy')

# Plot loss

plt.title('Test 4: Loss Over 16000 Steps with a LR of 1e-4 over 2 Cosine Annealing Restarts')
plt.xlabel('Number of Steps')
plt.ylabel('Loss')
plt.plot(loss, color = 'pink')
plt.savefig('Test 4 20260528 n16000 lr1e-4 2restarts.')
plt.show()



