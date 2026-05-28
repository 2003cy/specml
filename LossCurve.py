import numpy as np
import matplotlib.pyplot as plt

# Load the loss
loss = np.load('loss_curve 20260528 n16000 lr5e-5 4restarts.npy')

# Plot loss

plt.title('Test 5: Loss Over 8000 Steps with a LR of 1e-5 over 4 Cosine Annealing Restarts')
plt.xlabel('Number of Steps')
plt.ylabel('Loss')
plt.plot(loss, color = 'pink')
plt.savefig('Test 5 20260528 n16000 lr5e-5 4restarts')
plt.show()



