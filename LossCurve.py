# import numpy as np
# import matplotlib.pyplot as plt

# # Load the loss
# loss = np.load('loss_curve 20260528 n20000 lr5e-4 4restarts.npy')

# # Plot loss

# plt.title('Test 6: Loss Over 20000 Steps with a LR of 1e-4 over 4 Cosine Annealing Restarts')
# plt.xlabel('Number of Steps')
# plt.ylabel('Loss')
# plt.yscale(f'log')
# plt.plot(loss[::100], color = 'pink')
# plt.savefig('Test 6 20260528 n20000 lr5e-4 4restarts')
# plt.show()



import numpy as np
import matplotlib.pyplot as plt

def smooth(y, window=50):
    return np.convolve(y, np.ones(window)/window, mode='valid')

files = {
    'Test 5: ps 20 o10': 'loss_curve 20260528 n20000 lr5e-4 4restarts.npy',
    'Test 6: ps 10 o2': 'loss_curve.npy'

}

plt.figure(figsize=(12, 6))

for label, filepath in files.items():
    loss = np.load(filepath)
    plt.plot(smooth(loss, window=50)[::100], label=label, alpha=0.8)

plt.title('Loss Curves Comparison')
plt.xlabel('Number of Steps per 100')
plt.ylabel('Loss')
plt.yscale(f'log')
plt.legend()
plt.tight_layout()
plt.savefig('loss_curves_comparison.png', dpi=150)
plt.show()
