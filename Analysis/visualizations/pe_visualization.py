import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


length = 128
dim = 256

matrix = np.zeros([length, dim])
for i in range(length):
    for j in range(dim):
        if j % 2 == 0:
            matrix[i, j] = np.sin(i/np.power(10000,2*j/dim))
        else:
            matrix[i, j] = np.cos(i/np.power(10000,2*j/dim))

def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P

P = getPositionEncoding(seq_len=20, d=64, n=10000)

plt.figure()
ax = plt.gca()
im = ax.imshow(P)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size="2%", pad=0.05)
# cax = plt.matshow(P)
# plt.gcf().colorbar(cax, fraction=0.046, pad=0.04)
plt.colorbar(im, cax)
ax.set_ylabel('Position index in the sequence', fontsize=20)
ax.set_xlabel('Feature depth index', fontsize=20)
plt.show()