import numpy as np

ps = 4

xs = np.arange(ps//2, ps//2+ps*56, ps)
ys = np.arange(ps//2, ps//2+ps*56, ps)
xv, yv = np.meshgrid(xs, ys, indexing='ij')
grid = np.stack((xv, yv), axis=2)
print(grid[:5, 0, :])