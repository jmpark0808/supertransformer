import numpy as np
import matplotlib.pyplot as plt
import copy
# N = 300
# t = np.linspace(0, 2*np.pi, N)
# x = 5*np.cos(t) #+ 0.2*np.random.normal(size=N) + 1
# y = 4*np.sin(t) #+ 0.2*np.random.normal(size=N)

# plt.plot(x, y, '.')     # given points

# xmean, ymean = x.mean(), y.mean()
# x -= xmean
# y -= ymean
# U, S, V = np.linalg.svd(np.stack((x, y)))



# tt = np.linspace(0, 2*np.pi, 1000)
# circle = np.stack((np.cos(tt), np.sin(tt)))    # unit circle
# transform = np.sqrt(2/N) * U.dot(np.diag(S))   # transformation matrix

# fit = transform.dot(circle) + np.array([[xmean], [ymean]])
# plt.plot(fit[0, :], fit[1, :], 'r')
# plt.show()


N = 300
t = np.linspace(0, 2*np.pi, N)
x = 5*np.cos(t) + 0.2*np.random.normal(size=N) + 1
y = 4*np.sin(t+0.5) + 0.2*np.random.normal(size=N)
plt.plot(x, y, '.')     # given points

xmean, ymean = x.mean(), y.mean()
x -= xmean
y -= ymean
U, S, V = np.linalg.svd(np.stack((x, y)))

angle = np.arctan2(U[1], U[0])
angle_positive = np.arctan2(np.abs(U[1]), U[0])
print(np.sqrt(2/N) *S[0], np.sqrt(2/N) *S[1], angle[0], angle[1], angle_positive[0], angle_positive[1])

tt = np.linspace(0, 2*np.pi, 1000)
circle = np.stack((np.cos(tt), np.sin(tt)))    # unit circle
transform = np.sqrt(2/N) * U.dot(np.diag(S))   # transformation matrix
fit = transform.dot(circle) + np.array([[xmean], [ymean]])
plt.plot(fit[0, :], fit[1, :], 'r')
plt.show()

