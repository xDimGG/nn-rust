import math
import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib import cm

def gen_y(x: float) -> float:
	return x + random.gauss(0, x / 2)

xs = np.random.rand(1000) * 2
ys = np.array([gen_y(x) for x in xs])
line_x = np.array([xs.min() - 0.1, xs.max() + 0.1])

data = list(zip(xs, ys))

best_m, best_b = np.linalg.lstsq(np.vstack([xs, np.ones(len(xs))]).T, ys, rcond=None)[0]
m = -2
b = -2

print(best_m, best_b)

def cost(y: float, t: float) -> float:
	return (y - t) ** 2

def dcost_dy(y: float, t: float) -> float:
	return 2 * (y - t)

# x: input value
# y: resulting value
# t: true value
def get_gradients(x: float, y: float, t: float) -> tuple[float, float]:
	dc_dy = dcost_dy(y, t)
	dy_dm = x
	dy_db = 1

	return dc_dy * dy_dm, dc_dy * dy_db

fig = plt.figure(figsize=plt.figaspect(0.5))

ax = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

# optimization surface
possible_m = np.arange(-2, 4, 0.01)
possible_b = np.arange(-2, 4, 0.01)

X, Y = np.meshgrid(possible_m, possible_b)
Z = np.zeros((possible_m.shape[0], possible_b.shape[0]))

for x, t in data:
	Z += cost(X * x + Y, t)

Z /= len(data)

ax2.set_xlabel('m')
ax2.set_ylabel('b')
ax2.set_zlabel('cost')
ax2.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, zorder=1)

gradient_ball, = ax2.plot(m, b, 0, color='lightgreen', linestyle='', marker='o', zorder=2)

# linreg
ax.scatter(xs, ys)
ax.plot(line_x, best_m * line_x + best_b, color='lightgreen')
ax.set_ylim(ys.min() - 0.5, ys.max() + 0.5)
fit_line, = ax.plot(line_x, m * line_x + b, color='red')

good_m = False
good_b = False
dc_dm_momentum = 0
dc_db_momentum = 0
beta = 0.8

def animate(i):
	global m, b, dc_dm_momentum, dc_db_momentum, good_m, good_b, gradient_ball

	dc_dm_sum = dc_dm_momentum
	dc_db_sum = dc_db_momentum
	alpha = 0.005

	SAMPLES = len(data)
	total_cost = 0

	for x, t in random.sample(data, SAMPLES):
		y = m * x + b
		total_cost += cost(y, t)
		delta_m, delta_b = get_gradients(x, y, t)
		dc_dm_sum += delta_m
		dc_db_sum += delta_b

	dc_dm_sum /= SAMPLES
	dc_db_sum /= SAMPLES

	dc_dm_momentum = beta * dc_dm_momentum + dc_dm_sum
	dc_db_momentum = beta * dc_db_momentum + dc_db_sum

	m -= dc_dm_momentum * alpha
	b -= dc_db_momentum * alpha

	if abs(m - best_m) < 0.01 and not good_m:
		print(f'good m after {i} epochs')
		good_m = True

	if abs(b - best_b) < 0.01 and not good_b:
		print(f'good b after {i} epochs')
		good_b = True

	fit_line.set_ydata(m * line_x + b)
	gradient_ball.set_data([m], [b])
	gradient_ball.set_3d_properties(total_cost / SAMPLES)
	return fit_line, gradient_ball

ani = animation.FuncAnimation(fig, animate, interval=40, save_count=100, blit=True)
# ani.save('lin_reg.gif', writer=animation.PillowWriter(fps=30))

plt.show()
