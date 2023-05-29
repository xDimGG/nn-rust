import random
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from matplotlib import animation
import sklearn.model_selection

def gen_y(x: float) -> float:
	return x + random.gauss(x, x)

xs = np.random.rand(1000) * 5
ys = np.array([3 * x ** 2 + 4 * x + random.gauss(0.1, x) for x in xs])
line_xs = np.linspace(xs.min() - 0.1, xs.max() + 0.1, 200)

poly = Polynomial([0.1, -0.1, 0.1])

def cost(y: float, t: float) -> float:
	return (y - t) ** 2

def dcost_dy(y: float, t: float) -> float:
	return 2 * (y - t)

# x: input value
# y: resulting value
# t: true value
def get_gradient(x: float, poly: Polynomial, t: float) -> list[float]:
	y = poly(x)
	dc_dy = dcost_dy(y, t)

	return dc_dy * (x ** np.arange(poly.degree() + 1))

fig, ax = plt.subplots()
data = list(zip(xs, ys))

ax.scatter(xs, ys)
ax.plot(line_xs, Polynomial.fit(xs, ys, 3)(line_xs), color='green')
fit_line, = ax.plot(line_xs, poly(line_xs), color='red')

momentum = np.zeros(poly.degree() + 1)
beta = 0.9
ax.set_ylim(ys.min(), ys.max())

def animate(i):
	global poly, momentum

	alpha = 0.005

	polynomial_alpha = alpha * 10.0 ** -np.arange(poly.degree() + 1)

	SAMPLES = len(data)
	gradient = np.zeros(poly.degree() + 1)
	overall_cost = 0

	for x, t in random.sample(data, SAMPLES):
		gradient += get_gradient(x, poly, t)
		overall_cost += cost(poly(x), t)

	gradient /= SAMPLES

	momentum = momentum * beta + gradient
	poly.coef -= momentum * polynomial_alpha
	if i % 500 == 0:
		print(overall_cost)

	fit_line.set_ydata(poly(line_xs))
	return fit_line,

ani = animation.FuncAnimation(fig, animate, interval=20, save_count=200,  blit=True)
# ani.save('poly_reg.gif', writer=animation.PillowWriter(fps=30))

plt.show()
