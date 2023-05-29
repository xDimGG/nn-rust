import random
import numpy as np

def cost(y: float, t: float) -> float:
	return (y - t) ** 2

def cost_d(y: float, t: float) -> float:
	return 2 * (y - t)

class Network():
	def __init__(self) -> None:
		self.weights = []
		self.biases = []
		self.activations = []

	# activation is a 2-tuple of functions representing the fn and the derivative of the fn wrt x
	def add_layer(self, n: int, activation=None) -> None:
		if len(self.biases) > 0:
			self.weights += [np.random.rand(n, len(self.biases[-1]))]

		self.biases += [np.random.rand(n, 1)]
		self.activations += [activation]

	def forward_pass(self, inputs: list[float]) -> tuple[list[np.ndarray], list[np.ndarray]]:
		assert len(inputs) == len(self.biases[0])

		# value after neuron activation function
		a_values = [np.array(inputs)[:, None]]

		# value before activation function
		z_values = [None]

		for weights, biases, (activation_fn, _) in zip(self.weights, self.biases[1:], self.activations[1:]):
			z_values += [(weights @ a_values[-1]) + biases]
			a_values += [activation_fn(z_values[-1])]

		return a_values, z_values

	def predict(self, inputs: list[float]) -> list[float]:
		a, _ = self.forward_pass(inputs)
		return a[-1][:, 0]

	# generate c from true values and output values
	def compute_cost_gradient(self, outputs: list[float], true: list[float], cost_d=cost_d) -> list[float]:
		return [cost_d(y, t) for y, t in zip(outputs, true)]

	# gradient from a, z, and c
	# c: cost gradient (dC/da)
	def compute_gradients(self, a_vals: list[np.ndarray], z_vals: list[np.ndarray], c: list[float]) -> tuple[list[np.ndarray], list[np.ndarray]]:
		w_gradients = []
		b_gradients = []
		da_dz = [np.array(c) * self.activations[-1][1](z_vals[-1])]

		# backward pass
		for weights, activation, a, z in zip(
				self.weights[::-1], self.activations[::-1][1::], a_vals[::-1], z_vals[::-1]):
			# print(weights, biases)
			# da_dz = dz_da[-1] * activation_fn(z)
			# print(dz_da, da_dz)
			# dz/db
			b_gradients += [da_dz[-1]]
			w_gradients += [np.repeat((a * da_dz[-1]), weights.shape[1], axis=1)]
			if activation is not None:
				da_dz += [(weights.T @ da_dz[-1]) * activation[1](z)]

		return w_gradients[::-1], b_gradients[::-1]

	def gradient_from_data(self, inputs: list[float], true: list[float], cost_d=cost_d) -> tuple[list[np.ndarray], list[np.ndarray]]:
		a, z = self.forward_pass(inputs)
		c = self.compute_cost_gradient(list(a[-1]), true)

		return self.compute_gradients(a, z, c)

	def apply_gradient(self, w_gradients: list[np.ndarray], b_gradients: list[np.ndarray], alpha: float) -> None:
		for i, w in enumerate(w_gradients):
			self.weights[i] -= w * alpha

		for i, b in enumerate(b_gradients):
			self.weights[i] -= b * alpha

	def learn(
			self,
			x: list[list[float]],
			y: list[list[float]],
			epochs: int,
			batch_size: int,
			alpha: float,
			beta: float,
			cost=(cost, cost_d)) -> None:

		# momentum gradients
		w_grad = [np.zeros(w.shape) for w in self.weights]
		b_grad = [np.zeros(b.shape) for b in self.biases[1:]]

		for epoch in range(epochs):
			# apply momentum beta
			for i, w in enumerate(w_grad):
				w_grad[i] = w * beta
			for i, b in enumerate(b_grad):
				b_grad[i] = b * beta

			cost_val = 0

			for _ in range(batch_size):
				i = random.randrange(0, len(x))
				w_n, b_n = self.gradient_from_data(x[i], y[i], cost[1])
				cost_val += sum(map(lambda z: cost[0](*z), zip(y[i], self.predict(x[i]))))

				for i, w in enumerate(w_n):
					w_grad[i] += w / batch_size
				for i, b in enumerate(b_n):
					b_grad[i] += b / batch_size

			self.apply_gradient(w_grad, b_grad, alpha)

			print(f'Cost after epoch {epoch + 1}/{epochs}: {cost_val / batch_size}')

relu = (lambda x: np.maximum(x, 0), lambda x: 1.0 * (x > 0))
linear = (lambda x: x, lambda _: 1)

nw = Network()
nw.add_layer(1)
nw.add_layer(100, linear)
nw.add_layer(1, linear)

xs = np.random.rand(100) * 10
ys = xs * 5 + 4

nw.learn([[x] for x in xs.tolist()], [[y] for y in ys.tolist()], 1000, 10, 0.000001, 0.6)

for x in xs.tolist():
	print(x, nw.predict([x]), x * 5 + 4)

# nw.biases[1][0, 0] = 0.6
# nw.biases[1][1, 0] = 0.2

# nw.biases[2][0, 0] = 0.1

# nw.weights[0][0, 0] = 0.5
# nw.weights[0][0, 1] = 0.4
# nw.weights[0][1, 0] = 0.25
# nw.weights[0][1, 1] = 0.2
# nw.weights[1][0, 0] = 0.5
# nw.weights[1][0, 1] = 0.4

# print(nw.biases)
# print(nw.weights)

# print(nw.forward_pass([2, 4]))
# grad_w, grad_b = nw.gradient_from_data([2, 4], [3])
# print(f'{grad_w=}')
# print(f'{grad_b=}')

# nw.biases[1][:, 0] = 0.5
# nw.biases[2][:, 0] = 0.5

# print(nw.weights)

# nw.weights[0][0, 0] = 0.2
# nw.weights[0][1, 0] = 0.3
# nw.weights[1][0, 0] = 0.3
# nw.weights[1][0, 1] = 0.2

# print(nw.weights, nw.biases)
# w, b = nw.learn([1], [12])
