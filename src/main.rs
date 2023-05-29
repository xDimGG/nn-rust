use std::iter;
use rand::{prelude::random, thread_rng, seq::SliceRandom};
use mnist::*;

// Activation function
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Activation {
	Sigmoid,
	ReLU,
	Linear,
	Softmax,
	None,
}

impl Activation {
	fn eval(&self, layer: &Vec<f64>, index: usize) -> f64 {
		let x = layer[index];

		match self {
			Self::Sigmoid => {
				if x < -40.0 {
					0.0
				} else if x > 40.0 {
					1.0
				} else {
					1.0 / (1.0 + (-x).exp())
				}
			},
			Self::ReLU => {
				if x > 0.0 {
					x
				} else {
					0.0
				}
			},
			Self::Linear => x,
			Self::Softmax => x.exp() / layer.iter().map(|j| j.exp()).sum::<f64>(),
			Self::None => 0.0,
		}
	}

	fn eval_derivative(&self, layer: &Vec<f64>, index: usize) -> f64 {
		let x = layer[index];

		match self {
			Self::Sigmoid => {
				let s = self.eval(layer, index);
				s * (1.0 - s)
			},
			Self::ReLU => {
				if x > 0.0 {
					1.0
				} else {
					0.0
				}
			},
			Self::Linear => 1.0,
			Self::Softmax => {
				let s = self.eval(layer, index);
				s * (1.0 - s)
			},
			Self::None => 0.0,
		}
	}
}

// Cost function
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cost {
	MSE,
}

impl Cost {
	fn eval(&self, y: f64, t: f64) -> f64 {
		match self {
			Self::MSE => (y - t) * (y - t),
		}
	}

	fn eval_derivative(&self, y: f64, t: f64) -> f64 {
		match self {
			Self::MSE => 2.0 * (y - t),
		}
	}
}

#[derive(Debug)]
struct Neuron {
	activation: Activation,
	weights: Vec<f64>, // outgoing weights
	bias: f64, // own bias
}

impl Neuron {
	fn new(activation: Activation, weights: Vec<f64>, bias: f64) -> Self {
		Self {
			activation,
			weights,
			bias,
		}
	}
}

#[derive(Debug)]
struct Layer {
	neurons: Vec<Neuron>,
}

impl Layer {
	fn new(n: usize, activation: Activation) -> Self {
		Layer {
			neurons: (0..n)
				.map(|_| Neuron::new(
					activation,
					Vec::new(),
					if activation == Activation::None { 0.0 } else { random() }
				))
				.collect(),
		}
	}
}

#[derive(Debug)]
struct Network {
	layers: Vec<Layer>,
	input_size: usize,
	output_size: usize,
	cost: Cost,
}

#[derive(Debug)]
enum NetworkError {
	MissingIOLayers,
	BadInput,
}

impl Network {
	fn new(cost: Cost) -> Self {
		Self {
			layers: Vec::new(),
			input_size: 0,
			output_size: 0,
			cost,
		}
	}

	fn add(&mut self, l: Layer) {
		if self.layers.len() == 0 {
			self.input_size = l.neurons.len()
		} else {
			for n in &mut self.layers.last_mut().unwrap().neurons {
				n.weights = (0..l.neurons.len()).map(|_| random::<f64>() - 0.5).collect()
			}
		}

		self.output_size = l.neurons.len();
		self.layers.push(l)
	}

	fn predict(&self, x: &Vec<f64>) -> Result<Vec<f64>, NetworkError> {
		if x.len() != self.input_size {
			return Err(NetworkError::BadInput);
		}
		if self.layers.len() < 2 {
			return Err(NetworkError::MissingIOLayers);
		}

		let mut input = x.clone();
		for window in self.layers.windows(2) {
			if let [prev, next] = window {
				let mut outputs: Vec<f64> = next.neurons.iter().map(|n| n.bias).collect();

				for (j, next_neuron) in next.neurons.iter().enumerate() {
					for (i, prev_neuron) in prev.neurons.iter().enumerate() {
						outputs[j] += prev_neuron.weights[j] * input[i];
					}

					outputs[j] = next_neuron.activation.eval(&outputs, j);
				}

				input = outputs;
			}
		}

		Ok(input)
	}

	fn learn(&mut self, data: &Vec<(Vec<f64>, Vec<f64>)>, epochs: usize, alpha: f64, beta: f64, batch_size: usize, log: bool) -> Result<(), NetworkError> {
		if self.layers.len() < 2 {
			return Err(NetworkError::MissingIOLayers);
		}

		let rng = &mut thread_rng();
		let alpha = alpha / batch_size as f64;

		let mut momentum: Option<(Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>)> = Option::None;

		for epoch in 0..epochs {
			let mut gradient: Option<(Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>)> = Option::None;
			let mut total_cost = 0.0;

			for (input, output) in data.choose_multiple(rng, batch_size) {
				if input.len() != self.input_size {
					return Err(NetworkError::BadInput);
				}

				if output.len() != self.output_size {
					return Err(NetworkError::BadInput);
				}

				let (grad_w, grad_b, c) = self.gradients_cost(input, output);
				total_cost += c;

				if let Some((grad_sum_w, grad_sum_b)) = &mut gradient {
					for i in 0..grad_w.len() {
						for j in 0..grad_w[i].len() {
							for k in 0..grad_w[i][j].len() {
								grad_sum_w[i][j][k] += grad_w[i][j][k];
							}
						}
					}

					for i in 0..grad_b.len() {
						for j in 0..grad_b[i].len() {
							grad_sum_b[i][j] += grad_b[i][j];
						}
					}
				} else {
					gradient = Some((grad_w, grad_b))
				}
			}

			if let Some((momentum_w, momentum_b)) = &mut momentum {
				let (grad_w, grad_b) = gradient.unwrap();

				for i in 0..momentum_w.len() {
					for j in 0..momentum_w[i].len() {
						for k in 0..momentum_w[i][j].len() {
							momentum_w[i][j][k] = beta * momentum_w[i][j][k] + grad_w[i][j][k];
						}
					}
				}

				for i in 0..momentum_b.len() {
					for j in 0..momentum_b[i].len() {
						momentum_b[i][j] = beta * momentum_b[i][j] + grad_b[i][j];
					}
				}
			} else {
				momentum = gradient;
			}

			// Apply negative gradients
			if let Some((grad_w, grad_b)) = &momentum {
				for layer in 0..self.layers.len() {
					for i in 0..self.layers[layer].neurons.len() {
						self.layers[layer].neurons[i].bias -= alpha * grad_b[layer][i];
						for j in 0..self.layers[layer].neurons[i].weights.len() {
							self.layers[layer].neurons[i].weights[j] -= alpha * grad_w[layer][i][j];
						}
					}
				}
			}

			if log {
				total_cost /= batch_size as f64;
				println!("Epoch {}/{epochs}: cost={total_cost}", epoch + 1)
			}
		}

		Ok(())
	}

	fn gradients_cost(&mut self, x: &Vec<f64>, t: &Vec<f64>) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>, f64) {
		let mut mat_a = Vec::with_capacity(self.layers.len()); // A matrix
		let mut mat_z = Vec::with_capacity(self.layers.len()); // Z matrix
		mat_a.push(x.clone()); // Input layer's a results are equal to the input
		mat_z.push(Vec::new()); // Input layer has no z's

		// Forward pass
		for window in self.layers.windows(2) {
			if let [prev, next] = window {
				let last_layer = mat_a.last().unwrap();
				let mut a_outputs = vec![0.0; next.neurons.len()];
				let mut z_outputs: Vec<f64> = next.neurons.iter().map(|n| n.bias).collect();

				for (j, next_neuron) in next.neurons.iter().enumerate() {
					for (i, prev_neuron) in prev.neurons.iter().enumerate() {
						z_outputs[j] += prev_neuron.weights[j] * last_layer[i];
					}
				}

				for (j, next_neuron) in next.neurons.iter().enumerate() {
					a_outputs[j] = next_neuron.activation.eval(&z_outputs, j);
				}

				mat_a.push(a_outputs);
				mat_z.push(z_outputs);
			}
		}

		let mut grad_w: Vec<Vec<Vec<f64>>> = self.layers[0..self.layers.len()-1]
			.iter()
			.map(|l| l.neurons
				.iter()
				.map(|n| vec![0.0; n.weights.len()])
				.collect())
			.collect();

		// equivalent to grad_b
		let mut grad_da_dz: Vec<Vec<f64>> = self.layers
			.iter()
			.map(|l| vec![0.0; l.neurons.len()])
			.collect();

		let cost_gradient = mat_a
			.last()
			.unwrap()
			.iter()
			.zip(t)
			.map(|(&y, &t)| self.cost.eval_derivative(y, t))
			.collect::<Vec<f64>>();

		let mut grad_dz_da: Vec<Vec<f64>> = self.layers[0..self.layers.len()-1]
			.iter()
			.map(|l| vec![0.0; l.neurons.len()])
			.chain(iter::once(cost_gradient))
			.collect();

		// Backwards pass
		for layer in (1..self.layers.len()).rev() {
			if layer != self.layers.len()-1 {
				for i in 0..self.layers[layer].neurons.len() {
					for j in 0..self.layers[layer + 1].neurons.len() {
						grad_dz_da[layer][i] += grad_da_dz[layer + 1][j] * self.layers[layer].neurons[i].weights[j];
					}
				}
			}

			for (i, neuron) in self.layers[layer].neurons.iter().enumerate() {
				grad_da_dz[layer][i] = grad_dz_da[layer][i] * neuron.activation.eval_derivative(&mat_z[layer], i);
			}

			for (i, _prev_neuron) in self.layers[layer - 1].neurons.iter().enumerate() {
				for (j, _neuron) in self.layers[layer].neurons.iter().enumerate() {
					grad_w[layer - 1][i][j] = grad_da_dz[layer][j] * mat_a[layer - 1][i];
				}
			}
		}

		let cost = mat_a
			.last()
			.unwrap()
			.iter()
			.zip(t)
			.map(|(&y, &t)| self.cost.eval(y, t))
			.sum::<f64>() / t.len() as f64;

		(grad_w, grad_da_dz, cost)
	}
}

// https://github.com/busyboredom/rust-mnist/blob/main/examples/perceptron.rs#L113
fn one_hot(value: u8) -> Vec<f64> {
	let mut arr: [f64; 10] = [0.0; 10];
	arr[usize::from(value)] = 1.0;
	arr.to_vec()
}

// https://github.com/busyboredom/rust-mnist/blob/main/examples/perceptron.rs#L127
fn largest(arr: Vec<f64>) -> usize {
	arr.iter()
		.enumerate()
		.max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
		.map(|(index, _)| index)
		.unwrap()
}

fn main() {
	// let mut nw = Network::new(Cost::MSE);
	// nw.add(Layer::new(28*28, Activation::None));
	// nw.add(Layer::new(8, Activation::Sigmoid));
	// nw.add(Layer::new(10, Activation::Softmax));

	// let Mnist {
	// 	trn_img,
	// 	trn_lbl,
	// 	tst_img,
	// 	tst_lbl,
	// 	..
	// } = MnistBuilder::new()
	// 	.label_format_digit()
	// 	.training_set_length(50_000)
	// 	.validation_set_length(10_000)
	// 	.test_set_length(10_000)
	// 	.finalize();

	// let trn_img = trn_img
	// 	.iter()
	// 	.map(|&x| 2.0 * (x as f64) / 256.0 - 1.0)
	// 	.collect::<Vec<f64>>()
	// 	.chunks(28*28)
	// 	.map(|s| s.to_vec())
	// 	.collect::<Vec<Vec<f64>>>();
	
	// let trn_lbl = trn_lbl
	// 	.iter()
	// 	.map(|&x| one_hot(x))
	// 	.collect::<Vec<Vec<f64>>>();

	// let tst_img = tst_img
	// 	.iter()
	// 	.map(|&x| 2.0 * (x as f64) / 256.0 - 1.0)
	// 	.collect::<Vec<f64>>()
	// 	.chunks(28*28)
	// 	.map(|s| s.to_vec())
	// 	.collect::<Vec<Vec<f64>>>();
	
	// let data = trn_img
	// 	.into_iter()
	// 	.zip(trn_lbl)
	// 	.map(|(img, lbl)| (img, lbl))
	// 	.collect();

	// nw.learn(&data, 100, 0.001, 0.8, 100, true).unwrap();

	// let mut correct = 0;

	// for (img, &lbl) in tst_img.iter().zip(&tst_lbl) {
	// 	let pred = nw.predict(img).unwrap();

	// 	if lbl as usize == largest(pred) {
	// 		correct += 1;
	// 	}
	// }

	// println!("Test data: {correct}/{}", tst_img.len())

	// let ltgt_data = (0..1000).map(|_| {
	// 	let x: f64 = random();
	// 	// let y = 10.0 * x + 1.0;
	// 	let y1 = if x < 0.5 { 1.0 } else { 0.0 };
	// 	let y2 = if x < 0.5 { 0.0 } else { 1.0 };

	// 	(vec![x].repeat(28*28), vec![y1, y2].repeat(5))
	// }).collect();

	// dbg!(&nw);

	// nw.learn(&ltgt_data, 1000, 0.01, 0.8, 100, true).unwrap();

	// println!("{}, {}", nw.predict(&vec![1.0]).unwrap()[0]);
	// println!("{}", nw.predict(&vec![1.5]).unwrap()[0]);
	// for _ in 0..20 {
	// 	let x: f64 = random();
	// 	let pre = nw.predict(&vec![x]).unwrap();

	// 	println!("{x} is probably {} than 0.5", if pre[0] < pre[1] { "greater" } else { "less" })
	// }

	// dbg!(nw);

	// nw.learn(&vec![
	// 	(vec![1.3], vec![0.0]),
	// 	(vec![0.4], vec![0.4]),
	// ], 500, 0.01, 2).unwrap();
	// let mut correct = 0;
	// for _ in 0..100 {
	// 	let x = random::<f64>();
	// 	let ys = nw.predict(&vec![x]).unwrap();
	// 	if (ys[1] > ys[0] && x > 0.5) || (ys[1] < ys[0] && x < 0.5) {
	// 		correct += 1;
	// 	} else {
	// 		println!("{x}, {}, {}", ys[0], ys[1])
	// 	}
	// }

	// println!("{correct}")

	// Example on paper
	let mut nw = Network::new(Cost::MSE);
	nw.add(Layer::new(2, Activation::ReLU));
	nw.add(Layer::new(2, Activation::ReLU));
	nw.add(Layer::new(1, Activation::ReLU));

	nw.layers[0].neurons[0].weights[0] = 0.5;
	nw.layers[0].neurons[0].weights[1] = 0.25;
	nw.layers[0].neurons[1].weights[0] = 0.4;
	nw.layers[0].neurons[1].weights[1] = 0.2;

	nw.layers[1].neurons[0].bias = 0.6;
	nw.layers[1].neurons[0].weights[0] = 0.5;
	nw.layers[1].neurons[1].bias = 0.2;
	nw.layers[1].neurons[1].weights[0] = 0.4;

	nw.layers[2].neurons[0].bias = 0.1;

	nw.gradients_cost(&vec![2.0, 4.0], &vec![3.0]);
	dbg!(nw.predict(&vec![2.0, 4.0]).unwrap());
	// End example on paper
}
