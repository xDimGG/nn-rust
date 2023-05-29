import tensorflow as tf
from tensorflow import keras

# define the input layer
inputs = keras.layers.Input(shape=(1,))

# define the dense layer with 64 units
dense_layer = keras.layers.Dense(units=64, activation='relu')(inputs)

# define the output layers with 2 units and softmax activation for categorical classification
output1 = keras.layers.Dense(units=2, activation='softmax')(dense_layer)

# define the final model with the input and output layers
model = keras.models.Model(inputs=inputs, outputs=output1)

model.compile(loss='categorical_crossentropy', optimizer='adam')

# define a function to generate test data
def generate_test_data(num_samples):
	for _ in range(num_samples):
		# generate a random input number x
		x = tf.random.uniform(shape=(1,))

		# generate the corresponding labels y
		if x < 0.5:
			y = tf.one_hot(0, depth=2)  # category 1 is true
		else:
			y = tf.one_hot(1, depth=2)  # category 2 is true

		yield x

# create a dataset from the test data generator function
test_data = tf.data.Dataset.from_generator(
	generate_test_data, args=[1000],
	output_signature=(tf.TensorSpec(shape=(1,), dtype=tf.float32), tf.TensorSpec(shape=(2,), dtype=tf.int32))
)

model.fit(test_data, epochs=500, steps_per_epoch=10, batch_size=128)

# print model summary
# model.summary()
print(model.predict([[0.1]]))
