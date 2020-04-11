import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np 
import matplotlib.pyplot as plt 
import cv2
import tensorflow as tf 

warnings.filterwarnings('ignore')

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.disable_v2_behavior()

print("-----------------  Developed by Akshay Kumaar M  ----------------")
print("   ____    _    _   _    __              _____                   \n  / ___|  / \\  | \\ | |  / _| ___  _ __  |  ___|_ _  ___ ___  ___ \n | |  _  / _ \\ |  \\| | | |_ / _ \\| '__| | |_ / _` |/ __/ _ \\/ __|\n | |_| |/ ___ \\| |\\  | |  _| (_) | |    |  _| (_| | (_|  __/\\__ \\\n  \\____/_/   \\_\\_| \\_| |_|  \\___/|_|    |_|  \\__,_|\\___\\___||___/\n")
print("----------------- [ https://github.com/aksh-ai ] ----------------")

def leaky_relu(x, alpha=0.2):
	return tf.maximum(x, x*alpha)

class Dense(object):
	def __init__(self, name, X1, X2, apply_batch_norm, fun=tf.nn.relu):
		# Weight parameters
		self.W = tf.get_variable("W_%s" % name, shape=(X1, X2), initializer=tf.random_normal_initializer(stddev=0.02),)
		self.b = tf.get_variable("b_%s" % name, shape=(X2,), initializer=tf.zeros_initializer(),)

		# layer attributes
		self.fun = fun
		self.name = name
		self.apply_batch_norm = apply_batch_norm

		# params list for updating weights
		self.params = [self.W, self.b]

	def forward(self, X, reuse, is_training):
		out = tf.matmul(X, self.W) + self.b

		if self.apply_batch_norm:
			out = tf.contrib.layers.batch_norm(out, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training, reuse=reuse, scope=self.name,)
		
		return self.fun(out)

class Conv:
	def __init__(self, name, feat_in, feat_out, apply_batch_norm, filters=5, stride=2, fun=tf.nn.relu):
		# Weight parameters
		self.W = tf.get_variable("W_%s" % name, shape=(filters, filters, feat_in, feat_out), initializer=tf.truncated_normal_initializer(stddev=0.02),)
		self.b = tf.get_variable("b_%s" % name, shape=(feat_out,), initializer=tf.zeros_initializer(),)

		# layer attributes
		self.name = name
		self.fun = fun
		self.stride = stride
		self.apply_batch_norm = apply_batch_norm

		# params list for updating weights
		self.params = [self.W, self.b]

	def forward(self, X, reuse, is_training):
		conv_out = tf.nn.conv2d(X, self.W, strides=[1, self.stride, self.stride, 1], padding='SAME')
		conv_out = tf.nn.bias_add(conv_out, self.b)

		if self.apply_batch_norm:
			conv_out = tf.contrib.layers.batch_norm(conv_out, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training, reuse=reuse, scope=self.name,)
		
		return self.fun(conv_out)

class FractionalStrideConv:
	def __init__(self, name, feat_in, feat_out, output_shape, apply_batch_norm, filters=5, stride=2, fun=tf.nn.relu):
		# Weight parameters
		self.W = tf.get_variable("W_%s" % name, shape=(filters, filters, feat_out, feat_in), initializer=tf.random_normal_initializer(stddev=0.02),)
		self.b = tf.get_variable("b_%s" % name, shape=(feat_out,), initializer=tf.zeros_initializer(),)

		# layer attributes
		self.fun = fun
		self.stride = stride
		self.name = name
		self.output_shape = output_shape
		self.apply_batch_norm = apply_batch_norm

		# params list for updating weights
		self.params = [self.W, self.b]

	def forward(self, X, reuse, is_training):
		conv_out = tf.nn.conv2d_transpose(value=X, filter=self.W, output_shape=self.output_shape, strides=[1, self.stride, self.stride, 1],)
		conv_out = tf.nn.bias_add(conv_out, self.b)

		if self.apply_batch_norm:
			conv_out = tf.contrib.layers.batch_norm(conv_out, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training, reuse=reuse, scope=self.name,)

		return self.fun(conv_out)				

class GAN:
	def __init__(self, img_size, num_channels, disc_size, gen_size):
		# GAN attributes
		self.img_size = img_size
		self.num_channels = num_channels
		self.latent_dim = gen_size['z']

		# Input data
		self.X = tf.placeholder(tf.float32, shape=(None, img_size, img_size, num_channels), name='X')
		# Input noise
		self.Z = tf.placeholder(tf.float32, shape=(None, self.latent_dim), name='Z')

		# Batch size
		self.batch_size = tf.placeholder(tf.int32, shape=(), name='batch_size')

		# our discriminator
		logits = self.init_discriminator(self.X, disc_size)

		# our generator
		self.sample_images = self.init_generator(self.Z, gen_size)

		# get sample logits from discriminator
		with tf.variable_scope("discriminator") as scope:
			scope.reuse_variables()
			sample_logits = self.disc_forward(self.sample_images, True)

		# get sample images for test from generator
		with tf.variable_scope("generator") as scope:
			scope.reuse_variables()
			self.test_sample = self.gen_forward(self.Z, reuse=True, is_training=False)

		# loss functions
		# seperate losses for discriminator fake and real operations
		self.d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits))
		self.d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=sample_logits, labels=tf.zeros_like(sample_logits))

		# loss function of discriminator
		self.d_loss = tf.reduce_mean(self.d_loss_real) + tf.reduce_mean(self.d_loss_fake)
		# loss function of generator
		self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=sample_logits, labels=tf.ones_like(sample_logits)))

		real_predictions = tf.cast(logits > 0, tf.float32)
		fake_predictions = tf.cast(sample_logits < 0, tf.float32)
		
		num_predictions = 2.0*BATCH_SIZE
		
		# accuracy operation
		num_correct = tf.reduce_sum(real_predictions) + tf.reduce_sum(fake_predictions)
		self.d_accuracy = num_correct / num_predictions

		# optimizers
		# discriminator params for updating weights by the optimizer
		self.d_params = [t for t in tf.trainable_variables() if t.name.startswith('d')]
		# generator params for updating weights by the optimizer
		self.g_params = [t for t in tf.trainable_variables() if t.name.startswith('g')]

		# Adam optimizer for generator and discriminator, reduce losses respectively
		self.d_train_operation = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA1).minimize(self.d_loss, var_list=self.d_params)
		self.g_train_operation = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA1).minimize(self.g_loss, var_list=self.g_params)

		# session and variables initialization
		self.init_operation = tf.global_variables_initializer()
		self.sess = tf.InteractiveSession()
		self.sess.run(self.init_operation)

		# model saver object
		self.saver = tf.train.Saver()

	def init_discriminator(self, X, disc_size):
		with tf.variable_scope("discriminator") as scope:
			# build convolutional layers
			self.d_conv_layers = []
			feat_in = self.num_channels
			dim = self.img_size

			count = 0

			for feat_out, filters, stride, apply_batch_norm in disc_size['conv_layers']:
				name = "d_conv_layer_%s" % count
				count += 1

				layer = Conv(name, feat_in, feat_out, apply_batch_norm, filters, stride, leaky_relu)
				self.d_conv_layers.append(layer)
				feat_in = feat_out
				# print("Discriminator Dimensions:", dim)
				dim = int(np.ceil(float(dim) / stride))


			feat_in = feat_in * dim * dim

			# build dense layers
			self.d_dense_layers = []
			for feat_out, apply_batch_norm in disc_size['dense_layers']:
				name = "d_dense_layer_%s" % count
				count += 1

				layer = Dense(name, feat_in, feat_out, apply_batch_norm, leaky_relu)
				feat_in = feat_out
				self.d_dense_layers.append(layer)

			# output layer
			name = "d_final_dense_layer_%s" % count
			self.d_final_layer = Dense(name, feat_in, 1, False, lambda x: x)

			# get sample logits
			logits = self.disc_forward(X)

			# return the logits
			return logits

	def disc_forward(self, X, reuse=None, is_training=True):
		output = X

		for layer in self.d_conv_layers:
			output = layer.forward(output, reuse, is_training)

		output = tf.contrib.layers.flatten(output)

		for layer in self.d_dense_layers:
			output = layer.forward(output, reuse, is_training)

		logits = self.d_final_layer.forward(output, reuse, is_training)

		return logits

	def init_generator(self, Z, gen_size):
		with tf.variable_scope("generator") as scope:
			# size of data
			dims = [self.img_size]
			dim = self.img_size
			for _, _, stride, _ in reversed(gen_size['conv_layers']):
				dim = int(np.ceil(float(dim) / stride))
				dims.append(dim)

			# dimensions are backwards
			dims = list(reversed(dims))
			
			'''for k in dims:
				print("Generator Dimensions:", k)'''

			self.g_dims = dims

			# build dense layers
			feat_in = self.latent_dim
			self.g_dense_layers = []
			count = 0
			for feat_out, apply_batch_norm in gen_size['dense_layers']:
				name = "g_dense_layer_%s" % count
				count += 1

				layer = Dense(name, feat_in, feat_out, apply_batch_norm)
				self.g_dense_layers.append(layer)
				feat_in = feat_out

			# output dense layer
			feat_out = gen_size['projection'] * dims[0] * dims[0]
			name = "g_dense_layer_%s" % count
			layer = Dense(name, feat_in, feat_out, not gen_size['bn_after_project'])
			self.g_dense_layers.append(layer)

			# fractionally strided convolutional layer
			feat_in = gen_size['projection']
			self.g_conv_layers = []

			# output activation either tanh or sigmoid
			num_relus = len(gen_size['conv_layers']) - 1
			activation_functions = [tf.nn.relu]*num_relus + [gen_size['output_activation']]

			# build "deconvolutional" layer
			for i in range(len(gen_size['conv_layers'])):
				name = "g_fs_conv_layer_%s" % i
				feat_out, filters, stride, apply_batch_norm = gen_size['conv_layers'][i]
				fun = activation_functions[i]
				output_shape = [self.batch_size, dims[i+1], dims[i+1], feat_out]
				# print("Input Features:", feat_in, "Output Features:", feat_out, "Output Shape:", output_shape)
				layer = FractionalStrideConv(name, feat_in, feat_out, output_shape, apply_batch_norm, filters, stride, fun)
				self.g_conv_layers.append(layer)
				feat_in = feat_out

			# output
			self.gen_size = gen_size

			return self.gen_forward(Z)

	def gen_forward(self, Z, reuse=None, is_training=True):
		# output from dense
		output = Z
		for layer in self.g_dense_layers:
			output = layer.forward(output, reuse, is_training)

		# project and reshape
		output = tf.reshape(output, [-1, self.g_dims[0], self.g_dims[0], self.gen_size['projection']],)

		# apply batch normalization
		if self.gen_size['bn_after_project']:
			output = tf.contrib.layers.batch_norm(output, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training, reuse=reuse, scope='bn_after_project')

		# output via fractionally strided convolutional layers
		for layer in self.g_conv_layers:
			output = layer.forward(output, reuse, is_training)

		return output

	def fit(self, X):
		d_losses = []
		g_losses = []
		d_accs = []
		
		offset = 0
		
		N = len(X)
		num_batches = N // BATCH_SIZE
		print("Total batches per epoch is {}\n".format(num_batches))
		total_iters = 0

		for i in range(EPOCHS):
			print("Epoch", i+1)
			np.random.shuffle(X)
			for offset in range(num_batches):
				batch = preprocess(X[offset*BATCH_SIZE:(offset+1)*BATCH_SIZE])

				Z = np.random.uniform(-1, 1, size=(BATCH_SIZE, self.latent_dim))

				# train the discriminator
				_, d_loss, d_acc = self.sess.run((self.d_train_operation, self.d_loss, self.d_accuracy), feed_dict={self.X: batch, self.Z: Z, self.batch_size: BATCH_SIZE},)
				d_losses.append(d_loss)

				# train the generator
				_, g_loss1 = self.sess.run((self.g_train_operation, self.g_loss), feed_dict={self.Z: Z, self.batch_size: BATCH_SIZE},)

				# do it again
				_, g_loss2 = self.sess.run((self.g_train_operation, self.g_loss), feed_dict={self.Z: Z, self.batch_size: BATCH_SIZE},)

				# store the loss
				g_losses.append((g_loss1 + g_loss2)/2) 
				
				# store the accuracy
				d_accs.append(d_acc)

				# print("Discriminator Accuracy: %.2f  |  Discriminator Loss: %.2f  |  Generator Loss: %.2f" % (d_acc, d_loss, g_losses[offset]))

				# save samples periodically
				total_iters += 1

				self.saver.save(self.sess, "models\\GAN_face")

				if total_iters % SAVE_PERIOD == 0:
					print("Saving sample {}".format(total_iters))
					
					if not os.path.exists('generated_samples'):
						os.mkdir('generated_samples')
						
					samples = self.sample(64) 

					d = self.img_size
					
					if samples.shape[-1] == 1:
						samples = samples.reshape(64, d, d)
						flat_image = np.empty((8*d, 8*d))

						k = 0
						for i in range(8):
							for j in range(8):
								flat_image[i*d:(i+1)*d, j*d:(j+1)*d] = samples[k].reshape(d, d)
								k += 1

					else:
						flat_image = np.empty((8*d, 8*d, 3))
						k = 0
						for i in range(8):
							for j in range(8):
								flat_image[i*d:(i+1)*d, j*d:(j+1)*d] = samples[k]
								k += 1

					sp.imsave('generated_samples\\sample%d.png' % total_iters, flat_image,)
					
			print("Discriminator Accuracy: %.2f  |  Discriminator Loss: %.2f  |  Generator Loss: %.2f" % (d_accs[offset], d_losses[offset], g_losses[offset]))

		# plot the losses and save them
		plt.clf()
		plt.plot(g_losses, label='Generator Loss')
		plt.plot(d_losses, label='Discriminator Loss')
		plt.title('GAN Loss')
		plt.legend()
		plt.savefig('loss_metrics.png')

	def sample(self, n):
		# generate a sample from noise
		Z = np.random.uniform(-1, 1, size=(n, self.latent_dim))
		samples = self.sess.run(self.test_sample, feed_dict={self.Z: Z, self.batch_size: n})
		return samples

	def save_weights(self, path):
		# save model weights
		self.saver.save(self.sess, path)
		print("Saved successfully")

dimensions = 64
channels = 3
d = dimensions
LEARNING_RATE = 0.0002
BETA1 = 0.5

disc_sizes = {
	'conv_layers': [
		(64, 5, 2, False),
		(128, 5, 2, True),
		(256, 5, 2, True),
		(512, 5, 2, True)
	],
	'dense_layers': [],
	}
	
gen_sizes = {
	'z': 100,
	'projection': 512,
	'bn_after_project': True,
	'conv_layers': [
		(256, 5, 2, True),
		(128, 5, 2, True),
		(64, 5, 2, True),
		(channels, 5, 2, False)
	],
	'dense_layers': [],
	'output_activation': tf.tanh,
	}

choice = input("\n1) Generate samples of faces\n2) Generate single face\n3) Exit\n\nEnter your choice: ")

if int(choice)==1: 
	print("\nGenerating samples...\n")

	BATCH_SIZE = 64

	model = GAN(dimensions, channels, disc_sizes, gen_sizes)

	with tf.compat.v1.Session() as sess:
		model.saver.restore(model.sess, os.path.join("models", "GAN_face")) 

	samples = model.sample(64)	

	if samples.shape[-1] == 1:
		samples = samples.reshape(64, d, d)
		flat_image = np.empty((8*d, 8*d))

		k = 0
		for i in range(8):
			for j in range(8):
				flat_image[i*d:(i+1)*d, j*d:(j+1)*d] = samples[k].reshape(d, d)
				k += 1

	else:
		flat_image = np.empty((8*d, 8*d, 3))
		k = 0
		for i in range(8):
			for j in range(8):
				flat_image[i*d:(i+1)*d, j*d:(j+1)*d] = samples[k]
				k += 1

	img = flat_image 

elif int(choice)==2:
	print("\nGenerating single face...\n")

	BATCH_SIZE = 1

	model = GAN(dimensions, channels, disc_sizes, gen_sizes)

	with tf.compat.v1.Session() as sess:
		model.saver.restore(model.sess, os.path.join("models", "GAN_face"))

	samples = model.sample(1) 

	img = samples.reshape(dimensions, dimensions, 3)

else:
	exit()

plt.imshow(img)
plt.show()

exit()