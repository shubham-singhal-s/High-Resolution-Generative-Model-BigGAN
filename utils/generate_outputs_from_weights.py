"""
Script to get the outputs from the BigGAN generator once it's trained.

Author: Shubham Singhal
Github: shubham21197

Usage: python get_samples.py <number_of_samples> <file_name>
number_of_samples: Number of samples to generate (defaul: 1)
file_name: Name of the file that contains the model weights. Must be in the same directory as the script and must be a .h5 file (default: gen)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from self_attention import SelfAttention
import tensorflow_addons as tfa
import sys
from PIL import Image
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

# Enable memory growth to avoid OOM errors
physical_devices = tf.config.list_physical_devices('GPU')
try:
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
	print('Memory growth enabled')
except:
	pass

# Parameters
latent_dim = 128
samples = 1
file = 'gen'

if len(sys.argv) > 1:
    samples = int(sys.argv[1])
    if len(sys.argv) > 2:
        file = sys.argv[2]

def block_up(x, filters, kernel_size, padding, use_bias, initializer):
    x0 = x
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.UpSampling2D()(x)
    x = tfa.layers.SpectralNormalization(layers.Conv2D(filters, kernel_size=kernel_size, padding=padding, use_bias=use_bias, kernel_initializer=initializer))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = tfa.layers.SpectralNormalization(layers.Conv2D(filters, kernel_size=kernel_size, padding=padding, use_bias=use_bias, kernel_initializer=initializer))(x)

    x0 = layers.UpSampling2D()(x0)
    x0 = tfa.layers.SpectralNormalization(layers.Conv2D(filters, 1, padding=padding, use_bias=use_bias, kernel_initializer=initializer))(x0)

    return layers.Add()([x0, x])

def generator(latent_dim):
    dim = 32
    initializer = tf.keras.initializers.Orthogonal()

    # input layer
    inputs = layers.Input(shape=(latent_dim,))
    x = tfa.layers.SpectralNormalization(layers.Dense(4 * 4 * 16 * dim, use_bias=False))(inputs)
    x = layers.Reshape((4, 4, 16 * dim))(x)

    # Upsample: 4x4x256 -> 8x8x128
    x = block_up(x, 16 * dim, 3, 'same', False, initializer)

    # Upsample: 8x8x128 -> 16x16x64
    x = block_up(x, 8 * dim, 3, 'same', False, initializer)

    # Upsample: 16x16x64 -> 32x32x64
    x = block_up(x, 8 * dim, 3, 'same', False, initializer)

    # Upsample: 32x32x64 -> 64x64x32
    x = block_up(x, 4 * dim, 3, 'same', False, initializer)
    x = SelfAttention()(x)

    # Upsample: 64x64x32 -> 128x128x16
    x = block_up(x, 2 * dim, 3, 'same', False, initializer)

    # Upsample: 128x128x16 -> 256x256x8
    x = block_up(x, dim, 3, 'same', False, initializer)

    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # output layer
    outputs = layers.Conv2D(3, kernel_size=3, padding='same', activation='tanh', use_bias=False, kernel_initializer=initializer)(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Build the SAGAN model
gen = generator(latent_dim)
gen.load_weights('./' + file + '.h5')

def generate_and_save_images(image, epoch):
    image = image * 0.5 + 0.5
    image = (image * 255).astype('uint8')
    image = Image.fromarray(image, 'RGB')
    image.save(f'genned/output_{file}_{epoch}.png')


generated = tf.random.normal([samples, latent_dim])
predictions = gen.predict(generated)
for i, img in enumerate(predictions):
    generate_and_save_images(img, i)


