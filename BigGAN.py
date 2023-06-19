"""
This is an implementation of BigGAN, a GAN architecture that uses self-attention and spectral normalization to generate high-resolution images.
The model is designed to train on 256x256 colour images, for a single class (Mountains).

Author: Shubham Singhal
Github: shubham21197

Usage: python BigGAN.py <folder_suffix> <offset>
folder_suffix: A string that will be appended to the folder names for saving models and generated  (default: '')
offset: The number of epochs to offset the training by. Useful for continuing a previous training. If not null, 
        the weights saved in folder_suffix will be used. (default: null)

"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import glob
from time import time
import matplotlib.pyplot as plt
from self_attention import SelfAttention
import tensorflow_addons as tfa
import sys
import os

# Enable memory growth for GPU; otherwise, TF will allocate all memory at once
physical_devices = tf.config.list_physical_devices('GPU')
try:
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
	print('Memory growth enabled')
except:
	pass

# Model properties and system args
BATCH_SIZE = 32
print(f'Starting script with batch size {BATCH_SIZE}')
base_path = "./data/"
latent_dim = 128
epochs = 20000

# Settings to continue training from a previous model
offset = 0
n_folder = ''
continueTrain = offset > 0

if len(sys.argv) > 1:
    n_folder = sys.argv[1]
    if len(sys.argv) > 2:
        offset = int(sys.argv[2])
        print('Offset found, continuing training')
        continueTrain = offset > 0
if not os.path.exists(base_path + 'models' + n_folder):
    os.makedirs(base_path + 'models' + n_folder)
if not os.path.exists(base_path + 'genned' + n_folder):
    os.makedirs(base_path + 'genned' + n_folder)

# Generator residual block
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

# Discriminator residual block
def block_down(x, filters, kernel_size, padding, use_bias, initializer, downsample=True, activation=True):
    x0 = x
    if activation:
        x = layers.LeakyReLU()(x)
    x = tfa.layers.SpectralNormalization(layers.Conv2D(filters, kernel_size=kernel_size, padding=padding, use_bias=use_bias, kernel_initializer=initializer))(x)
    x = layers.LeakyReLU()(x)
    x = tfa.layers.SpectralNormalization(layers.Conv2D(filters, kernel_size=kernel_size, padding=padding, use_bias=use_bias, kernel_initializer=initializer))(x)
    if downsample:
        x = layers.AveragePooling2D()(x)
        x0 = tfa.layers.SpectralNormalization(layers.Conv2D(filters, 1, padding=padding, use_bias=use_bias, kernel_initializer=initializer))(x0)
        x0 = layers.AveragePooling2D()(x0)
    return layers.Add()([x0, x])

# Generator model
def generator(latent_dim):
    dim = 32
    print(f'Creating generator with {dim} parameters')
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

# Discriminator model
def discriminator():
    dim = 32
    print(f'Creating discriminator with {dim} parameters')
    initializer = tf.keras.initializers.Orthogonal()

    # input layer
    inputs = layers.Input(shape=(256, 256, 3))

    # Add noise to avoid overfit
    # x = layers.GaussianNoise(0.2)(inputs)

    # Downsample: 256x256x3 -> 128x128x8
    x = block_down(inputs, dim, 3, 'same', False, initializer, downsample=True, activation=False)

    # Downsample: 128x128x8 -> 64x64x16
    x = block_down(x, 2 * dim, 3, 'same', False, initializer, downsample=True, activation=True)
    x = SelfAttention()(x)

    # Downsample: 64x64x16 -> 32x32x32
    x = block_down(x, 4 * dim, 3, 'same', False, initializer, downsample=True, activation=True)

    # Downsample: 32x32x32 -> 16x16x64
    x = block_down(x, 8 * dim, 3, 'same', False, initializer, downsample=True, activation=True)

    # Downsample: 16x16x64 -> 8x8x64
    x = block_down(x, 8 * dim, 3, 'same', False, initializer, downsample=True, activation=True)

    # Downsample: 8x8x64 -> 4x4x128
    x = block_down(x, 16 * dim, 3, 'same', False, initializer, downsample=True, activation=True)

    # Final block
    x = block_down(x, 16 * dim, 3, 'same', False, initializer, downsample=False, activation=True)

    # Dense
    x = layers.GlobalAveragePooling2D()(x)

    # output layer
    outputs = layers.Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Initialise or load models
gen = generator(latent_dim)
disc = discriminator()
if continueTrain:
    gen.load_weights(base_path + 'models' + n_folder + '/gen.h5')
    disc.load_weights(base_path + 'models' + n_folder + '/disc.h5')
seed = tf.random.normal((4, latent_dim), seed=42)

# Loss function
model_loss = tf.keras.losses.Hinge()
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0, beta_2=0.999, epsilon=1e-8)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, beta_1=0, beta_2=0.999, epsilon=1e-8)

def discriminator_loss(real_output, fake_output):
    real_loss = model_loss(tf.ones_like(real_output), real_output)
    fake_loss = model_loss(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return model_loss(tf.ones_like(fake_output), fake_output)

# Training: Called every train step
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, latent_dim])
  
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = gen(noise, training=True)
      
        real_output = disc(images, training=True)
        fake_output = disc(generated_images, training=True)
    
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        
    gradients_of_generator = gen_tape.gradient(gen_loss, gen.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, disc.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, disc.trainable_variables))
    
    return gen_loss, disc_loss


# Load the dataset
def load_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, [256, 256])
    image = (image - 127.5) / 127.5
    return image

def load_dataset(folder_path):
    all_images = glob.glob(folder_path + '/*.jpg')
    dataset = tf.data.Dataset.from_tensor_slices(all_images)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

dataset = load_dataset(base_path + 'compressed')
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Saving the current results to track GAN stability
def generate_and_save_images(model, epoch, test_input):
    predictions = model.predict(test_input)

    plt.figure(figsize=(2, 2))
    
    for i in range(predictions.shape[0]):
        plt.subplot(2, 2, i+1)
        plt.imshow(predictions[i, :, :, :] * 0.5 + 0.5)
        plt.axis('off')
        
    plt.savefig(base_path + 'genned' + n_folder + '/image_at_epoch_{:04d}.png'.format(epoch), dpi=300)
    plt.close('all')

# Train the model
trainingStart = time()
for epoch in range(offset, offset + epochs):
    for real_images in dataset:
        gan_loss, disc_loss = train_step(real_images)

    # Print metrics
    print('Epoch', epoch)
    print('Discriminator loss:', disc_loss)
    print('Generator loss:', gan_loss)

    # Save generated images every 10 epochs
    if epoch % 10 == 0:
        print('-------------ETA', ((time() - trainingStart)/(epoch - offset + 1)) * (offset + epochs - epoch - 1) / 3600, 'hours--------------')
        generate_and_save_images(gen, epoch, seed)
    
    # Save the model weights every 100 epochs
    if epoch % 100 == 0 and epoch != offset:
        gen.save(base_path + 'models' + n_folder + '/gen.h5')
        disc.save(base_path + 'models' + n_folder + '/disc.h5')


