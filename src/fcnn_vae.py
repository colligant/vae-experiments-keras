import tensorflow.keras as keras
import tensorflow as tf
import os
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
import numpy as np

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        # tf.keras.backend.random_normal?!
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5*z_log_var) * epsilon


class Encoder(layers.Layer):
    """Maps satellite image patches to a feature representation """

    def __init__(self, input_channels, latent_channels, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.latent_channels = latent_channels
        self.inp = layers.Input(shape=(None, None, input_channels))
        base = 8
        self.c1 = layers.Conv2D(base, kernel_size=3, activation='relu',
                padding='same')#, kernel_regularizer=l2(0.001))
        self.c2 = layers.Conv2D(base, kernel_size=3, activation='relu',
                padding='same')#, kernel_regularizer=l2(0.001))
        self.d1 = layers.MaxPooling2D()
        self.c3 = layers.Conv2D(base*2, kernel_size=3, activation='relu',
                padding='same')#, kernel_regularizer=l2(0.001))
        self.d2 = layers.MaxPooling2D()
        self.c4 = layers.Conv2D(base*2, kernel_size=3, activation='relu',
                padding='same')#, kernel_regularizer=l2(0.001))
        self.mean = layers.Conv2D(latent_channels, kernel_size=1)
        self.log_var = layers.Conv2D(latent_channels, kernel_size=1)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.c2(x)
        x = self.d1(x)
        x = self.c3(x)
        x = self.d2(x)
        x = self.c4(x)
        z_mean = self.mean(x)
        z_log_var = self.log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, output_channels, latent_channels, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        base = 8
        self.c1 = layers.Conv2D(base*2, kernel_size=3, activation='relu', 
                padding='same')#, kernel_regularizer=l2(0.001))
        self.c2 = layers.Conv2D(base*2, kernel_size=3, activation='relu',
                padding='same')#, kernel_regularizer=l2(0.001))
        self.u1 = layers.UpSampling2D()
        self.c3 = layers.Conv2D(base, kernel_size=3, activation='relu',
                padding='same')#, kernel_regularizer=l2(0.001))
        self.u2 = layers.UpSampling2D()
        self.c4 = layers.Conv2D(base, kernel_size=3, activation='relu',
                padding='same')#, kernel_regularizer=l2(0.001))
        self.cout = layers.Conv2D(output_channels, kernel_size=1,
                activation='sigmoid')

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.c2(x)
        x = self.u1(x)
        x = self.c3(x)
        x = self.u2(x)
        x = self.c4(x)
        return self.cout(x)


class VariationalAutoEncoder(keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(
        self,
        latent_channels,
        name="autoencoder",
        **kwargs
    ):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.encoder = Encoder(input_channels=1, latent_channels=latent_channels)
        self.decoder = Decoder(output_channels=1, latent_channels=latent_channels)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed


latent_channels = 10

vae = VariationalAutoEncoder(latent_channels)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
bce_loss_fn = tf.keras.losses.BinaryCrossentropy()
mse_loss_fn = tf.keras.losses.MeanSquaredError()

loss_metric = tf.keras.metrics.Mean()

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = tf.expand_dims(x_train, -1)
x_test = tf.expand_dims(x_test, -1)

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

test_dataset = tf.data.Dataset.from_tensor_slices(x_test)
test_dataset = test_dataset.shuffle(buffer_size=1024).batch(64)

epochs = 5

# Iterate over epochs.
if not os.path.isfile('fcnn_vae_2channel1down.index'):
    for epoch in range(epochs):
        print("Start of epoch %d" % (epoch+1,))
        # Iterate over the batches of the dataset.
        for step, x_batch_train in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                reconstructed = vae(x_batch_train)
                # Compute reconstruction loss
                loss = bce_loss_fn(x_batch_train, reconstructed)
                loss += sum(vae.losses)  # Add KLD regularization loss

            grads = tape.gradient(loss, vae.trainable_weights)
            optimizer.apply_gradients(zip(grads, vae.trainable_weights))

            loss_metric(loss)

            if step % 100 == 0:
                print("step %d: mean loss = %.4f" % (step, loss_metric.result()))
    vae.save_weights('fcnn_vae_2channel1down')
else:
    vae.load_weights('fcnn_vae')

plt.style.use('dark_background')
fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(4,4))
for i in range(4):
    for j in range(4):
        rand_code = tf.random.normal(shape=(1,7,7,latent_channels))
        ax[i, j].imshow(tf.squeeze(vae.decoder(rand_code)))
        ax[i, j].axis('off')
        ax[i, j].set_aspect('auto')

plt.subplots_adjust(wspace=0.02, hspace=0.02)

# plt.savefig('fcnn_vae.png', bbox_inches='tight')
plt.show()

# for x_batch_test in test_dataset:
#     break


# plt.style.use('dark_background')
# fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(4,4))
# for i in range(4):
#     for j in range(4):
#         _, _, encoded = vae.encoder(x_batch_test)
#         ax[i, j].imshow(tf.squeeze(encoded[i]))
#         ax[i, j].axis('off')
#         ax[i, j].set_aspect('auto')
# 
# plt.subplots_adjust(wspace=0.02, hspace=0.02)
# 
# plt.savefig('fcnn_vae_embedding2downsample.png', bbox_inches='tight')
# plt.show()
