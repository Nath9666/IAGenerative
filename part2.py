import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
y_train = tf.keras.utils.to_categorical(y_train, 10)

latent_dim = 100
num_classes = 10

# Conditional Generator
def build_generator():
    noise = layers.Input(shape=(latent_dim,))
    label = layers.Input(shape=(num_classes,))
    x = layers.Concatenate()([noise, label])
    
    x = layers.Dense(7 * 7 * 256)(x)
    x = layers.Reshape((7, 7, 256))(x)
    x = layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    img = layers.Conv2DTranspose(1, kernel_size=7, activation="sigmoid", padding="same")(x)
    
    model = tf.keras.Model([noise, label], img)
    return model

# Conditional Discriminator
def build_discriminator():
    img = layers.Input(shape=(28, 28, 1))
    label = layers.Input(shape=(num_classes,))
    label_embedding = layers.Dense(28 * 28)(label)
    label_embedding = layers.Reshape((28, 28, 1))(label_embedding)
    
    x = layers.Concatenate()([img, label_embedding])
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Flatten()(x)
    validity = layers.Dense(1, activation="sigmoid")(x)
    
    model = tf.keras.Model([img, label], validity)
    return model

# Build and compile the models
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer=tf.keras.optimizers.Adam(0.0002), loss="binary_crossentropy", metrics=["accuracy"])
discriminator.trainable = False

noise = layers.Input(shape=(latent_dim,))
label = layers.Input(shape=(num_classes,))
img = generator([noise, label])
validity = discriminator([img, label])
cgan = tf.keras.Model([noise, label], validity)
cgan.compile(optimizer=tf.keras.optimizers.Adam(0.0002), loss="binary_crossentropy")

# Training the cGAN
def train_cgan(generator, discriminator, cgan, epochs, batch_size=128):
    for epoch in range(epochs):
        for _ in range(batch_size):
            # Train discriminator
            noise = tf.random.normal([batch_size, latent_dim])
            fake_labels = tf.keras.utils.to_categorical(np.random.randint(0, num_classes, batch_size), num_classes)
            fake_images = generator.predict([noise, fake_labels])
            real_images = x_train[np.random.randint(0, x_train.shape[0], batch_size)]
            real_labels = y_train[np.random.randint(0, y_train.shape[0], batch_size)]
            
            real_validity = tf.ones((batch_size, 1))
            fake_validity = tf.zeros((batch_size, 1))
            
            d_loss_real = discriminator.train_on_batch([real_images, real_labels], real_validity)
            d_loss_fake = discriminator.train_on_batch([fake_images, fake_labels], fake_validity)
            
            # Train generator
            misleading_validity = tf.ones((batch_size, 1))
            g_loss = cgan.train_on_batch([noise, fake_labels], misleading_validity)
        
        print(f"Epoch {epoch + 1}/{epochs}, D Loss: {d_loss_real[0] + d_loss_fake[0]}, G Loss: {g_loss}")

train_cgan(generator, discriminator, cgan, epochs=10)

# Generate and visualize images
def generate_images(generator, n_images):
    noise = tf.random.normal([n_images, latent_dim])
    labels = tf.keras.utils.to_categorical(np.arange(n_images) % num_classes, num_classes)
    generated_images = generator.predict([noise, labels])
    fig, axes = plt.subplots(1, n_images, figsize=(20, 4))
    for i, img in enumerate(generated_images):
        axes[i].imshow(img.squeeze(), cmap="gray")
        axes[i].axis("off")
    plt.show()
    plt.savefig('generated_images_cgan.png')

generate_images(generator, 10)