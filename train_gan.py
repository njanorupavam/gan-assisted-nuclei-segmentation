import numpy as np
import cv2
import os
from gan_model import build_generator, build_discriminator
import tensorflow as tf

IMG_SIZE = 256

def load_images(folder):
    images = []
    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
        images.append(img)
    return np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


X = load_images("data/images")

generator = build_generator()
discriminator = build_discriminator()

discriminator.compile(optimizer='adam', loss='binary_crossentropy')

discriminator.trainable = False

gan_input = tf.keras.Input(shape=(256,256,1))
fake = generator(gan_input)
validity = discriminator(fake)

gan = tf.keras.Model(gan_input, validity)
gan.compile(optimizer='adam', loss='binary_crossentropy')


# 🔥 TRAIN LOOP
epochs = 10
batch_size = 4

for epoch in range(epochs):

    idx = np.random.randint(0, X.shape[0], batch_size)
    real_imgs = X[idx]

    fake_imgs = generator.predict(real_imgs)

    d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((batch_size,1)))
    d_loss_fake = discriminator.train_on_batch(fake_imgs, np.zeros((batch_size,1)))

    g_loss = gan.train_on_batch(real_imgs, np.ones((batch_size,1)))

    print(f"Epoch {epoch+1} | D Loss: {d_loss_real + d_loss_fake} | G Loss: {g_loss}")

generator.save("gan_generator.h5")

print("✅ GAN trained!")