import tensorflow as tf


class GBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size=4, strides=2, padding="same", **kwargs):
        super().__init__(**kwargs)
        self.conv2d_trans = tf.keras.layers.Conv2DTranspose(
            out_channels,
            kernel_size,
            strides,
            padding,
            use_bias=False
        )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.ReLU()

    def call(self, x, training=False):
        x = self.conv2d_trans(x)
        x = self.batch_norm(x, training=training)
        x = self.activation(x)
        return x


class CriticBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        out_channels,
        kernel_size=4,
        strides=2,
        padding="same",
        alpha=0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.conv2d = tf.keras.layers.Conv2D(
            out_channels,
            kernel_size,
            strides,
            padding,
            use_bias=False
        )
        self.activation = tf.keras.layers.LeakyReLU(alpha)

    def call(self, x):
        x = self.conv2d(x)
        x = self.activation(x)
        return x


def build_wgan_generator(latent_dim=100, base_filters=64):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1, 1, latent_dim)),
        GBlock(base_filters * 8, strides=1, padding="valid"),
        GBlock(base_filters * 4),
        GBlock(base_filters * 2),
        GBlock(base_filters),
        tf.keras.layers.Conv2DTranspose(
            3,
            kernel_size=4,
            strides=2,
            padding="same",
            use_bias=False,
            activation="tanh"
        )
    ])


def build_wgan_critic(base_filters=64):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64, 64, 3)),
        CriticBlock(base_filters),
        CriticBlock(base_filters * 2),
        CriticBlock(base_filters * 4),
        CriticBlock(base_filters * 8),
        tf.keras.layers.Conv2D(1, kernel_size=4, use_bias=False)
    ])


def make_wgan_optimizers(lr_g=0.0001, lr_c=0.0001):
    optimizer_c = tf.keras.optimizers.Adam(
        learning_rate=lr_c,
        beta_1=0.0,
        beta_2=0.9
    )
    optimizer_g = tf.keras.optimizers.Adam(
        learning_rate=lr_g,
        beta_1=0.0,
        beta_2=0.9
    )
    return optimizer_c, optimizer_g


def initialize_weights(model, stddev=0.02):
    for variable in model.trainable_variables:
        variable.assign(
            tf.random.normal(shape=variable.shape, mean=0.0, stddev=stddev)
        )


def gradient_penalty(critic, real_images, fake_images):
    batch_size = tf.shape(real_images)[0]
    alpha = tf.random.uniform(
        shape=[batch_size, 1, 1, 1],
        minval=0.0,
        maxval=1.0
    )

    interpolated = alpha * real_images + (1.0 - alpha) * fake_images

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = critic(interpolated, training=True)

    grads = tape.gradient(pred, interpolated)
    grads = tf.reshape(grads, [batch_size, -1])
    grad_norm = tf.norm(grads, axis=1)

    gp = tf.reduce_mean((grad_norm - 1.0) ** 2)
    return gp


@tf.function
def train_critic_step(real_images, critic, generator, optimizer_c, latent_dim, lambda_gp):
    batch_size = tf.shape(real_images)[0]
    noise = tf.random.normal((batch_size, 1, 1, latent_dim))

    with tf.GradientTape() as tape:
        fake_images = generator(noise, training=True)

        critic_real = critic(real_images, training=True)
        critic_fake = critic(fake_images, training=True)

        wasserstein_loss = tf.reduce_mean(critic_fake) - tf.reduce_mean(critic_real)
        gp = gradient_penalty(critic, real_images, fake_images)
        loss_c = wasserstein_loss + lambda_gp * gp

    gradients = tape.gradient(loss_c, critic.trainable_variables)
    optimizer_c.apply_gradients(zip(gradients, critic.trainable_variables))

    return loss_c, wasserstein_loss, gp


@tf.function
def train_generator_step(generator, critic, optimizer_g, batch_size, latent_dim):
    noise = tf.random.normal((batch_size, 1, 1, latent_dim))

    with tf.GradientTape() as tape:
        fake_images = generator(noise, training=True)
        critic_fake = critic(fake_images, training=True)
        loss_g = -tf.reduce_mean(critic_fake)

    gradients = tape.gradient(loss_g, generator.trainable_variables)
    optimizer_g.apply_gradients(zip(gradients, generator.trainable_variables))

    return loss_g


def train_wgan_gp(
    dataset,
    generator,
    critic,
    num_epochs=40,
    latent_dim=100,
    lr_g=0.0001,
    lr_c=0.0001,
    lambda_gp=10.0,
    n_critic=5,
    fid_interval=10,
    real_features_cache=None,
    evaluate_fid_fn=None,
    experiment_name="wgan_gp",
    model_dir="models"
):
    optimizer_c, optimizer_g = make_wgan_optimizers(lr_g=lr_g, lr_c=lr_c)

    initialize_weights(generator)
    initialize_weights(critic)

    history = {
        "critic_loss": [],
        "generator_loss": [],
        "wasserstein_loss": [],
        "gradient_penalty": [],
        "fid": []
    }

    best_epoch = 1
    best_fid = float("inf")

    for epoch in range(1, num_epochs + 1):
        epoch_critic_loss = 0.0
        epoch_generator_loss = 0.0
        epoch_wasserstein = 0.0
        epoch_gp = 0.0
        critic_steps = 0
        generator_steps = 0

        for real_images, _ in dataset:
            batch_size = real_images.shape[0]

            for _ in range(n_critic):
                loss_c, wass_loss, gp = train_critic_step(
                    real_images,
                    critic,
                    generator,
                    optimizer_c,
                    latent_dim,
                    lambda_gp
                )
                epoch_critic_loss += float(loss_c)
                epoch_wasserstein += float(wass_loss)
                epoch_gp += float(gp)
                critic_steps += 1

            loss_g = train_generator_step(
                generator,
                critic,
                optimizer_g,
                batch_size,
                latent_dim
            )
            epoch_generator_loss += float(loss_g)
            generator_steps += 1

        avg_critic_loss = epoch_critic_loss / max(critic_steps, 1)
        avg_generator_loss = epoch_generator_loss / max(generator_steps, 1)
        avg_wasserstein = epoch_wasserstein / max(critic_steps, 1)
        avg_gp = epoch_gp / max(critic_steps, 1)

        history["critic_loss"].append(avg_critic_loss)
        history["generator_loss"].append(avg_generator_loss)
        history["wasserstein_loss"].append(avg_wasserstein)
        history["gradient_penalty"].append(avg_gp)

        print(
            f"Epoch {epoch:03d}/{num_epochs} | "
            f"critic_loss {avg_critic_loss:.3f} | "
            f"generator_loss {avg_generator_loss:.3f} | "
            f"wass {avg_wasserstein:.3f} | "
            f"gp {avg_gp:.3f}"
        )

        if (
            evaluate_fid_fn is not None
            and real_features_cache is not None
            and epoch % fid_interval == 0
        ):
            fid = evaluate_fid_fn(
                generator,
                latent_dim,
                real_features_cache,
                experiment_name=experiment_name
            )
            history["fid"].append((epoch, fid))

            if fid < best_fid:
                best_fid = fid
                best_epoch = epoch
                generator.save_weights(f"{model_dir}/best_G_{experiment_name}.weights.h5")
                critic.save_weights(f"{model_dir}/best_C_{experiment_name}.weights.h5")

                print(f"  >> Saved best checkpoint at epoch {epoch} (FID {fid:.2f})")

    return history, best_epoch, best_fid