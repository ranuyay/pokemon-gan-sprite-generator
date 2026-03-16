import tensorflow as tf

def initialize_weights(model, stddev=0.02):
    for variable in model.trainable_variables:
        variable.assign(tf.random.normal(shape=variable.shape, mean=0.0, stddev=stddev))


def make_dcgan_loss():
    return tf.keras.losses.BinaryCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.SUM
    )


def make_dcgan_optimizers(lr_g=0.0005, lr_d=0.0005):
    optimizer_d = tf.keras.optimizers.Adam(
        learning_rate=lr_d,
        beta_1=0.5,
        beta_2=0.999
    )
    optimizer_g = tf.keras.optimizers.Adam(
        learning_rate=lr_g,
        beta_1=0.5,
        beta_2=0.999
    )
    return optimizer_d, optimizer_g


@tf.function
def train_discriminator_step(real_images, net_d, net_g, loss_fn, optimizer_d, latent_dim):
    batch_size = tf.shape(real_images)[0]
    noise = tf.random.normal((batch_size, 1, 1, latent_dim))

    with tf.GradientTape() as tape:
        fake_images = net_g(noise, training=True)

        real_logits = net_d(real_images, training=True)
        fake_logits = net_d(fake_images, training=True)

        real_labels = tf.ones_like(real_logits)
        fake_labels = tf.zeros_like(fake_logits)

        loss_real = loss_fn(real_labels, real_logits)
        loss_fake = loss_fn(fake_labels, fake_logits)
        loss_d = loss_real + loss_fake

    gradients = tape.gradient(loss_d, net_d.trainable_variables)
    optimizer_d.apply_gradients(zip(gradients, net_d.trainable_variables))
    return loss_d


@tf.function
def train_generator_step(net_d, net_g, loss_fn, optimizer_g, batch_size, latent_dim):
    noise = tf.random.normal((batch_size, 1, 1, latent_dim))

    with tf.GradientTape() as tape:
        fake_images = net_g(noise, training=True)
        fake_logits = net_d(fake_images, training=True)

        target_labels = tf.ones_like(fake_logits)
        loss_g = loss_fn(target_labels, fake_logits)

    gradients = tape.gradient(loss_g, net_g.trainable_variables)
    optimizer_g.apply_gradients(zip(gradients, net_g.trainable_variables))
    return loss_g


def train_dcgan(
    dataset,
    net_d,
    net_g,
    num_epochs=40,
    latent_dim=100,
    lr_g=0.0005,
    lr_d=0.0005,
    checkpoint_prefix="models/dcgan"
):
    loss_fn = make_dcgan_loss()
    optimizer_d, optimizer_g = make_dcgan_optimizers(lr_g=lr_g, lr_d=lr_d)

    initialize_weights(net_d)
    initialize_weights(net_g)

    history = {
        "loss_d": [],
        "loss_g": []
    }

    best_epoch = 1
    best_gap = float("inf")

    for epoch in range(1, num_epochs + 1):
        epoch_loss_d = 0.0
        epoch_loss_g = 0.0
        batches = 0

        for real_images, _ in dataset:
            batch_size = real_images.shape[0]

            loss_d = train_discriminator_step(
                real_images,
                net_d,
                net_g,
                loss_fn,
                optimizer_d,
                latent_dim
            )

            loss_g = train_generator_step(
                net_d,
                net_g,
                loss_fn,
                optimizer_g,
                batch_size,
                latent_dim
            )

            epoch_loss_d += float(loss_d)
            epoch_loss_g += float(loss_g)
            batches += 1

        avg_loss_d = epoch_loss_d / batches
        avg_loss_g = epoch_loss_g / batches

        history["loss_d"].append(avg_loss_d)
        history["loss_g"].append(avg_loss_g)

        gap = abs(avg_loss_g - avg_loss_d)
        if gap < best_gap and avg_loss_d > 0.1:
            best_gap = gap
            best_epoch = epoch
            net_g.save_weights(f"{checkpoint_prefix}_generator.weights.h5")
            net_d.save_weights(f"{checkpoint_prefix}_discriminator.weights.h5")

        print(
            f"Epoch {epoch:03d}/{num_epochs} | "
            f"loss_D {avg_loss_d:.3f} | "
            f"loss_G {avg_loss_g:.3f}"
        )

    return history, best_epoch