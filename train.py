from model import *
from keras.optimizers import Adam
from data_generator import image_generator
from config import *
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 1
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

train_step_per_epoch = len(os.listdir(image_source_dir + 'train/a')) / batch_size + 1
test_step_per_epoch = len(os.listdir(image_source_dir + 'test/a')) / batch_size + 1
train_image_generator = image_generator(image_source_dir + 'train/a',
                                        image_source_dir + 'train/b', batch_size=batch_size,
                                        shuffle=True)
test_image_generator = image_generator(image_source_dir + 'test/a',
                                       image_source_dir + 'test/b', batch_size=batch_size,
                                       shuffle=False)

opt1 = Adam(0.001)
opt2 = Adam(0.001)
discriminator = get_discriminator()
print(discriminator.summary())
generator = get_generator()
generator.compile(optimizer=opt2, loss='mae', metrics=['mean_absolute_percentage_error'])
print(generator.summary())
generator_train = get_generator_training_model(generator, discriminator)
print(generator_train.summary())
if os.path.exists(combined_filepath):
    generator_train.load_weights(combined_filepath, by_name=True)
    generator.load_weights(generator_filepath, by_name=True)
    print('weights loaded!')
discriminator.compile(optimizer=opt1, loss='mse', metrics=['acc'], loss_weights=None, sample_weight_mode=None)
generator_train.compile(optimizer=opt2, loss=['mse', 'mae'],
                        metrics=['mean_absolute_percentage_error'],
                        loss_weights=[1, 10])
real = np.ones((batch_size, 32, 32, 1))
fake = np.zeros((batch_size, 32, 32, 1))
best_loss = 1000

for i in range(epoch):
    train_step = 0
    for imgA, imgB in train_image_generator:
        train_step += 1
        if train_step > train_step_per_epoch:
            test_step = 0
            total_loss = 0
            total_mape = 0
            for imgA, imgB in test_image_generator:
                test_step += 1
                if test_step > test_step_per_epoch:
                    break
                gloss, mape = generator.test_on_batch(imgA, imgB)
                # print generator.metrics_names
                total_loss += gloss
                total_mape += mape
            print('epoch:{} test loss g:{} \n   test mape:{}'.format(i + 1, total_loss / (test_step - 1),
                                                                     total_mape / (test_step - 1)))
            if total_loss / (test_step - 1) < best_loss:
                print('test loss improved from {} to {}'.format(best_loss, total_loss / (test_step - 1)))
                generator_train.save_weights(combined_filepath, overwrite=True)
                generator.save_weights(generator_filepath, overwrite=True)
                best_loss = total_loss / (test_step - 1)
            break
        discriminator.trainable = True
        fakeB = generator.predict(imgA)
        if debug:
            fakeb = (fakeB[0] + 1) * 127.5
            fakeb = np.clip(fakeb, 0, 255)
            fakeb = fakeb.astype(np.uint8)
            fakeb = Image.fromarray(fakeb)
            fakeb.save('predict/' + str(i + 1) + '_' + str(train_step) + '.png')
            print("{} saved".format('predict/' + str(i + 1) + '_' + str(train_step) + '.png'))
            imgb = (imgB[0] + 1) * 127.5
            imgb = np.clip(imgb, 0, 255)
            imgb = imgb.astype(np.uint8)
            imgb = Image.fromarray(imgb)
            imgb.save('predict/' + str(i + 1) + '_' + str(train_step) + '_real.png')
            print("{} saved".format('predict/' + str(i + 1) + '_' + str(train_step) + '_real.png'))
            print('realB:', imgB[0], imgB.shape)
            # print descriminator.trainable
            # print descriminator.summary()
            d_fake = discriminator.predict(np.concatenate((imgA, fakeB), axis=-1))
            d_real = discriminator.predict(np.concatenate((imgA, imgB), axis=-1))
            print('d_real:', np.squeeze(d_real[0]), d_real.shape)
            print('d_fake:', np.squeeze(d_fake[0]), d_fake.shape)
        loss_fake, fake_acc = discriminator.train_on_batch(np.concatenate((imgA, fakeB), axis=-1), fake)
        loss_real, real_acc = discriminator.train_on_batch(np.concatenate((imgA, imgB), axis=-1), real)
        print(
            'epoch:{} train step:{}, loss d_fake:{}, loss d_real:{}, fake_acc:{}, real_acc:{}'.format(i + 1, train_step,
                                                                                                      loss_fake,
                                                                                                      loss_real,
                                                                                                      fake_acc,
                                                                                                      real_acc))
        # print descriminator.metrics_names
        discriminator.trainable = False
        # print generator_train.summary()
        loss = generator_train.train_on_batch([imgA, imgB], [real, imgB])
        # print generator_train.metrics_names
        # print descriminator.trainable
        print('epoch:{} train step:{} loss fool:{} loss g:{}'.format(i + 1, train_step, loss[1], loss[0] - loss[1]))

