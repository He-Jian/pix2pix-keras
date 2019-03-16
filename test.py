import keras.backend as K
from model import *
from keras.optimizers import Adam
from PIL import Image
from config import *
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 1
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

K.set_learning_phase(0)  # test mode

if direction == 'a2b':
    test_image_dir = image_source_dir + 'test/a'
else:
    test_image_dir = image_source_dir + 'test/b'
test_image_list = os.listdir(test_image_dir)
opt = Adam(0.001)

model = get_generator()
#generator.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
#print generator.summary()
#generator.load_weights(filepath)#, by_name=True)
#discriminator = get_discriminator()
#model = get_generator_training_model(generator, discriminator)
print model.summary()
if os.path.exists(generator_filepath):
    model.load_weights(generator_filepath)#, by_name=True)
    print('weights loaded!')
model.compile(optimizer=opt, loss='mse',
                                            metrics=['mean_absolute_percentage_error'],)

n_batch=len(test_image_list)/batch_size

for i in range(n_batch):
    batch_x=[]
    for j in range(batch_size):

        a = Image.open(os.path.join(test_image_dir, test_image_list[i*batch_size+j])).convert('RGB')
        a = a.resize((image_size, image_size), Image.BICUBIC)
        a = np.asarray(a) / 127.5 - 1
        batch_x.append(a)
    batch_x= np.array(batch_x)
    fakeb_batch = model.predict(batch_x)
    for j in range(len(batch_x)):
        fakeb = fakeb_batch[j]
        fakeb = (fakeb + 1) * 127.5
        fakeb = np.clip(fakeb, 0, 255)
        fakeb = fakeb.astype(np.uint8)
        fakeb = Image.fromarray(fakeb)
        fakeb.save('predict/' + test_image_list[i*batch_size+j])
        print("{} saved".format('predict/' + test_image_list[i*batch_size+j]))


