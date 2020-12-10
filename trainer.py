from utils import Distiller
from utils import preprocess
from utils import imagenetlabels
from students import facemodel
from students import vgg
import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

data = tfds.load('imagenet2012_subset', shuffle_files=True, data_dir = 'D:\\tensorflow_datasets')
batch_size = 10
input_shape=[224,224]
# labels = imagenetlabels()
num_classes = 1000
train = data['train'].map(lambda x: (x['image'], x['label']))
train= train.map(lambda x, y: (tf.image.resize(x, input_shape, method='nearest'),y))
train = train.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, tf.one_hot(y, 1000)))
train = train.batch(10)
validate = data['validation'].map(lambda x: (x['image'], x['label']))
validate = validate.map(lambda x, y: (tf.image.resize(x, input_shape, method='nearest'),y))
validate = validate.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, tf.one_hot(y, 1000)))
validate = validate.batch(10)

# student = facemodel.create_model()
# student.compile(optimizer=keras.optimizers.Adam(),
#     metrics=[keras.metrics.CategoricalAccuracy()], 
#     loss=keras.losses.CategoricalCrossentropy(from_logits=True))
student = tf.keras.models.load_model('foo.hdf5')


callback_list = [ModelCheckpoint('./models/inet2/weights.{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=True)]
# Distill teacher to student
student.fit(x=train, epochs=5, validation_data=validate)
# Evaluate student on test dataset
score = student.evaluate(validate)
print(score)

student.save('foo1.hdf5')