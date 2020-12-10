from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

def create_model():
    model = Sequential()
    model.add(Conv2D(256, 3, input_shape=(224,224,3), activation="relu", padding="same"))
    model.add(Conv2D(256, 3, activation="relu", padding="same"))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25)),
    model.add(Conv2D(128, 3, activation="relu", padding="same"))
    model.add(Conv2D(128, 3, activation="relu", padding="same"))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25)),
    model.add(Conv2D(64, 3, activation="relu", padding="same"))
    model.add(Conv2D(64, 3, activation="relu", padding="same"))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25)),
    model.add(Flatten())
    model.add(Dropout(0.25)),
    model.add(Dense(500, activation="relu"))
    model.add(Dropout(0.25)),
    model.add(Dense(500, activation="relu"))
    model.add(Dense(1000, activation="softmax"))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model

model = create_model()
model.summary()