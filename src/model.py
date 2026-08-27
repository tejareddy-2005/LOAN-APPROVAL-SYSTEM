from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout

def build_model(input_shape):
    model = Sequential([
        Conv1D(32, 2, activation='relu', input_shape=input_shape),
        
        Flatten(),
        
        Dense(64, activation='relu'),
        Dropout(0.5),
        
        Dense(32, activation='relu'),
        Dropout(0.3),
        
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model