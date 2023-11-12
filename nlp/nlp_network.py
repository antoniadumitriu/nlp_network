import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load texts from the file
with open('texts.txt', 'r') as text_file:
    texts = text_file.read().splitlines()

# Load labels from the file
with open('labels.txt', 'r') as labels_file:
    labels = [float(label) for label in labels_file.read().splitlines()]

# Tokenize the texts
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad the sequences to have consistent length
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences)

# Convert labels to NumPy array with float32 data type
labels = np.array(labels, dtype=np.float32)

# Build the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=16, input_length=padded_sequences.shape[1]))
model.add(tf.keras.layers.LSTM(50))  # LSTM layer with 50 units
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, labels, epochs=10, batch_size=1)

# Now you can use the trained model for predictions on new data
new_texts = ["I can't believe how good this is!", "Very disappointed with the service."]
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(new_sequences, maxlen=padded_sequences.shape[1])

predictions = model.predict(new_padded_sequences)
for i, text in enumerate(new_texts):
    sentiment = "Positive" if predictions[i] > 0.5 else "Negative"
    print(f"Text: {text}, Sentiment: {sentiment}, Confidence: {predictions[i][0]:.4f}")
