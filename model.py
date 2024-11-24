import os
import re
import pickle
import gensim
import gensim.downloader as api
import numpy as np
import tensorflow as tf
import nltk
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, Dropout, LSTM, Input
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Initialize stopwords
STOP_WORDS = set(stopwords.words('english'))

# Function to clean and preprocess text data
def clean_sentence(sentence: str) -> list:
    """
    Cleans the input sentence by removing unwanted characters and stopwords.
    Returns a list of cleaned tokens.
    """
    # Remove XML-like review tags
    sentence = re.sub(r"(<\/?review_text>)", '', sentence)

    # Convert to lowercase
    sentence = sentence.lower()

    # Remove email addresses and URLs
    sentence = re.sub(r"(\bhttp.+?|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)", '', sentence)

    # Replace '@' symbols with 'a' (for Twitter handles)
    sentence = re.sub(r'@', 'a', sentence)

    # Remove punctuation
    sentence = re.sub(r"[^\w\s(\w+\-\w+)]", '', sentence)

    # Tokenize the sentence and remove stopwords
    sentence = word_tokenize(sentence)
    sentence = [word for word in sentence if word not in STOP_WORDS]

    return sentence


# Function to load and process data from files
def process_data(folders, path, label):
    """
    Processes the reviews from the specified folders and returns cleaned text and labels.
    """
    x_data, y_data = [], []
    review_pattern = re.compile(r"<review_text>.*?</review_text>", flags=re.DOTALL)

    for folder in folders:
        for review_type in ["negative", "positive"]:
            file_path = f"{path}{folder}/{review_type}.review"
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    reviews = file.read()
                    reviews = re.findall(review_pattern, reviews)
                    print(f"\tReading {len(reviews)} {review_type.capitalize()} reviews from {folder}")
                    for sentence in reviews:
                        x_data.append(clean_sentence(sentence))
                        y_data.append(label if review_type == "positive" else 0)
            else:
                print(f"File not found: {file_path}")

    return x_data, y_data


# Set paths for training and testing data
data_path = "content/sorted_data_acl/"
train_folders = ["books", "dvd", "electronics"]
test_folders = ["kitchen_&_housewares"]

# Process training data
print("Reading training data:")
x_train, y_train = process_data(train_folders, data_path, label=1)

# Process test data
print("Reading test data:")
x_test, y_test = process_data(test_folders, data_path, label=1)

# Save training data to files
with open("x_train.pkl", "wb") as file:
    pickle.dump(x_train, file)
print("x_train saved to 'x_train.pkl'.")

with open("y_train.pkl", "wb") as file:
    pickle.dump(y_train, file)
print("y_train saved to 'y_train.pkl'.")

# Save test data to files
with open("x_test.pkl", "wb") as file:
    pickle.dump(x_test, file)
print("x_test saved to 'x_test.pkl'.")

with open("y_test.pkl", "wb") as file:
    pickle.dump(y_test, file)
print("y_test saved to 'y_test.pkl'.")


# =============================================================================
# Data Exploration and Analysis
# =============================================================================

# Calculate sentence lengths for the training data
sentence_lengths = [len(sentence) for sentence in x_train]
sentence_lengths.sort()

print("Max size:", max(sentence_lengths))
print("Min size:", min(sentence_lengths))
print("Top 20 sizes:", sentence_lengths[-20:])

# Plot histogram of sentence lengths
plt.hist(sentence_lengths, bins=[0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 300, 2000])
plt.xlabel("Sentence Length")
plt.ylabel("Frequency")
plt.title("Histogram of Sentence Lengths in Training Data")
plt.show()

# =============================================================================
# Vocabulary Construction
# =============================================================================

# Build vocabulary from training data
vocab = set(word for sentence in x_train for word in sentence)
vocab.add('')  # Add dummy word for padding or unknowns

print("Vocabulary size:", len(vocab))

# Create word-to-ID and ID-to-word mappings
word_to_id = {word: idx for idx, word in enumerate(vocab)}
id_to_word = {idx: word for idx, word in enumerate(vocab)}

# =============================================================================
# Sentence Encoding and Padding
# =============================================================================

# Function to encode sentences into integer sequences
def encode_sentence(sentence):
    """
    Encodes each word in a sentence into its corresponding ID from the vocabulary.
    """
    return [word_to_id.get(word, word_to_id['']) for word in sentence]

# Encode training and test sentences
x_train_encoded = [encode_sentence(sentence) for sentence in x_train]
x_test_encoded = [encode_sentence(sentence) for sentence in x_test]

# Define max sequence length for padding
MAX_SEQ_LEN = 125
dummy_id = word_to_id['']

# Pad the encoded sentences to ensure consistent length
x_train_padded = pad_sequences(x_train_encoded, maxlen=MAX_SEQ_LEN, dtype='int', padding='post', truncating='post', value=dummy_id)
x_test_padded = pad_sequences(x_test_encoded, maxlen=MAX_SEQ_LEN, dtype='int', padding='post', truncating='post', value=dummy_id)

print("Train shape:", x_train_padded.shape)
print("Test shape:", x_test_padded.shape)

# =============================================================================
# Label Preparation
# =============================================================================

# Convert labels to numpy arrays and reshape them for model training
y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)

print("Train labels shape:", y_train.shape)
print("Test labels shape:", y_test.shape)

# =============================================================================
# Load Pretrained Word2Vec Model and Prepare Embedding Matrix
# =============================================================================

# Load GloVe model (Twitter 200-dimensional embeddings)
word2vec_model_path = "glove-twitter-200.model"
if not os.path.exists(word2vec_model_path):
    print("Loading Word2Vec model...")
    w2v_model = api.load('glove-twitter-200')
    w2v_model.save(word2vec_model_path)
    print(f"Model saved locally as {word2vec_model_path}.")
else:
    w2v_model = gensim.models.KeyedVectors.load(word2vec_model_path)
    print("Model loaded from local storage.")

# Create embedding matrix
embedding_size = w2v_model['mahmoud'].shape[0]
embedding_matrix = np.zeros((len(vocab), embedding_size))

# Fill the embedding matrix with pretrained embeddings
for word, idx in word_to_id.items():
    if word in w2v_model:
        embedding_matrix[idx] = w2v_model[word]
    else:
        embedding_matrix[idx] = np.zeros(embedding_size)  # Use zero for out-of-vocabulary words

print("Embedding matrix shape:", embedding_matrix.shape)

# =============================================================================
# Build and Compile LSTM Model
# =============================================================================

# Clear any previous session to avoid memory issues
tf.keras.backend.clear_session()

# Build the model using Bidirectional LSTM
lstm_model = Sequential(name='SentimentAnalysisModel')
lstm_model.add(Input(shape=(MAX_SEQ_LEN,), dtype='int32'))
lstm_model.add(Embedding(input_dim=len(vocab), 
                        output_dim=embedding_size, 
                        weights=[embedding_matrix], 
                        trainable=False))  # Freeze the embeddings
lstm_model.add(Bidirectional(LSTM(units=64, return_sequences=True)))  # Bidirectional LSTM layer
lstm_model.add(Dropout(0.5))  # Dropout to reduce overfitting
lstm_model.add(LSTM(units=64))  # Second LSTM layer
lstm_model.add(Dropout(0.5))  # Dropout layer
lstm_model.add(Dense(1, activation='sigmoid'))  # Output layer

# Compile the model with the Adam optimizer
lstm_model.compile(optimizer=Adam(learning_rate=0.0001), 
                   loss='binary_crossentropy', 
                   metrics=['accuracy'])

# Save the model
lstm_model.save('lstm_model.keras')
print("Model saved as 'lstm_model.keras'.")

# =============================================================================
# Train the Model
# =============================================================================

# Shuffle the training data
train_data, train_labels = shuffle(x_train_padded, y_train, random_state=42)

# Train the model
lstm_model.fit(train_data, 
               train_labels, 
               validation_split=0.20, 
               batch_size=50, 
               epochs=50)

# =============================================================================
# Custom Prediction Function
# =============================================================================

def lstm_predict(sentence: str):
    """
    Predicts the sentiment of a given sentence using the trained LSTM model.
    """
    # Clean and encode the sentence
    cleaned_sentence = clean_sentence(sentence)
    encoded_sentence = encode_sentence(cleaned_sentence)
    padded_sentence = pad_sequences([encoded_sentence], maxlen=MAX_SEQ_LEN, dtype='int32', padding='post', truncating='post', value=dummy_id)
    
    # Predict the sentiment (0 = Negative, 1 = Positive)
    prediction = round(lstm_model.predict(padded_sentence)[0][0])
    
    # Output the prediction
    if prediction == 0:
        print("Negative Review")
    elif prediction == 1:
        print("Positive Review")
    else:
        print('Error')
