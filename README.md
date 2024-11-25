#	Introduction:
_This projects aims to train a machine learning model to analyse the sentiments of customers reviews and create a prediction function to allow for an input text to be subject to sentiment analysis based on the trained model._

##	Problem statement:
Our goal was to analyse the sentiments expressed in Amazon reviews and classify them as positive or negative. We leveraged machine learning techniques to build a sentiment analysis model that can automatically classify a given text input based on its sentiment. 
Specifically, **we used _Long Short-Term Memory (LSTM)_ networks within a _Bi-directional architecture_, powered by _pretrained word embeddings_ and train/test it on the dataset found here**: https://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html 
The following steps were taken:
-Data preparation: The data was loaded, cleaned and labelled and split to training and testing data. 
*Data exploration: Data metrics are extracted and visualised to facilitate better model tuning in subsequent steps.
*Vocabulary construction: A vocabulary set is created from the data.
*Sentence encoding and padding: The Data is encoded and padded for uniformity. 
*Label preparation: Label data is also reformated for better model compliance.
*Model fitting: The model is prepared, build, compiled and trained.
*Custom prediction: A function is provided to perform sentiment analysis on an text input.


##	Code analysis:
Listed below are parts of the code used with a brief explanation of it's function, along with any improvements done and/or challenges faced. Comments are also included inline to achieve best readability and follow best coding practices.

### 1) Importing libraries
Listed below is a list of the libraries/modules/packages used with a brief explanation of their function.


```os:``` Provides a way to interact with the operating system, including file and directory management.

```re:``` Enables regular expression matching for pattern searching and manipulation in strings.

```pickle:``` Serializes and deserializes Python objects, allowing them to be saved to or loaded from files.

```gensim:``` A library for topic modeling and document similarity, often used with pretrained word embeddings.

```gensim.downloader:``` Facilitates downloading and loading pretrained word embeddings (e.g., GloVe, Word2Vec).

```numpy:``` A library for numerical computing, including array manipulation and mathematical operations.

```tensorflow:``` An open-source library for building and training machine learning and deep learning models.

```nltk:``` A library for natural language processing, including tokenization, stemming, and stopword removal.

```matplotlib.pyplot:``` A plotting library for creating visualizations like histograms and graphs.

```nltk.corpus.stopwords:``` Provides predefined lists of stopwords for various languages, useful in text preprocessing.

```nltk.tokenize.word_tokenize:``` Splits text into individual words or tokens for analysis.

```tensorflow.keras.preprocessing.sequence.pad_sequences:``` Pads sequences to ensure uniform length for model input.

```tensorflow.keras.models.Sequential:``` Enables the creation of neural networks by stacking layers sequentially.

```tensorflow.keras.layers:``` Provides prebuilt layers (e.g., Dense, LSTM, Dropout) for designing neural networks.

```tensorflow.keras.optimizers.Adam:``` Implements the Adam optimization algorithm for training deep learning models.

```sklearn.utils.shuffle:``` Randomly shuffles data and labels to prevent patterns during training.


### 2) Data Collection & Preprocessing: 
We imported the product reviews from multiple categories (books, electronics, DVDs). Cleaned the data by removing unwanted tags, URLs, emails, and special characters. Tokenized sentences and removed stopwords.

We used the NLTK stopwords dataset, which contains a list of common words (like "the", "is", "and") that are often removed during text processing because they don't contribute significant meaning to the analysis.
```
# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
```
The Punct tokenizer model was used to split the sentences into smaller words/tokens. 

> [!CAUTION]
> At this point we'd like to note the importance of choosing when to call, or where to store/cache large
> data structures or models as they can severelly impact your codes speed.
```
# Initialize stopwords
STOP_WORDS = set(stopwords.words('english'))
```

The clean_sentence function preprocesses text by removing XML tags, URLs, emails, special characters, and stopwords. It converts text to lowercase, normalizes @ to a, tokenizes the text into words, and returns a list of cleaned tokens for further use.
```
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

    # Replace '@' symbols with 'a' (for obscenities prevention)
    sentence = re.sub(r'@', 'a', sentence)

    # Remove punctuation
    sentence = re.sub(r"[^\w\s(\w+\-\w+)]", '', sentence)

    # Tokenize the sentence and remove stopwords
    sentence = word_tokenize(sentence)
    sentence = [word for word in sentence if word not in STOP_WORDS]

    return sentence
```

The process_data function reads review files from specified folders, extracts text matching a review pattern, cleans the text using clean_sentence, and assigns labels (positive or negative). It returns the processed text (x_data) and their corresponding labels (y_data).
```
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
```

### 3) Data exploration

-Sentence length analysis: The length of each training sentence is calculated and sorted to understand the distribution of sentence lengths. Key metrics, such as the maximum, minimum, and top 20 longest sentences, are displayed.
-Visualization: A histogram of sentence lengths is plotted to provide a visual representation of sentence length frequency, offering insights into data variability and guiding padding decisions for model training.
```
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
```

### 4) Vocabulary construction

Vocabulary building: A vocabulary set is created from the training data, containing all unique words. A special placeholder ('') is added for padding and unknown words to ensure uniformity during model input.
	Mappings: Two dictionaries are generated:
		-word_to_id: Maps each word to a unique numerical ID.
		-id_to_word: Maps numerical IDs back to their corresponding words. These mappings are essential for encoding and decoding text during processing and inference.
```
# Build vocabulary from training data
vocab = set(word for sentence in x_train for word in sentence)
vocab.add('')  # Add dummy word for padding or unknowns

print("Vocabulary size:", len(vocab))

# Create word-to-ID and ID-to-word mappings
word_to_id = {word: idx for idx, word in enumerate(vocab)}
id_to_word = {idx: word for idx, word in enumerate(vocab)}
```

### 5) Sentence encoding and padding

Encoding: Sentences are converted into sequences of numerical IDs based on the vocabulary mapping. Unknown words are replaced with the placeholder ID.
Padding: To ensure that all input sentences have a uniform length for the LSTM model, the sequences are padded or truncated to a predefined MAX_SEQ_LEN (125). Padding is done using the placeholder ID, ensuring consistent input dimensions.
```
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
```
### 6) Label preparation

Formatting: Labels for training and testing data are converted into NumPy arrays and reshaped into columns. This reshaping aligns the label format with the model's expected input structure.
```
# Convert labels to numpy arrays and reshape them for model training
y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)

print("Train labels shape:", y_train.shape)
print("Test labels shape:", y_test.shape)
```

### 7) Embedding preparation

Load Pretrained Model: The GloVe 200-dimensional embeddings are either downloaded or loaded from local storage. These embeddings provide pretrained vector representations for words, capturing semantic relationships.
Embedding Matrix: An embedding matrix is created to map each word in the vocabulary to its GloVe vector. For words not in the GloVe model, the matrix is filled with zeros. This matrix serves as the weight for the embedding layer in the LSTM model.

```
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
```

### 8) Model building and compilation

LSTM Model Construction:
  The model starts with an embedding layer initialized with the pretrained embeddings. The layer is frozen to prevent updates during training.
  Two Bidirectional LSTM layers are added to capture contextual dependencies in both directions.
  Dropout layers are included to reduce overfitting.
  A dense output layer with a sigmoid activation function is used for binary sentiment classification.
Compilation: The model is compiled using the Adam optimizer with a learning rate of 0.0001. Binary cross-entropy is chosen as the loss function, suitable for binary classification tasks. Accuracy is set as the evaluation metric.
```
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
```

### 9) Model training

Data Shuffling: The training data is shuffled randomly to prevent overfitting and improve model generalization.
Training: The model is trained on the shuffled data, with 20% reserved for validation. The training process uses a batch size of 50 and runs for 50 epochs. Validation metrics are tracked to monitor performance during training.
```
# Shuffle the training data
train_data, train_labels = shuffle(x_train_padded, y_train, random_state=42)

# Train the model
lstm_model.fit(train_data, 
               train_labels, 
               validation_split=0.20, 
               batch_size=50, 
               epochs=50)
```

### 10) Custom prediction

Prediction Function:
    The lstm_predict function allows real-time sentiment prediction for input sentences.
    Sentences are cleaned, encoded, and padded similarly to the training process.
    The LSTM model predicts a sentiment score, which is rounded to 0 (negative) or 1 (positive). The result is printed with an appropriate sentiment label.
```
def lstm_predict(sentence: str):
  
  #Predicts the sentiment of a given sentence using the trained LSTM model.
  #Clean and encode the sentence
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
```

# Conclusion
We successfully demonstrated the fundamental tools of NLP and implemented a sentiment analysis tool using a Recurrent Neural Network, Bidirectional LongShortTermMemory model with a pretrained embedding layer(GloVe) to classify text reviews as positive or negative. After initial implementation was complete we tuned the model and adjusted the hyperparameters to achieve a ~95% accuracy. Further impovements can posibly be achieved by:
- varying all the different hyper-parameters for the Recurrent Neural Network.
- Using individual characters and one-shot encoding instead of tokenized words and the embeding layer.
- Plenty more 
