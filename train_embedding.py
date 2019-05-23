import os
import string
import numpy as np
from os import listdir
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.preprocessing.sequence import pad_sequences
from utils import load_doc

def clean_doc(doc, vocab):
    # Turn doc into clean tokens
    tokens = doc.split()

    table = str.maketrans('','', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [w for w in tokens if w in vocab]
    tokens = ' '.join(tokens)

    return tokens

def process_docs(directory, vocab, is_training):
    documents = list()

    for filename in listdir(directory):
        if is_training and filename.startswith('cv9'):
            continue
        if not is_training and not filename.startswith('cv9'):
            continue

        path = os.path.join(directory, filename)
        doc = load_doc(path)
        tokens = clean_doc(doc, vocab)

        documents.append(tokens)

    return documents

def train(vocab_filename):
    vocab = load_doc(vocab_filename)
    vocab = vocab.split()
    vocab = set(vocab)

    positive_docs = process_docs('txt_sentoken/pos', vocab, True)
    negative_docs = process_docs('txt_sentoken/neg', vocab, True)
    train_docs = positive_docs + negative_docs

    tokenizer.fit_on_texts(train_docs)
    encoded_docs = tokenizer.texts_to_sequences(train_docs)

    max_length = max([len(s.split()) for s in train_docs])
    X_train = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    y_train = np.array([0 for _ in range(900)] + [1 for _ in range(900)])

    positive_docs = process_docs('txt_sentoken/pos', vocab, False)
    negative_docs = process_docs('txt_sentoken/neg', vocab, False)
    test_docs = negative_docs + positive_docs


    encoded_test_docs = tokenizer.texts_to_sequences(test_docs)
    X_test = pad_sequences(encoded_test_docs, maxlen=max_length, padding='post')
    y_test = np.array([0 for _ in range(100)] + [1 for _ in range(100)])

    vocab_size = len(tokenizer.word_index) + 1
    model = create_model(vocab_size, max_length)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, verbose=2)

    print((X_test.shape, y_test.shape))
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test Accuracy: %f' % (acc*100))


def create_model(vocab_size, max_length):
    # Initialize Model
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model


if __name__=='__main__':
    tokenizer = Tokenizer()
    vocab_filename = 'vocab.txt'
    train(vocab_filename)
