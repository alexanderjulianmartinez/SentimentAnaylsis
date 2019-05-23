import os
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
from utils import load_doc
import string

def clean_doc(doc):
    """ Convert document into cleaned tokens """

    # Split into tokens by white space
    tokens = doc.split()

    # Remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]

    # Remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]

    # Filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]

    # Filter out short tokens
    tokens = [word for word in tokens if len(word)>1]

    return tokens

def add_doc_to_vocab(filename, vocab):
    # Load doc and add to vocab
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    vocab.update(tokens)

def process_docs(directory, vocab, is_training):
    # Load all docs in a directory
    for filename in listdir(directory):
        if is_training and filename.startswith('cv9'):
            continue
        if not is_training and not filename.startswith('cv9'):
            continue

        # Create the full path of the file to open
        path = os.path.join(directory, filename)

        add_doc_to_vocab(path, vocab)

def save_list(lines, filename):
    # Save list to file
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


if __name__=="__main__":
    vocab = Counter()
    process_docs('txt_sentoken/neg', vocab, True)
    process_docs('txt_sentoken/pos', vocab, True)
    print(len(vocab))
    print(vocab.most_common(50))
    save_list(vocab, 'vocab.txt')