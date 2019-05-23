import os
from os import listdir
from utils import load_doc
from string import punctuation
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer

def doc_to_clean_lines(doc, vocab):
    clean_lines = list()
    lines = doc.splitlines()

    for line in lines:
        tokens = line.split()
        table = str.maketrans('', '', punctuation)
        tokens = [w.translate(table) for w in tokens]
        tokens = [w for w in tokens if w in vocab]
        clean_lines.append(tokens)

    return clean_lines

def process_docs(directory, vocab, is_training):
    lines = list()
    for filename in listdir(directory):
        if is_training and filename.startswith('cv9'):
            continue
        if not is_training and not filename.startswith('cv9'):
            continue
        path = os.path.join(directory,filename)
        doc = load_doc(path)
        doc_lines = doc_to_clean_lines(doc, vocab)

        lines += doc_lines

    return lines

def train(filename):
    vocab = load_doc(vocab_filename)
    vocab = vocab.split()
    vocab = set(vocab)

    positive_lines = process_docs('txt_sentoken/pos', vocab, True)
    negative_lines = process_docs('txt_sentoken/neg', vocab, True)
    sentences = negative_lines + positive_lines
    print('Total training sentences: %d' % len(sentences))

    model = Word2Vec(sentences, size=100, window=5, workers=8, min_count=1)
    words = list(model.wv.vocab)
    print('Vocabulary size: %d' % len(words))

    filename = 'embedding_word2vec.txt'
    model.wv.save_word2vec_format(filename, binary=False)


if __name__=='__main__':
    tokenizer = Tokenizer()
    vocab_filename = 'vocab.txt'
    train(vocab_filename)