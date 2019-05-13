from nltk.corpus import stopwords
import string

def load_doc(filename):
    """ Load document into memory. """
    file = open(filename, 'r')
    text = file.read()

    file.close()

    return text

def clean_doc(doc):
    """ Convert document into cleaned tokens """

    # Split into tokens by white space
    tokens = doc.split()

    # Remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate() for w in tokens]

    # Remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]

    # Filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]

    # Filter out short tokens
    tokens = [word for word in tokens if len(word)>1]

    return tokens


if __name__=="__main__":
    filename = ''
    text = load_doc(filename)
    tokens = clean_doc(text)
    print(tokens)