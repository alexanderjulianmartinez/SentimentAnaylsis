def load_doc(filename):
    """ Load document into memory. """
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text
