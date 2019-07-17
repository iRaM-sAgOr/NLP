import re
import zipfile
import lxml.etree
from gensim.models import FastText
from gensim.models import Word2Vec


def tokenization_after_preprocessing(input_text):
    # remove parenthesis
    input_text_noparens = re.sub(r'\([^)]*\)', '', input_text)
    # store as list of sentences
    sentences_strings_ted = []
    for line in input_text_noparens.split('\n'):
        m = re.match(r'^(?:(?P<precolon>[^:]{,20}):)?(?P<postcolon>.*)$', line)
        sentences_strings_ted.extend(sent for sent in m.groupdict()['postcolon'].split('.') if sent)
    # store as list of lists of words
    sentences_ted = []
    for sent_str in sentences_strings_ted:
        tokens = re.sub(r"[^a-z0-9]+", " ", sent_str.lower()).split()
        sentences_ted.append(tokens)
    return sentences_ted


def word2vec(sentence_ted, window, min_count, sg, word):
    model_ted = Word2Vec(sentences=sentence_ted, size=100, window=window, min_count=min_count, workers=4, sg=sg)
    print(model_ted.wv.most_similar(word))


def fasttext(sentence_ted, window, min_count, sg, word):
    model_ted = FastText(sentences=sentence_ted, size=100, window=window, min_count=min_count, workers=4, sg=sg)
    print(model_ted.wv.most_similar(word))


if __name__ == '__main__':
    with zipfile.ZipFile('ted_en-20160408.zip', 'r') as z:
        doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))
        input_text = '\n'.join(doc.xpath('//content/text()'))
    word_input = (input("You want to test any word = "))
    sentences_ted = tokenization_after_preprocessing(input_text)
    method = int(input("Please enter the method of embedding Word2Vec=0 and Fasttext = 1!"))
    window = int(input("Please enter the window size of your corpus!"))
    min_count = int(input("Please enter the minimum count of any word in corpus!"))
    sq = int(input("Please enter the method of CBOW=0 or Skip-gram=1!"))
    if method == 0:
        word2vec(sentences_ted, window, min_count, sq, word_input)
    else:
        fasttext(sentences_ted, window, min_count, sq, word_input)
