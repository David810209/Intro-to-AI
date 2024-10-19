import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import SnowballStemmer
import re

def remove_stopwords(text: str) -> str:
    '''
    E.g.,
        text: 'Here is a dog.'
        preprocessed_text: 'Here dog.'
    '''
    stop_word_list = stopwords.words('english')
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    preprocessed_text = ' '.join(filtered_tokens)
    return preprocessed_text


def preprocessing_function(text: str) -> str:
    preprocessed_text = remove_stopwords(text)
    
    # TO-DO 0: Other preprocessing function attemption
    # Begin your code 
    preprocessed_text = preprocessed_text.replace("<br />", " ").lower()
    preprocessed_text = re.sub(r'[^\w\s]','',preprocessed_text) 
    words = nltk.word_tokenize(preprocessed_text)
    stemmer = nltk.SnowballStemmer(language='english')
    words = [stemmer.stem(word) for word in words]
    preprocessed_text = ' '.join(words)
    # End your code
    
    return preprocessed_text

# test='''
# I expected to see a BOOGEYMAN similar movie, and instead i watched a drama with some meaningless thriller spots.
# '''
# print(remove_stopwords(test))
# print(preprocessing_function(test))