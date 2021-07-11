from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


# load a clean dataset
def load_dataset(filename):
    return load(open(filename, 'rb'))


# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# calculate the maximum document length
def max_length(lines):
    return max([len(s.split()) for s in lines])


# encode a list of lines
def encode_text(tokenizer, lines, length):
    # integer encode
    encoded = tokenizer.texts_to_sequences(lines)
    # pad encoded sequences
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded


# classify a review as negative or positive
def predict_sentiment(review, tokenizer, max_length, model):
    # encode and pad review
    padded = encode_text(tokenizer, review, max_length)
    # predict sentiment
    yhat = model.predict([padded, padded, padded], verbose=0)
    # retrieve predicted percentage and label
    percent_pos = yhat[0, 0]
    if round(percent_pos) == 0:
        return (1 - percent_pos), 'NEGATIVE'
    return percent_pos, 'POSITIVE'


# Load training dataset
train_lines, tran_labels = load_dataset('train.pkl')
# create the tokenizer
tokenizer = create_tokenizer(train_lines)

# load the model
model = load_model('model.h5')
reviews = ["I am not satisfied",
           "worst product I have never bought, bad",
           "I am satisfied",
           "Great product, I love it. Totally worth it!"]

max_length = max_length(train_lines)

for review in reviews:
    percent, sentiment = predict_sentiment(review, tokenizer, max_length, model)
    print("Sentiment: " + sentiment + " Percent: " + str(percent))



