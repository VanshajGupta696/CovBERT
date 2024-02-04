#importing libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Text cleaning
import re
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Data preprocessing
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from langdetect import detect, LangDetectException
import contractions
from nltk.tokenize import word_tokenize

# Naive Bayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

# PyTorch LSTM
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# Tokenization for LSTM
from collections import Counter
from gensim.models import Word2Vec

# Transformers library for BERT
import transformers
from transformers import BertModel
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#custom function for plotting the confusion matrix

def conf_matrix(y, y_pred, title, labels):
    fig, ax =plt.subplots(figsize=(5,5))
    ax=sns.heatmap(confusion_matrix(y, y_pred), annot=True, cmap="Purples", fmt='g', cbar=False, annot_kws={"size":30})
    plt.title(title, fontsize=25)
    ax.xaxis.set_ticklabels(labels, fontsize=16)
    ax.yaxis.set_ticklabels(labels, fontsize=14.5)
    ax.set_ylabel('Test', fontsize=25)
    ax.set_xlabel('Predicted', fontsize=25)
    plt.show()

import time

# Set seed for reproducibility
import random
seed_value = 2042
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

# Set style for plots
sns.set_style("whitegrid")
sns.despine()
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc("axes", labelweight="bold", labelsize="large", titleweight="bold", titlepad=10)

# Define stop words for text cleaning
stop_words = set(stopwords.words('english'))

# Initialize lemmatizer for text cleaning
lemmatizer = WordNetLemmatizer()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the data
sentiment_data = pd.read_csv("/content/drive/MyDrive/finalSentimentdata2.csv")
sentiment_data.head()

sentiment_data = sentiment_data.drop(columns=['Unnamed: 0','length'])

sentiment_data.head()

sentiment_data.duplicated().sum()

#remove duplicate tweets

sentiment_data = sentiment_data[~sentiment_data.duplicated()]

"""## Defining functions to clean the data:"""

# Clean emojis from text
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Remove punctuations, stopwords, links, mentions and new line characters
def strip_all_entities(text):
    text = re.sub(r'\r|\n', ' ', text.lower())  # Replace newline and carriage return with space, and convert to lowercase
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)  # Remove links and mentions
    text = re.sub(r'[^\x00-\x7f]', '', text)  # Remove non-ASCII characters
    banned_list = string.punctuation
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Clean hashtags at the end of the sentence, and keep those in the middle of the sentence by removing just the # symbol
def clean_hashtags(tweet):
    # Remove hashtags at the end of the sentence
    new_tweet = re.sub(r'(\s+#[\w-]+)+\s*$', '', tweet).strip()

    # Remove the # symbol from hashtags in the middle of the sentence
    new_tweet = re.sub(r'#([\w-]+)', r'\1', new_tweet).strip()

    return new_tweet

# Filter special characters such as & and $ present in some words
def filter_chars(text):
    return ' '.join('' if ('$' in word) or ('&' in word) else word for word in text.split())

# Remove multiple spaces
def remove_mult_spaces(text):
    return re.sub(r"\s\s+", " ", text)

# Function to check if the text is in English, and return an empty string if it's not
def filter_non_english(text):
    try:
        lang = detect(text)
    except LangDetectException:
        lang = "unknown"
    return text if lang == "en" else ""

# Expand contractions
def expand_contractions(text):
    return contractions.fix(text)

# Remove numbers
def remove_numbers(text):
    return re.sub(r'\d+', '', text)

# Lemmatize words
def lemmatize(text):
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

# Remove short words
def remove_short_words(text, min_len=2):
    words = text.split()
    long_words = [word for word in words if len(word) >= min_len]
    return ' '.join(long_words)

# Replace elongated words with their base form
def replace_elongated_words(text):
    regex_pattern = r'\b(\w+)((\w)\3{2,})(\w*)\b'
    return re.sub(regex_pattern, r'\1\3\4', text)

# Remove repeated punctuation
def remove_repeated_punctuation(text):
    return re.sub(r'[\?\.\!]+(?=[\?\.\!])', '', text)

# Remove extra whitespace
def remove_extra_whitespace(text):
    return ' '.join(text.split())

def remove_url_shorteners(text):
    return re.sub(r'(?:http[s]?://)?(?:www\.)?(?:bit\.ly|goo\.gl|t\.co|tinyurl\.com|tr\.im|is\.gd|cli\.gs|u\.nu|url\.ie|tiny\.cc|alturl\.com|ow\.ly|bit\.do|adoro\.to)\S+', '', text)

# Remove spaces at the beginning and end of the tweet
def remove_spaces_tweets(tweet):
    return tweet.strip()

# Remove short tweets
def remove_short_tweets(tweet, min_words=3):
    words = tweet.split()
    return tweet if len(words) >= min_words else ""

# Function to call all the cleaning functions in the correct order
def clean_tweet(tweet):
    tweet = remove_emoji(tweet)
    tweet = expand_contractions(tweet)
    tweet = filter_non_english(tweet)
    tweet = strip_all_entities(tweet)
    tweet = clean_hashtags(tweet)
    tweet = filter_chars(tweet)
    tweet = remove_mult_spaces(tweet)
    tweet = remove_numbers(tweet)
    tweet = lemmatize(tweet)
    tweet = remove_short_words(tweet)
    tweet = replace_elongated_words(tweet)
    tweet = remove_repeated_punctuation(tweet)
    tweet = remove_extra_whitespace(tweet)
    tweet = remove_url_shorteners(tweet)
    tweet = remove_spaces_tweets(tweet)
    tweet = remove_short_tweets(tweet)
    tweet = ' '.join(tweet.split())  # Remove multiple spaces between words
    return tweet

sentiment_data['text_clean'] = [clean_tweet(tweet) for tweet in sentiment_data['text']]

sentiment_data.head()

print(f'There are around {int(sentiment_data["text_clean"].duplicated().sum())} duplicated tweets, we will remove them.')

sentiment_data.drop_duplicates("text_clean", inplace=True)

#checking if the classes balanced

sentiment_data.sentiment.value_counts()

"""## Plotting the word cloud for the 50 most common words in the tweets/posts"""

from wordcloud import WordCloud, STOPWORDS

word_cloud = WordCloud(
                    background_color='white',
                    stopwords=set(STOPWORDS),
                    max_words=50,
                    max_font_size=40,
                    scale=5,
                    random_state=1).generate(str(sentiment_data['text']))
fig = plt.figure(1, figsize=(10,10))
plt.axis('off')
fig.suptitle('Word Cloud for top 50 common words', fontsize=20)
fig.subplots_adjust(top=2.3)
plt.imshow(word_cloud)
plt.show()

"""## Label Encoding of the sentiments"""

from sklearn.preprocessing import LabelEncoder
lb= LabelEncoder()

lb.fit(sentiment_data['sentiment'])

sentiments = list(lb.classes_)

sentiments

"""Therefore, the encoding is:
anger - 0; fear - 1; joy - 2; sad - 3;
"""

sentiment_data['sentiment']= lb.fit_transform(sentiment_data['sentiment'])

sentiment_data.head()

#splitting the data into train and test set

X = sentiment_data['text_clean']
y = sentiment_data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed_value)

#splitting the training set into train and validation sets

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=seed_value)

#oversampling to reduce any anomalies caused by unbalanced classes

ros = RandomOverSampler()
X_train, y_train = ros.fit_resample(np.array(X_train).reshape(-1, 1), np.array(y_train).reshape(-1, 1));
train_os = pd.DataFrame(list(zip([x[0] for x in X_train], y_train)), columns = ['text_clean', 'sentiment']);

X_train = train_os['text_clean'].values
y_train = train_os['sentiment'].values

#checking if the classes are balanced now

(unique, counts) = np.unique(y_train, return_counts=True)
np.asarray((unique, counts)).T

"""## Naive Bayes Implementation:"""

clf = CountVectorizer()
X_train_cv =  clf.fit_transform(X_train)
X_test_cv = clf.transform(X_test)

#TF-IFD transformation to give weights to the words based on their frequency (rare words will be given higher weight)

tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_cv)
X_train_tf = tf_transformer.transform(X_train_cv)
X_test_tf = tf_transformer.transform(X_test_cv)

#instantiating an NB model

nb_clf = MultinomialNB()

nb_clf.fit(X_train_tf, y_train)

nb_pred = nb_clf.predict(X_test_tf)

print('Classification Report for Naive Bayes:\n',classification_report(nb_pred, y_test))

conf_matrix(y_test,nb_pred,'Naive Bayes Sentiment Analysis\nConfusion Matrix', sentiments)

"""## SVM Implementation:"""

from sklearn.svm import SVC

svc = SVC()

svc.fit(X_train_tf, y_train)

svc_pred = svc.predict(X_test_tf)

print('Classification Report for SVM:\n',classification_report(svc_pred, y_test))

conf_matrix(y_test,svc_pred,'SVM Sentiment Analysis\nConfusion Matrix', sentiments)

"""## Logistic Regression Implementation:"""

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train_tf, y_train)

logreg_pred = logreg.predict(X_test_tf)

print('Classification Report for Logistic Regression:\n',classification_report(logreg_pred, y_test))

conf_matrix(y_test,logreg_pred,'Logistic Regression Sentiment Analysis\nConfusion Matrix', sentiments)

"""## LSTM Implementation:"""

sentiment_data_lstm = sentiment_data.copy()

sentiment_data_lstm.head()

def Tokenize(column, seq_len):
    ##Create vocabulary of words from column
    corpus = [word for text in column for word in text.split()]
    count_words = Counter(corpus)
    sorted_words = count_words.most_common()
    vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}

    ##Tokenize the columns text using the vocabulary
    text_int = []
    for text in column:
        r = [vocab_to_int[word] for word in text.split()]
        text_int.append(r)
    ##Add padding to tokens
    features = np.zeros((len(text_int), seq_len), dtype = int)
    for i, review in enumerate(text_int):
        if len(review) <= seq_len:
            zeros = list(np.zeros(seq_len - len(review)))
            new = zeros + review
        else:
            new = review[: seq_len]
        features[i, :] = np.array(new)

    return sorted_words, features

sentiment_data_lstm['text_len'] = [len(text.split()) for text in sentiment_data_lstm.text_clean]

max_len = np.max(sentiment_data_lstm['text_len'])
max_len

vocabulary, tokenized_column = Tokenize(sentiment_data_lstm["text_clean"], max_len)

X_lstm = tokenized_column

y_lstm = pd.get_dummies(sentiment_data_lstm['sentiment'])
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=1)
X_train_lstm, X_val_lstm, y_train_lstm, y_val_lstm = train_test_split(X_train_lstm, y_train_lstm, test_size=0.25, random_state=1)

print('Train Set ->', X_train_lstm.shape, y_train_lstm.shape)
print('Validation Set ->', X_val_lstm.shape, y_val_lstm.shape)
print('Test Set ->', X_test_lstm.shape, y_test_lstm.shape)

import keras.backend as K

def f1_score(precision, recall):
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
from keras.metrics import Precision, Recall
from keras.optimizers.legacy import SGD
from keras.optimizers import RMSprop, Adam
from keras import datasets

from keras.callbacks import LearningRateScheduler
from keras.callbacks import History

from keras import losses

vocab_size = 5000
embedding_size = 64
epochs=20
learning_rate = 0.1
decay_rate = learning_rate / epochs
momentum = 0.8

sgd = SGD(learning_rate=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
# Build model
model= Sequential()
model.add(Embedding(vocab_size, embedding_size, input_length=max_len))
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

import tensorflow as tf
tf.keras.utils.plot_model(model, show_shapes=True)

# Compile model
optimizer = Adam(learning_rate=0.001)

model.compile(loss='categorical_crossentropy', optimizer=optimizer,
               metrics=['accuracy', Precision(), Recall()])

from keras.callbacks import ModelCheckpoint

checkpoint_filepath = 'best_model_lstm.h5'

model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath,
                                   monitor='val_accuracy',
                                   verbose=1,
                                   save_best_only=True,
                                   mode='max')

# Train model
batch_size = 64
history = model.fit(X_train_lstm, y_train_lstm,
                    validation_data=(X_val_lstm, y_val_lstm),
                    batch_size=batch_size, epochs=epochs, verbose=1,
                    callbacks=[model_checkpoint])

from keras.models import load_model

best_model = load_model(checkpoint_filepath)

# Evaluate model on the test set
loss, accuracy, precision, recall = best_model.evaluate(X_test_lstm, y_test_lstm, verbose=0)
# Print metrics
print('')
print('Accuracy  : {:.4f}'.format(accuracy))
print('Precision : {:.4f}'.format(precision))
print('Recall    : {:.4f}'.format(recall))
print('F1 Score  : {:.4f}'.format(f1_score(precision, recall)))

import matplotlib.pyplot as plt

def plot_training_hist(history):

    plt.figure(figsize=(10,6))

    plt.plot(history.history['accuracy'], 'b-', label='train accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], 'g-', label='validation accuracy', linewidth=2)

    plt.title('Training history')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.ylim([0, 1])
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.show()

plot_training_hist(history)

"""## BERT Implementation:"""

import torch
import torch.nn as nn

!pip install transformers

import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device

BERT_PRE_TRAINED_MODEL = "/content/drive/MyDrive/bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(BERT_PRE_TRAINED_MODEL)

token_lens = []
for txt in sentiment_data.text:

    tokens = tokenizer.encode(txt, max_length=512)
    token_lens.append(len(tokens))

sns.distplot(token_lens)
plt.xlim([0, 256]);
plt.xlabel('Token count')

MAX_LEN=100

sentiment_data.head()

sentiment_data.head()

class Covid19Tweet(Dataset):

    def __init__(self, tweets, sentiment, tokenizer, max_len):


        self.tweets = tweets
        self.sentiment = sentiment
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.tweets)
    def __getitem__(self, item):

        tweets = str(self.tweets[item])
        sentiment = self.sentiment[item]
        encoding = self.tokenizer.encode_plus(
        tweets,
        add_special_tokens=True,
        max_length=self.max_len,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt')
        return {
        'tweet_text': tweets,
         'input_ids': encoding['input_ids'].flatten(),
         'attention_mask': encoding['attention_mask'].flatten(),
         'sentiments': torch.tensor(sentiment, dtype=torch.long)
          }

from sklearn.model_selection import train_test_split

train, val = train_test_split(
  sentiment_data,
  test_size=0.1,
  random_state=RANDOM_SEED
)

def create_data_loader(data, tokenizer, max_len, batch_size):

    ds = Covid19Tweet(tweets=data.text.to_numpy(),
    sentiment=data.sentiment.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len)
    return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4)
BATCH_SIZE = 32
train_data_loader = create_data_loader(train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(val, tokenizer, MAX_LEN, BATCH_SIZE)

df = next(iter(train_data_loader))
df.keys()

bert_model = BertModel.from_pretrained(BERT_PRE_TRAINED_MODEL)

class SentimentClassifier(nn.Module):

    def __init__(self, n_classes):

        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_PRE_TRAINED_MODEL, return_dict=False)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    def forward(self, input_ids, attention_mask):

        _, pooled_output = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask
    )
        output = self.drop(pooled_output)
        return self.out(output)

n_classes= 4

model = SentimentClassifier(n_classes)
model = model.to(device)

model

input_ids = df['input_ids'].to(device)
attention_mask = df['attention_mask'].to(device)

import torch.nn.functional as F

output = model(input_ids, attention_mask)

F.softmax(output, dim=1)

"""Displaying the model structure:"""

model.parameters

EPOCHS = 10
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)

def train_epoch(model,data_loader,loss_fn,optimizer,device,scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    for d in data_loader:

        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["sentiments"].to(device)
        outputs = model(input_ids=input_ids,attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):

    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["sentiments"].to(device)
            outputs = model(input_ids=input_ids,attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
        return correct_predictions.double() / n_examples, np.mean(losses)

from collections import defaultdict

history = defaultdict(list)
best_accuracy = 0
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    train_acc, train_loss = train_epoch(model,train_data_loader,loss_fn,optimizer,device,scheduler,len(train))
    print(f'Train loss {train_loss} accuracy {train_acc}')
    val_acc, val_loss = eval_model(model,val_data_loader,loss_fn,device,len(val))
    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()
    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)
    if val_acc > best_accuracy:

        torch.save(model.state_dict(), 'best_model_state.bin')
        best_accuracy = val_acc

train_acc = [tensor.cpu().numpy() for tensor in history['train_acc']]
val_acc = [tensor.cpu().numpy() for tensor in history['val_acc']]

plt.plot(train_acc, label='train accuracy')
plt.plot(val_acc, label='validation accuracy')
plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1]);

"""## Testing the model on the Sentiment Analysis dataset:"""

sentiment_data.head()

sample = sentiment_data.sample(1)

sample

sample_tweet = sample['text'].values[0]
sample_tweet

encoded_review = tokenizer.encode_plus(sample_tweet,max_length=MAX_LEN,add_special_tokens=True,
                                           return_token_type_ids=False,pad_to_max_length=True,return_attention_mask=True,
                                           return_tensors='pt')

input_ids = encoded_review['input_ids'].to(device)
attention_mask = encoded_review['attention_mask'].to(device)
output = model(input_ids, attention_mask)
_, prediction = torch.max(output, dim=1)
print('Review text :{}'.format(sample_tweet))
print('Sentiment :{}'.format(sentiments[prediction]))

"""## Testing the model on the GeoLocation Dataset:"""

geo_data.head()

geo_data_sent = geo_data.copy()

geo_data_sent.head()

geo_data_sent = geo_data_sent.drop(columns=['user_name', 'user_location', 'date', 'location', 'country'])

geo_data_sent.head()

sample_geo = geo_data_sent.sample(1)
sample_geo

sample_geo['text'] = sample_geo['text'].apply(lambda x: clean_tweet(x))
sample_geo

sample_geo['text'] = sample_geo['text'].apply(lambda x: remove_emoji(x))

sample_tweet_geo = sample_geo['text'].values[0]
sample_tweet_geo

encoded_review = tokenizer.encode_plus(sample_tweet_geo,max_length=MAX_LEN,add_special_tokens=True,
                                           return_token_type_ids=False,pad_to_max_length=True,return_attention_mask=True,
                                           return_tensors='pt')

input_ids = encoded_review['input_ids'].to(device)
attention_mask = encoded_review['attention_mask'].to(device)
output = model(input_ids, attention_mask)
_, prediction = torch.max(output, dim=1)
print('Review text :{}'.format(sample_tweet_geo))
print('Sentiment :{}'.format(sentiments[prediction]))

"""## ____ ____ ____ ____ ____ ____ ____ ____ ____ ____ ____ ____

## Custom Fine Tuned BERT Model
"""

sentiment_data = pd.read_csv("/content/drive/MyDrive/finalSentimentdata2.csv")

sentiment_data = sentiment_data.drop(columns=['Unnamed: 0'])

from sklearn.preprocessing import LabelEncoder
lb= LabelEncoder()
lb.fit(sentiment_data['sentiment'])

sentiment_data['sentiment']= lb.fit_transform(sentiment_data['sentiment'])

# Define stop words for text cleaning
stop_words = set(stopwords.words('english'))

# Initialize lemmatizer for text cleaning
lemmatizer = WordNetLemmatizer()

# Clean emojis from text
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Remove punctuations, stopwords, links, mentions and new line characters
def strip_all_entities(text):
    text = re.sub(r'\r|\n', ' ', text.lower())  # Replace newline and carriage return with space, and convert to lowercase
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)  # Remove links and mentions
    text = re.sub(r'[^\x00-\x7f]', '', text)  # Remove non-ASCII characters
    banned_list = string.punctuation
    table = str.maketrans('', '', banned_list)
    text = text.translate(table)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Clean hashtags at the end of the sentence, and keep those in the middle of the sentence by removing just the # symbol
def clean_hashtags(tweet):
    # Remove hashtags at the end of the sentence
    new_tweet = re.sub(r'(\s+#[\w-]+)+\s*$', '', tweet).strip()

    # Remove the # symbol from hashtags in the middle of the sentence
    new_tweet = re.sub(r'#([\w-]+)', r'\1', new_tweet).strip()

    return new_tweet

# Filter special characters such as & and $ present in some words
def filter_chars(text):
    return ' '.join('' if ('$' in word) or ('&' in word) else word for word in text.split())

# Remove multiple spaces
def remove_mult_spaces(text):
    return re.sub(r"\s\s+", " ", text)

# Expand contractions
def expand_contractions(text):
    return contractions.fix(text)

# Remove numbers
def remove_numbers(text):
    return re.sub(r'\d+', '', text)

# Lemmatize words
def lemmatize(text):
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

# Remove short words
def remove_short_words(text, min_len=2):
    words = text.split()
    long_words = [word for word in words if len(word) >= min_len]
    return ' '.join(long_words)

# Replace elongated words with their base form
def replace_elongated_words(text):
    regex_pattern = r'\b(\w+)((\w)\3{2,})(\w*)\b'
    return re.sub(regex_pattern, r'\1\3\4', text)

# Remove repeated punctuation
def remove_repeated_punctuation(text):
    return re.sub(r'[\?\.\!]+(?=[\?\.\!])', '', text)

# Remove extra whitespace
def remove_extra_whitespace(text):
    return ' '.join(text.split())

def remove_url_shorteners(text):
    return re.sub(r'(?:http[s]?://)?(?:www\.)?(?:bit\.ly|goo\.gl|t\.co|tinyurl\.com|tr\.im|is\.gd|cli\.gs|u\.nu|url\.ie|tiny\.cc|alturl\.com|ow\.ly|bit\.do|adoro\.to)\S+', '', text)

# Remove spaces at the beginning and end of the tweet
def remove_spaces_tweets(tweet):
    return tweet.strip()

# Remove short tweets
def remove_short_tweets(tweet, min_words=3):
    words = tweet.split()
    return tweet if len(words) >= min_words else ""

# Function to call all the cleaning functions in the correct order
def clean_tweet(tweet):
    tweet = remove_emoji(tweet)
    tweet = expand_contractions(tweet)
    tweet = strip_all_entities(tweet)
    tweet = clean_hashtags(tweet)
    tweet = filter_chars(tweet)
    tweet = remove_mult_spaces(tweet)
    tweet = remove_numbers(tweet)
    tweet = lemmatize(tweet)
    tweet = remove_short_words(tweet)
    tweet = replace_elongated_words(tweet)
    tweet = remove_repeated_punctuation(tweet)
    tweet = remove_extra_whitespace(tweet)
    tweet = remove_url_shorteners(tweet)
    tweet = remove_spaces_tweets(tweet)
    tweet = remove_short_tweets(tweet)
    tweet = ' '.join(tweet.split())  # Remove multiple spaces between words
    return tweet

sentiment_data['text_clean'] = [clean_tweet(tweet) for tweet in sentiment_data['text']]

sentiment_data.head()

import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader

def tokenize_tweets(tweets):
    # Simple tokenization: split by spaces and remove non-alphabetic characters
    tokens = []
    for tweet in tweets:
        # Basic preprocessing to split words and remove non-alphabetic characters
        words = re.findall(r'\b\w+\b', tweet.lower())
        tokens.extend(words)
    return tokens

tweets = sentiment_data["text_clean"]
tweets = [tweet for tweet in tweets if tweet is not None]
tokens = tokenize_tweets(tweets)

from collections import Counter

token_counts = Counter(tokens)

new_tokens = [token for token, count in token_counts.items() if count >= 2 and not token in tokenizer.vocab]
print(f"Number of new tokens to add: {len(new_tokens)}")

# Add new tokens to the tokenizer
tokenizer.add_tokens(new_tokens)

token_lens = []
for txt in sentiment_data.text:

    tokens = tokenizer.encode(txt, max_length=512)
    token_lens.append(len(tokens))

train, val = train_test_split(
  sentiment_data,
  test_size=0.1,
  random_state=RANDOM_SEED
)

train.shape,val.shape

class Covid19Tweet(Dataset):

    def __init__(self, tweets, sentiment, tokenizer, max_len):


        self.tweets = tweets
        self.sentiment = sentiment
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.tweets)
    def __getitem__(self, item):

        tweets = str(self.tweets[item])
        sentiment = self.sentiment[item]
        encoding = self.tokenizer.encode_plus(
        tweets,
        add_special_tokens=True,
        max_length=self.max_len,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt')
        return {
        'tweet_text': tweets,
         'input_ids': encoding['input_ids'].flatten(),
         'attention_mask': encoding['attention_mask'].flatten(),
         'sentiments': torch.tensor(sentiment, dtype=torch.long)
          }

def create_data_loader(data, tokenizer, max_len, batch_size):

    ds = Covid19Tweet(tweets=data.text.to_numpy(),
    sentiment=data.sentiment.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len)
    return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4)
BATCH_SIZE = 32
train_data_loader = create_data_loader(train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(val, tokenizer, MAX_LEN, BATCH_SIZE)

df = next(iter(train_data_loader))
df.keys()

class SentimentClassifierCovbert(nn.Module):

    def __init__(self, n_classes):

        super(SentimentClassifierCovbert, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_PRE_TRAINED_MODEL, return_dict=False)
        self.bert.resize_token_embeddings(len(tokenizer))
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    def forward(self, input_ids, attention_mask):

        _, pooled_output = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask
    )
        output = self.drop(pooled_output)
        return self.out(output)

model_covbert = SentimentClassifierCovbert(n_classes)
model_covbert = model_covbert.to(device)

input_ids = df['input_ids'].to(device)
attention_mask = df['attention_mask'].to(device)

EPOCHS = 10
optimizer = AdamW(model_covbert.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)

def train_epoch(model,data_loader,loss_fn,optimizer,device,scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    for d in data_loader:

        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["sentiments"].to(device)
        outputs = model(input_ids=input_ids,attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):

    model = model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["sentiments"].to(device)
            outputs = model(input_ids=input_ids,attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
        return correct_predictions.double() / n_examples, np.mean(losses)

from collections import defaultdict

history = defaultdict(list)
best_accuracy = 0
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)
    train_acc, train_loss = train_epoch(model_covbert,train_data_loader,loss_fn,optimizer,device,scheduler,len(train))
    print(f'Train loss {train_loss} accuracy {train_acc}')
    val_acc, val_loss = eval_model(model_covbert,val_data_loader,loss_fn,device,len(val))
    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()
    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)
    if val_acc > best_accuracy:

        torch.save(model_covbert.state_dict(), 'best_model_state.bin')
        best_accuracy = val_acc

import matplotlib.pyplot as plt

train_acc = [tensor.cpu().numpy() for tensor in history['train_acc']]
val_acc = [tensor.cpu().numpy() for tensor in history['val_acc']]

plt.plot(train_acc, label='train accuracy')
plt.plot(val_acc, label='validation accuracy')
plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1]);
