import nltk
# nltk.download()
# nltk.download('punkt')
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import re

with open("paragraph.txt", "r") as file:
    paragraph = file.read()

sentences = nltk.sent_tokenize(paragraph)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
corpus = []

#Stemming
# for i in range(len(sentences)):
#     words = nltk.word_tokenize(sentences[i])
#     words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
#     sentences[i] = ' '.join(words) 


#Lemmatization
# for i in range(len(sentences)):
#     words = nltk.word_tokenize(sentences[i])
#     words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
#     sentences[i] = ' '.join(words)    

for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
# from sklearn.feature_extraction.text import CountVectorizer
# cv = CountVectorizer(max_features = 1500)
# X = cv.fit_transform(corpus).toarray()

# Creating the TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()
print(X)