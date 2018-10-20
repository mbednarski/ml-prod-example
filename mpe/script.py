#from sklearn.feature_extraction import 
from pathlib import Path
import numpy as np
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import scikitplot as skplt



pos_directory = Path('data/raw/aclImdb/train/pos')
neg_directory = Path('data/raw/aclImdb/train/neg')

pos_examples = []
neg_examples = []

for fname in list(pos_directory.iterdir())[:1000]:
    pos_examples.append(fname.read_text())
for fname in list(neg_directory.iterdir())[:1000]:
    neg_examples.append(fname.read_text())


def normalize(s :str) -> str:
    s = s.replace('<br />', '')
    s = re.sub('[^a-zA-Z]', ' ', s)
    s = re.sub(r'\s\s+', ' ', s)
    s = s.lower()

    return s

pos_examples = [normalize(x) for x in pos_examples]
neg_examples = [normalize(x) for x in neg_examples]

y = np.concatenate((np.ones(len(pos_examples)), np.zeros(len(neg_examples))))

vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,3), max_features=2000)
X = vectorizer.fit_transform((pos_examples + neg_examples)).toarray()

clf = GaussianNB()
clf.fit(X, y)
y_pred = clf.predict(X)
y_prob = clf.predict_proba(X)


print(classification_report(y, y_pred))

# skplt.metrics.plot_confusion_matrix(y, y_pred)
# plt.savefig('cm.png')

# plt.clf()
skplt.metrics.plot_lift_curve(y, y_prob)
plt.savefig('cal.png')

