import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

class Issue:
    def __init__(self, data, target):
        self.data = data
        self.target = target

class Issues:
    def __init__(self, issues, target_names = None):
        self.data = [issue.data for issue in issues]
        self.target = [issue.target for issue in issues]
        self.target_names = target_names

    def addIssues(self, issues):
        self.data.extend([issue.data for issue in issues])
        self.target.extend([issue.target for issue in issues])

def getIssues(file):
    with open(file, 'r') as baseIssuesFile:
        baseIssues = json.load(baseIssuesFile)
        issues = Issues([Issue(bug, 0) for bug in baseIssues['bug']])
        issues.addIssues([Issue(feature, 1) for feature in baseIssues['feature']])
        issues.target_names = ['bug', 'feature']
        return issues

issues = getIssues('baseIssues.json')
X_train, X_test, y_train, y_test = train_test_split(issues.data, issues.target, 
    test_size=0.4)

'''
#print(X_test[0], y_test[0])
countVect = CountVectorizer()
X_train_counts = countVect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)
predicted = clf.predict(X_test)
#print(predicted == y_test)
'''
text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])

text_clf = text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)
print(np.mean(predicted == y_test))