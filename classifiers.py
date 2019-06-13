import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics

def generateConfusionMatrix(y_val, y_pred, classes):
    cm = metrics.confusion_matrix(y_val, y_pred)

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes, yticklabels=classes,
        title="Matriz de Confusão",
        ylabel="Real",
        xlabel="Predito")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2. else "black")
    fig.tight_layout()
    plt.show()

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
countVect = CountVectorizer()
X = countVect.fit_transform(issues.data)
tfidf_transformer = TfidfTransformer()
X = tfidf_transformer.fit_transform(X)
naiveBayes = MultinomialNB()
randomForest = RandomForestClassifier()
svmClassifier = SVC(kernel='linear', gamma='auto')
decisionTree = DecisionTreeClassifier(max_depth = 8)
kNeighborsClassifier = KNeighborsClassifier(n_neighbors=50)
gTB = GradientBoostingClassifier(n_estimators=30)
scores = cross_val_score(svmClassifier, X, issues.target, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(svmClassifier, X, issues.target, cv=5)
generateConfusionMatrix(issues.target, predicted, issues.target_names)
'''
scores = cross_val_score(gTB, X, issues.target, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scosigmoidres.mean(), scores.std() * 2))
scores = cross_val_score(decisionTree, X, issues.target, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(kNeighborsClassifier, X, issues.target, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
scores = cross_val_score(naiveBayes, X, issues.target, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
X_train, X_test, y_train, y_test = train_test_split(issues.data, issues.target, 
    test_size=0.4)

#print(X_test[0], y_test[0])
countVect = CountVectorizer()
X_train_counts = countVect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)
predicted = clf.predict(X_test)
#print(predicted == y_test)
'''
'''
text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])

text_clf = text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)
print(np.mean(predicted == y_test))
print(predicted[:10])
print(X_test[:10])
text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])

text_clf = text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)
print(predicted[:10])
'''