from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")

data = []
with open('voice_data.txt') as f:
    for i in f:
        # gender, maks, wmean = i.split()
        gender, wmean = i.split()
        data.append([gender, float(wmean)])
        # data.append([gender, float(maks), float(wmean)])

pred = 1
dataK = [i[pred] for i in data if i[0] == 'K']
dataM = [i[pred] for i in data if i[0] == 'M']

# bins = np.linspace(0, 300, 10)
# plt.hist(dataK, bins, alpha=0.5, label='Kobiety')
# plt.hist(dataM, bins, alpha=0.5, label='Mezczyzni')
# plt.legend()
# plt.show()
# exit()

X = np.array([i[1] for i in data])
X = X.reshape(-1, 1)
y = [i[0] for i in data]


classifier = LogisticRegression()
classifier.fit(X, y)

print(classifier.score(X, y))
print(classifier.predict([[167]]))

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(classifier, X, y)
plt.show()




# test = [[x[0][0], x[1]] for x in bests]
# test_x = [x[0] for x in test]
# test_y = [x[1] for x in test]
# plt.plot(test_x, test_y)
# plt.show()
# exit()
