# -*- coding: utf-8 -*-
"""Assignment 2 SVM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UxC6gqjdxtzlmsWS5bOUxwOcnxtZFtDc
"""

from sklearn.datasets import fetch_lfw_people
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns; sns.set()

faces = fetch_lfw_people(min_faces_per_person=60)

print(faces.target)

print(faces.target_names)

print(faces.images.shape)

fig, ax = plt.subplots(3, 5)

for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[],
            xlabel=faces.target_names[faces.target[i]])
plt.show()

from sklearn.svm import SVC

from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline

pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)

print(model)

from sklearn.model_selection import train_test_split

"""**Split the data into a training and testing set.**"""

Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target,
                                                random_state=42)

"""**Use grid search cross-validation to select our parameters**"""

from sklearn.model_selection import GridSearchCV

param_grid = {'svc__C': [1, 5, 10,50],
              'svc__gamma': [0.0001, 0.0005, 0.001,0.005]}
grid = GridSearchCV(model, param_grid=param_grid,cv=5)
grid.fit(Xtrain,ytrain)
print(grid.best_params_)

model = grid.best_estimator_
yfit = model.predict(Xtest)
print(yfit.shape)

"""**4x6 subplots of images using names as label with color black for correct instances and red for incorrect instances**"""

fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape(62, 47), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],
                   color='black' if yfit[i] == ytest[i] else 'red')
fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14)

plt.show()

from sklearn.metrics import classification_report
print(classification_report(ytest, yfit,
                            target_names=faces.target_names))

"""**Confusion matrix between features in a heatmap**"""

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, yfit)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=faces.target_names,
            yticklabels=faces.target_names)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()