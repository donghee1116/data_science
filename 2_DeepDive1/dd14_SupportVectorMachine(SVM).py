
import matplotlib.pyplot as plt
import numpy as np
from log.logfile import logger
#import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

#load the digits dataset
digits = datasets.load_digits()
print('Digits dataset keys :\n{}'.format(digits.keys()))
logger.debug('Digits dataset keys :\n{}'.format(digits.keys()))

print('dataset target name: \n{}', format(digits.target_names))
print('shape of datasets: \n{}, \nand target:{}'.format(digits.data.shape, digits.target.shape))
print('shape of the images: \n{}'.format(digits.images.shape))

#the images are also included in the dataset as digits.images
for i in range(0,4):
    plt.subplot(2, 4, i+1)
    plt.axis('off')
    plt.imshow(digits.images[i])
    plt.title("Training: {}".format(digits.target[i]))

plt.show()


#simple image classifier with SVM

n_samples = len(digits.images)
data_images = digits.images.reshape((n_samples, -1))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_images, digits.target)
print("Training data and target sizes: \n{}, {}".format(X_train.shape, y_train.shape))
print("Test data and target sizes: \n{}, {}".format(X_test.shape, y_test.shape))

classifier = svm.SVC(gamma=0.001)

#fit to the training data
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print("Classification report for classifier: %s:\n%s\n"% (classifier, metrics.classification_report(y_test,y_pred)))























