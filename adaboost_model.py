'''
AWS - BP KT Session.
'''

# 0. Imports.
import boto3
import numpy as np
import pickle
import os
import sys
import subprocess
# import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib


# 1. Construct dataset.
X1, y1 = make_gaussian_quantiles(cov=2.,
                                 n_samples=200, n_features=2,
                                 n_classes=2, random_state=1)
X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,
                                 n_samples=300, n_features=2,
                                 n_classes=2, random_state=1)
X = np.concatenate((X1, X2))
y = np.concatenate((y1, -y2 + 1))

# plot the dataset.
# plt.scatter(X[:, 1], X[:, 0], c=y, cmap='viridis')

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=0)
print(y_test.sum() / len(y_test))

# 2. Create and fit an AdaBoosted decision tree
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)

bdt.fit(X_train, y_train)
prediction_class = bdt.predict(X_test)
accuracy = accuracy_score(prediction_class, y_test)
accuracy = str(accuracy)

# 3. Store model test accuracy into DynamoDB.
session = boto3.session.Session(region_name='eu-west-1')
dynamodb = session.resource('dynamodb')

table = dynamodb.Table("ModelTestAccuracyTable")

model_version = "version1.0.3"

rand_number = int(abs(round(np.random.randn() * 1000, 0)))
print(rand_number)
table.put_item(
    Item={
        'pk': rand_number,
        'sk': 'adaboost',
        'model_version': {"V": model_version},
        'author': 'your_name',
        'model_accuracy': {"accuracy": accuracy}
    }
)  # , overwrite=False
print(rand_number)
# 4. Save the model to S3.
current_pwt = os.getcwd()
print(current_pwt)
# Please create a s3 bucket or using an exisiting bucket with particular key name.
s3_path = "s3://bp-aws-services-demos-hsz/aws-batch-demos/models/"

# 4.a Model with pickle
# save the model to disk
filename = '/finalized_model_pickle.sav'
filename_path = current_pwt + filename
print(filename_path)
pickle.dump(bdt, open(filename_path, 'wb'))

# Upload the file to S3
subprocess.run(["aws", "s3", "cp", filename_path, s3_path])
# subprocess.run(["aws", "s3", "cp", filename_path, s3_path, "--profile", "aws-terraform"])

# some time later... load the model from disk
loaded_model = pickle.load(open(filename_path, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)

# 4.b Model with joblib (is part of the SciPy ecosystem and provides utilities for pipelining Python jobs.)
# save the model to disk
filename = '/finalized_model_joblib.sav'
filename_path = current_pwt + filename
joblib.dump(bdt, filename_path)
print(filename_path)

# Upload the file to S3
subprocess.run(["aws", "s3", "cp", filename_path, s3_path])
# subprocess.run(["aws", "s3", "cp", filename_path, s3_path, "--profile", "aws-terraform"])

# some time later... load the model from disk
loaded_model = joblib.load(filename_path)
result = loaded_model.score(X_test, y_test)
print(result)


# Tips for Finalizing Your Model (link: https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/)
# This section lists some important considerations when finalizing your machine learning models.

# --- The pickle API for serializing standard Python objects.
# --- The joblib API for efficiently serializing Python objects with NumPy arrays.
# RUN pip3 install -r /src/requirements.txt
