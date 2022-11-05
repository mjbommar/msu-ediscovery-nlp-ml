"""this source file simplifies some of the boilerplate/overwhelming stuff behind the scenes so students don't
get distracted in the notebook."""

# SPDX-License-Identifier: Apache-2.0
# Copyright 2022, Michael Bommarito

# turn off warnings for this notebook
import warnings

warnings.filterwarnings('ignore')

# now make tensorflow quiet via env variable
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import after disabling warnings
import numpy
import pandas
import spacy
import transformers
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.cluster
import sklearn.linear_model
import sklearn.tree

# create a basic spacy pipeline with sm
nlp = spacy.load("en_core_web_trf")


def get_doc(text: str):
    """get the spacy doc object"""
    return nlp(text)


def get_tokens(text: str, remove_stopword: bool = True) -> list:
    """get all the tokens from a string of text"""
    doc = nlp(text)
    if remove_stopword:
        return [token.text for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    else:
        return [token.text for token in doc]

def get_ngrams(text: str, n: int) -> list:
    """get all the ngrams from a string of text"""
    doc = nlp(text)

    # get ngrams from the doc
    return [doc[i:i + n] for i in range(len(doc) - n + 1)]

def get_token_term_frequency(text: str, remove_stopword: bool = True) -> dict:
    """get the token term frequency for a string of text"""
    tokens = get_tokens(text, remove_stopword)
    return {token: tokens.count(token) for token in tokens}


def get_tokens_and_lemmas(text: str) -> list:
    """get all the tokens and lemmas from a string of text"""
    doc = nlp(text)
    return [(token.text, token.lemma_) for token in doc]


def get_tokens_and_pos(text: str) -> list:
    """get all the tokens and part of speech from a string of text"""
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]


def get_tokens_and_dep(text: str) -> list:
    """get all the tokens and dependency from a string of text"""
    doc = nlp(text)
    return [(token.text, token.dep_) for token in doc]


def get_tokens_and_ner(text: str) -> list:
    """get all the tokens and named entity recognition from a string of text"""
    doc = nlp(text)
    return [(token.text, token.ent_type_) for token in doc if token.ent_type_ != '']


def draw_cluster_points(color: bool = False):
    # sample two random clusters
    X, y = sklearn.datasets.make_blobs(n_samples=100, centers=2, n_features=2, random_state=0, cluster_std=0.5)

    # create a scatter plot of the data
    if color:
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
    else:
        plt.scatter(X[:, 0], X[:, 1])

    # add a title
    plt.title('Two Random Clusters', fontsize=20)

    # show the plot
    plt.show()

    # return
    return X, y

def draw_cluster_points_clustered(X, y):
    # create a scatter plot of the data
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')

    # cluster the data with k-means
    kmeans = sklearn.cluster.KMeans(n_clusters=2, random_state=0)
    kmeans.fit(X)

    # plot the cluster centers as a black circle
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='black', s=200, alpha=0.5)

    # draw an ellipse/silhouette around each cluster
    for i in range(2):
        # get the points in the cluster
        points = X[numpy.where(kmeans.labels_ == i)]

        # get the center of the cluster
        center = kmeans.cluster_centers_[i]

        # get the distance from the center to each point
        distances = [numpy.linalg.norm(point - center) for point in points]

        # get the maximum distance
        max_distance = numpy.max(distances)

        # create a circle around the center
        circle = plt.Circle(center, max_distance, color='black', fill=True, alpha=0.1)
        plt.gca().add_artist(circle)

    # add a title
    plt.title('Two Random Clusters, Clustered', fontsize=20)

    # show the plot
    plt.show()

def draw_classification_points():
    # sample two random clusters
    X, y = sklearn.datasets.make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2,
                                                random_state=1, n_clusters_per_class=1)

    # create a scatter plot of the data
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')

    # add a title
    plt.title('Two Random Clusters', fontsize=20)

    # show the plot
    plt.show()

    # return
    return X, y


def draw_classification_points_logistic(X, y):
    # create a scatter plot of the data
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')

    # fit a logistic classifier
    model = sklearn.linear_model.LogisticRegression()

    # fit the model
    model.fit(X, y)

    # get the slope and intercept
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]

    # get the x and y limits
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()

    # create a grid of points
    xx, yy = numpy.meshgrid(numpy.linspace(x_min, x_max, 100), numpy.linspace(y_min, y_max, 100))

    # get the decision boundary
    Z = model.predict(numpy.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # plot the decision boundary
    plt.contour(xx, yy, Z, colors='black', levels=[0.5], alpha=0.5, linestyles=['-'])

    # add a title
    plt.title('Two Random Clusters, Logistic Boundary', fontsize=20)

    # show the plot
    plt.show()



def draw_classification_points_tree(X, y):
    # create a scatter plot of the data
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')

    # fit a logistic classifier
    model = sklearn.tree.DecisionTreeClassifier()

    # fit the model
    model.fit(X, y)

    # get the x and y limits
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()

    # create a grid of points
    xx, yy = numpy.meshgrid(numpy.linspace(x_min, x_max, 100), numpy.linspace(y_min, y_max, 100))

    # get the decision boundary
    Z = model.predict(numpy.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # plot the decision boundary
    plt.contour(xx, yy, Z, colors='black', levels=[0.5], alpha=0.5, linestyles=['-'])

    # add a title
    plt.title('Two Random Clusters, Decision Tree Boundary', fontsize=20)

    # show the plot
    plt.show()


if __name__ == "__main__":
    X, y = draw_classification_points()
    draw_classification_points_logistic(X, y)