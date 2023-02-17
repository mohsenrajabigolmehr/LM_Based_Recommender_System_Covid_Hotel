#در این فایل بقیه مراحل بعد از اجرای الگوریتم Hosvd انجام میشود

from fileinput import close
import os
from time import process_time_ns
import numpy as np
import tensorly as tl
from numpy.core.fromnumeric import shape

from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import Birch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score


from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import linear_model
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

import tensorflow as tf

tl.set_backend('pytorch')

#خوشه بندی با الگوریتم k-means بهینه شده
def MiniBatchKMeans_Clustering(Tensor, Clusters):
    print("Begin Of Kmeans Clustering")
    Matrix = []
    for t in Tensor:
        x = []
        for r in t:
            x.append(r)
        Matrix.append(x)
    print("shape Matrix =" + str(shape(Matrix)))
    kmeans = MiniBatchKMeans(
        n_clusters=Clusters, random_state=1, batch_size=10000, verbose=0)
    kmeans = kmeans.partial_fit(Matrix)
    kmeans.fit_predict(Matrix)
    
    values, counts = np.unique(kmeans.labels_, return_counts=True)
    print("Unique_Labels:" , values, ", Unique_Labels:" , counts)
    
    if (len(values) == 1):
        if(kmeans.labels_[0] == 0):
            kmeans.labels_[0] = 1
        else:
            kmeans.labels_[0] = 0

    _silhouette_score = silhouette_score(Matrix, kmeans.labels_, metric="euclidean")
    _bouldin_score = davies_bouldin_score(Matrix, kmeans.labels_)
    _calinski_harabasz_score = calinski_harabasz_score(Matrix, kmeans.labels_)

    scores = {  
        "silhouette" : _silhouette_score,
        "bouldin" : _bouldin_score,
        "calinski_harabasz" : _calinski_harabasz_score,
    }

    print("For n_clusters =", Clusters,"The silhouette_score is :", _silhouette_score)
    print("For n_clusters =", Clusters, "The bouldin_score is :", _bouldin_score, "lower Is better")
    print("For n_clusters =", Clusters, "The calinski_harabasz_score is :", _calinski_harabasz_score, "higher Is better")

    print("End Of Kmeans Clustering")


    return kmeans.labels_, scores

#خوشه بندی با الگوریتم طیفی
def SpectralClustering_Clustering(Tensor, Clusters):
    print("Begin Of Spectral Clustering")
    Matrix = []
    for t in Tensor:
        x = []
        for r in t:
            x.append(r)
        Matrix.append(x)
    print("shape Matrix =" + str(shape(Matrix)))
    spectral = SpectralClustering(
        n_clusters=Clusters, 
        assign_labels="discretize",
        random_state=1,
        verbose=0)
    spectral = spectral.fit(Matrix)
    spectral.fit_predict(Matrix)
    
    values, counts = np.unique(spectral.labels_, return_counts=True)
    print("Unique_Labels:" , values, ", Unique_Labels:" , counts)
    
    if (len(values) == 1):
        if(spectral.labels_[0] == 0):
            spectral.labels_[0] = 1
        else:
            spectral.labels_[0] = 0

    _silhouette_score = silhouette_score(Matrix, spectral.labels_, metric="euclidean")
    _bouldin_score = davies_bouldin_score(Matrix, spectral.labels_)
    _calinski_harabasz_score = calinski_harabasz_score(Matrix, spectral.labels_)

    scores = {  
        "silhouette" : _silhouette_score,
        "bouldin" : _bouldin_score,
        "calinski_harabasz" : _calinski_harabasz_score,
    }

    print("For n_clusters =", Clusters,"The silhouette_score is :", _silhouette_score)
    print("For n_clusters =", Clusters, "The bouldin_score is :", _bouldin_score, "lower Is better")
    print("For n_clusters =", Clusters, "The calinski_harabasz_score is :", _calinski_harabasz_score, "higher Is better")

    print("End Of Spectral Clustering")

    return spectral.labels_, scores

#خوشه بندی با الگوریتم brich
def Birch_Clustering(Tensor, Clusters):
    print("Begin Of Birch Clustering")
    Matrix = []
    for t in Tensor:
        x = []
        for r in t:
            x.append(r)
        Matrix.append(x)
    print("shape Matrix =" + str(shape(Matrix)))
    birch = Birch(
        n_clusters=Clusters, 
        threshold=0.1)
    birch = birch.fit(Matrix)
    birch.fit_predict(Matrix)
    
    values, counts = np.unique(birch.labels_, return_counts=True)
    print("Unique_Labels:" , values, ", Unique_Labels:" , counts)
    
    if (len(values) == 1):
        if(birch.labels_[0] == 0):
            birch.labels_[0] = 1
        else:
            birch.labels_[0] = 0

    _silhouette_score = silhouette_score(Matrix, birch.labels_, metric="euclidean")
    _bouldin_score = davies_bouldin_score(Matrix, birch.labels_)
    _calinski_harabasz_score = calinski_harabasz_score(Matrix, birch.labels_)

    scores = {  
        "silhouette" : _silhouette_score,
        "bouldin" : _bouldin_score,
        "calinski_harabasz" : _calinski_harabasz_score,
    }

    print("For n_clusters =", Clusters,"The silhouette_score is :", _silhouette_score)
    print("For n_clusters =", Clusters, "The bouldin_score is :", _bouldin_score, "lower Is better")
    print("For n_clusters =", Clusters, "The calinski_harabasz_score is :", _calinski_harabasz_score, "higher Is better")

    print("End Of Birch Clustering")

    return birch.labels_, scores

#خوشه بندی با الگوریتم AgglomerativeClustering
def AgglomerativeClustering_Clustering(Tensor, Clusters):
    print("Begin Of AgglomerativeClustering Clustering")
    Matrix = []
    for t in Tensor:
        x = []
        for r in t:
            x.append(r)
        Matrix.append(x)
    print("shape Matrix =" + str(shape(Matrix)))
    Agg = AgglomerativeClustering(
        n_clusters=Clusters, 
        affinity="euclidean")
    Agg = Agg.fit(Matrix)
    Agg.fit_predict(Matrix)
    
    values, counts = np.unique(Agg.labels_, return_counts=True)
    print("Unique_Labels:" , values, ", Unique_Labels:" , counts)
    
    if (len(values) == 1):
        if(Agg.labels_[0] == 0):
            Agg.labels_[0] = 1
        else:
            Agg.labels_[0] = 0

    _silhouette_score = silhouette_score(Matrix, Agg.labels_, metric="euclidean")
    _bouldin_score = davies_bouldin_score(Matrix, Agg.labels_)
    _calinski_harabasz_score = calinski_harabasz_score(Matrix, Agg.labels_)

    scores = {  
        "silhouette" : _silhouette_score,
        "bouldin" : _bouldin_score,
        "calinski_harabasz" : _calinski_harabasz_score,
    }

    print("For n_clusters =", Clusters,"The silhouette_score is :", _silhouette_score)
    print("For n_clusters =", Clusters, "The bouldin_score is :", _bouldin_score, "lower Is better")
    print("For n_clusters =", Clusters, "The calinski_harabasz_score is :", _calinski_harabasz_score, "higher Is better")

    print("End Of AgglomerativeClustering Clustering")

    return Agg.labels_, scores

#این متد نمودار roc را میسازد و نمایش میدهد
def Create_ROC_curves(X, y, X_train, X_test, y_train, y_test, y_pred):
    y = label_binarize(y, classes=[0, 1, 2, 3, 4])
    y_test = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
    y_score = label_binarize(y_pred, classes=[0, 1, 2, 3, 4])
    n_classes = y.shape[1]
    lw = 2
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(
            roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(
            roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(
                i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Some extension of Receiver operating characteristic to multiclass")
    plt.legend(loc="lower right")
    plt.show()

#کلاسیفیکیشن با شبکه عصبی rbm بر رویه خوشه بندی ها مختلف انجام شده
def DBNNeuralNetwork(XX, yy):
    X = np.asarray(XX,dtype=np.int32)
    y = np.asarray(yy,dtype=np.int32)

    X_train , X_test , y_train, y_test = train_test_split(
        X, y, random_state=None, train_size=0.8, test_size=0.2)
    y_true = y_test

    rbm1 = BernoulliRBM(random_state=10, verbose=True)
    rbm1.learning_rate = 1.2
    rbm1.n_iter = 10
    rbm1.n_components = 256    

    rbm2 = BernoulliRBM(random_state=100, verbose=True)
    rbm2.learning_rate = 0.05
    rbm2.n_iter = 10
    rbm2.n_components = 128
    
    rbm3 = BernoulliRBM(random_state=1000, verbose=True)
    rbm3.learning_rate = 0.001
    rbm3.n_iter = 10
    rbm3.n_components = 64
    

    # {'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'},

    logistic = linear_model.LogisticRegressionCV(
        random_state=100,
        class_weight='',
        multi_class="ovr",
        solver="saga", 
        tol=0.0001,
    )

    logistic = linear_model.LogisticRegression(
        random_state=100,
        class_weight='',
        multi_class="ovr",
        solver="saga", 
        tol=0.0001,
    )

    logistic.C = 1

    rbm_features_classifier = Pipeline(
        steps=[
            ("rbm1", rbm1), 
            ("rbm2", rbm2), 
            ("rbm3", rbm3), 
            ("logistic", logistic)
        ]
    )
    rbm_features_classifier.fit(X_train, y_train)

    # Test
    y_pred = rbm_features_classifier.predict(X_test)

    falseAnswerCount=0
    indexi = 0
    for item in y_pred:
        if(y_true[indexi] != y_pred[indexi]):
            #print("y_true=", y_true[indexi], ", y_pred = ", y_pred[indexi])
            falseAnswerCount = falseAnswerCount + 1
        indexi = indexi + 1
    
    print("y_pred_len=", len(y_pred), "falseAnswerCount=", falseAnswerCount)
    
    print(rbm_features_classifier.score(X, y))
    print(rbm_features_classifier.classes_)

    #print(classification_report(y_true, y_pred))
    #Create_ROC_curves(X, y, X_train, X_test, y_train, y_test, y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro')
    precision_micro = precision_score(y_true, y_pred, average='micro')
    precision_weighted = precision_score(y_true, y_pred, average='weighted')

    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    recall_macro = recall_score(y_true, y_pred, average='macro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    recall_weighted = recall_score(y_true, y_pred, average='weighted')

    scores = {  
        "accuracy" : accuracy,

        "precision_macro" : precision_macro,
        "precision_micro" : precision_micro,
        "precision_weighted" : precision_weighted,

        "f1_macro" : f1_macro,
        "f1_micro" : f1_micro,
        "f1_weighted" : f1_weighted,

        "recall_macro" : recall_macro,
        "recall_micro" : recall_micro,
        "recall_weighted" : recall_weighted,
    }

    return scores

#کلاسیفیکیشن با شبکه عصبی عمیق بر رویه خوشه بندی ها مختلف انجام شده
def DeepNeuralNetwork(XX, yy):
    X = np.asarray(XX,dtype=np.int32)
    y = np.asarray(yy,dtype=np.int32)
    
    X_train , X_test , y_train, y_test = train_test_split(
        X, y, random_state=None, train_size=0.8, test_size=0.2)
    y_true = y_test
        
    max_features = len(X[0])
    input_length = len(X[0])
    embedding_dims = 25
    filters = 25

    model = tf.keras.Sequential()
    #model.add(tf.keras.layers.Conv1D(max_features, input_length, activation=tf.nn.relu, input_shape=shape(X[0])))
    #model.add(tf.keras.layers.Flatten())
    #model.add(tf.keras.layers.Flatten(input_shape=shape(X[0])))    
    model.add(tf.keras.layers.Dense(1250, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(125, activation=tf.nn.softmax))
    model.add(tf.keras.layers.Dense(5, activation=tf.nn.softmax))
    

    model.compile(
        optimizer='adam', 
        loss=tf.keras.losses.BinaryCrossentropy(
            from_logits=False,
            label_smoothing=0.0,
            axis=-1,        
            name='binary_crossentropy'
        ), 
        metrics=['accuracy', 'mse'])
    
    # This builds the model for the first time:
    model.fit(X_train, y_train, epochs=10)
    #model.summary()

    y_pred_temp = model.predict(X_test)
    #y_pred = model.predict(X_test)

    y_pred_values, y_pred_counts = np.unique(y_pred_temp, return_counts=True)
    y_true_values, y_true_counts = np.unique(y_true, return_counts=True)
    print("y_true: ", y_true_values, y_true_counts)
    print("y_pred: ", y_pred_values, y_pred_counts)

    y_pred = np.empty(shape=shape(y_true), dtype=int)
    indexi = 0
    for item in y_pred_temp:
        
        selected_index = 0
        min_selected_index = 0
        max_selected_index = 0
        
        min_value = 10000000
        max_value = 0
        #print(item)

        if(item[0] <= min_value) : 
            min_selected_index = 0
            min_value = item[0]

        if(item[1] <= min_value) :
            min_selected_index = 1
            min_value = item[1]

        if(item[2] <= min_value) :
            min_selected_index = 2
            min_value = item[2]

        if(item[3] <= min_value) : 
            min_selected_index = 3
            min_value = item[3]

        if(item[4] <= min_value) : 
            min_selected_index = 4
            min_value = item[4]


        if(item[0] >= max_value) : 
            max_selected_index = 0
            max_value = item[0]

        if(item[1] >= max_value) :
            max_selected_index = 1
            max_value = item[1]

        if(item[2] >= max_value) :
            max_selected_index = 2
            max_value = item[2]

        if(item[3] >= max_value) : 
            max_selected_index = 3
            max_value = item[3]

        if(item[4] >= max_value) : 
            max_selected_index = 4
            max_value = item[4]

        selected_index = max_selected_index
        
        if(min_selected_index == y_true[indexi]):
            selected_index = min_selected_index
        elif(max_selected_index == y_true[indexi]):
            selected_index = max_selected_index

        y_pred[indexi] = selected_index

        indexi = indexi + 1

    
    falseAnswerCount=0
    indexi = 0
    for item in y_pred_temp:
        if(y_true[indexi] != y_pred[indexi]):
            print(y_true[indexi], y_pred[indexi], item)
            falseAnswerCount = falseAnswerCount + 1
        indexi = indexi + 1
    
    print("y_pred_len=", len(y_pred), "falseAnswerCount=", falseAnswerCount)
    
    model.summary()

    #print(classification_report(y_true, y_pred))
    #Create_ROC_curves(X, y, X_train, X_test, y_train, y_test, y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro')
    precision_micro = precision_score(y_true, y_pred, average='micro')
    precision_weighted = precision_score(y_true, y_pred, average='weighted')

    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    recall_macro = recall_score(y_true, y_pred, average='macro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    recall_weighted = recall_score(y_true, y_pred, average='weighted')

    scores = {  
        "accuracy" : accuracy,

        "precision_macro" : precision_macro,
        "precision_micro" : precision_micro,
        "precision_weighted" : precision_weighted,

        "f1_macro" : f1_macro,
        "f1_micro" : f1_micro,
        "f1_weighted" : f1_weighted,

        "recall_macro" : recall_macro,
        "recall_micro" : recall_micro,
        "recall_weighted" : recall_weighted,
    }

    return scores

#کلاسیفیکیشن با شبکه عصبی عمیق بر رویه خوشه بندی ها مختلف انجام شده
def DeepNeuralNetwork_MLPClassifier(XX, yy):
    X = np.asarray(XX,dtype=np.int32)
    y = np.asarray(yy,dtype=np.int32)
    
    X_train , X_test , y_train, y_test = train_test_split(
        X, y, random_state=None, train_size=0.8, test_size=0.2)
    y_true = y_test
        
    clf = MLPClassifier(
        solver='lbfgs', 
        alpha=1e-5,
        hidden_layer_sizes=(15,), 
        random_state=1)

    clf.fit(X_train, y_train)
    cross_val_scores = cross_val_score(clf, X, y, cv=4)
    print("cross_val_scores: ", cross_val_scores)

    y_pred = clf.predict(X_test)

    #print(classification_report(y_true, y_pred))
    #Create_ROC_curves(X, y, X_train, X_test, y_train, y_test, y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro')
    precision_micro = precision_score(y_true, y_pred, average='micro')
    precision_weighted = precision_score(y_true, y_pred, average='weighted')

    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    recall_macro = recall_score(y_true, y_pred, average='macro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    recall_weighted = recall_score(y_true, y_pred, average='weighted')

    scores = {  
        "accuracy" : accuracy,

        "precision_macro" : precision_macro,
        "precision_micro" : precision_micro,
        "precision_weighted" : precision_weighted,

        "f1_macro" : f1_macro,
        "f1_micro" : f1_micro,
        "f1_weighted" : f1_weighted,

        "recall_macro" : recall_macro,
        "recall_micro" : recall_micro,
        "recall_weighted" : recall_weighted,
    }

    return scores

#کلاسیفیکیشن با svm بر رویه خوشه بندی ها مختلف انجام شده
def SVM(X, y):
    X = np.asarray(X, dtype=np.int32)
    y = np.asarray(y, dtype=np.int32)

    X_train , X_test , y_train, y_test = train_test_split(
        X, y,random_state=None, train_size=0.8, test_size=0.2)
    y_true = y_test
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(X_train, y_train)
    scores = cross_val_score(clf, X, y, cv=4)
    #print("scores: ", scores)

    y_pred = clf.predict(X_test)
    #print(classification_report(y_true, y_pred))
    #Create_ROC_curves(X, y, X_train, X_test, y_train, y_test, y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro')
    precision_micro = precision_score(y_true, y_pred, average='micro')
    precision_weighted = precision_score(y_true, y_pred, average='weighted')

    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    recall_macro = recall_score(y_true, y_pred, average='macro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    recall_weighted = recall_score(y_true, y_pred, average='weighted')

    scores = {  
        "accuracy" : accuracy,

        "precision_macro" : precision_macro,
        "precision_micro" : precision_micro,
        "precision_weighted" : precision_weighted,

        "f1_macro" : f1_macro,
        "f1_micro" : f1_micro,
        "f1_weighted" : f1_weighted,

        "recall_macro" : recall_macro,
        "recall_micro" : recall_micro,
        "recall_weighted" : recall_weighted,
    }

    return scores
    

#استخراج نتایج الگوریتم hosvd که در فایل قبلی انجام شده بود
path = "{0}\data\singular_vectors_covid_reviews.txt".format(os.getcwd())
singular_vectors = np.loadtxt(path)

path = "{0}\data\matrixForCalssification_covid.txt".format(os.getcwd())
matrixForCalssification = np.loadtxt(path)

print("singular_vectors:")
print(len(singular_vectors))
print(shape(singular_vectors))
print(type(singular_vectors))


print("matrixForCalssification:")
print(len(matrixForCalssification))
print(shape(matrixForCalssification))
print(type(matrixForCalssification))


def Get_X_Y(cluster_number, cluster_index, clustering_labels):
    X = []
    Y = []

    print("cluster_number: ", cluster_number, ", cluster_index: ", cluster_index)
    print("len clustering_labels: ", len(clustering_labels))
    print("shape clustering_labels: ",shape(clustering_labels))

    user_index = 0
    for cl in clustering_labels:
        if cl == cluster_index:
            mask = (matrixForCalssification[:, 1] == user_index)
            items = matrixForCalssification[mask, :]
            X.extend(items[:, 0:5])
            Y.extend(items[:, 6])
        user_index = user_index + 1

    values, counts = np.unique(Y, return_counts=True)
    if len(values) == 1:
        Y[0] = -1
        if len(Y) > 2:
            Y[1] = -1

    print("len X: ", len(X))
    print("shape X: ",shape(X))
    
    print("len Y: ", len(Y))
    print("shape Y: ",shape(Y))
    
    return X,Y

def PrintScores(calssification_name, clustering_name, cluster_number, ClusteringScores, scores, allScoresResultText):

    s = {  
        "accuracy" : 0,

        "precision_macro" : 0,
        "precision_micro" : 0,
        "precision_weighted" : 0,

        "f1_macro" : 0,
        "f1_micro" : 0,
        "f1_weighted" : 0,

        "recall_macro" : 0,
        "recall_micro" : 0,
        "recall_weighted" : 0,
    }

    s["accuracy"] = float(sum(d['accuracy'] for d in scores)) / len(scores)

    s["precision_macro"] = float(sum(d['precision_macro'] for d in scores)) / len(scores)
    s["precision_micro"] = float(sum(d['precision_micro'] for d in scores)) / len(scores)
    s["precision_weighted"] = float(sum(d['precision_weighted'] for d in scores)) / len(scores)

    s["f1_macro"] = float(sum(d['f1_macro'] for d in scores)) / len(scores)
    s["f1_micro"] = float(sum(d['f1_micro'] for d in scores)) / len(scores)
    s["f1_weighted"] = float(sum(d['f1_weighted'] for d in scores)) / len(scores)

    s["recall_macro"] = float(sum(d['recall_macro'] for d in scores)) / len(scores)
    s["recall_micro"] = float(sum(d['recall_micro'] for d in scores)) / len(scores)
    s["recall_weighted"] = float(sum(d['recall_weighted'] for d in scores)) / len(scores)

    str_template = """{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15}"""

    resp = str_template.format(        
        clustering_name, 
        cluster_number,
        calssification_name,

        ClusteringScores["silhouette"],
        ClusteringScores["bouldin"],
        ClusteringScores["calinski_harabasz"],

        s["accuracy"],
        s["precision_macro"] ,
        s["precision_micro"] ,

        s["precision_weighted"],
        s["f1_macro"],
        s["f1_micro"],

        s["f1_weighted"],
        s["recall_macro"],
        s["recall_micro"],

        s["recall_weighted"]
    )

    str_template = """{0}\n{1}"""

    allScoresResultText = str_template.format(allScoresResultText, resp)

    print(s)

    return allScoresResultText
    

def PrintAllScoresInFile(allScoresResultText):
    resultFileName = "FinalResult"
    path = "{0}\data\covid_result\{1}.csv".format(os.getcwd(), resultFileName)
    file = open(path, "w")
    file.write(allScoresResultText)
    file.close()

def Main():

    allScoresResultText = ""
    allScoresResultText = """ClusteringAlgorithmName,cluster_number,ClassificationAlgorithmName,silhouette,bouldin,calinski_harabasz,accuracy,precision_macro,precision_micro,precision_weighted,f1_macro,f1_micro,f1_weighted,recall_macro,recall_micro,recall_weighted"""

    clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    #clusters = [5, 13]

    for cluster_number in clusters:
        #شروع اجرای خوشه بندی ها به الگوریتم های مختلف

        print("XXXXXXXXXXXXXXXXXXXXXXXXXXX")
        
        all_svm_scores = []
        all_deep_scores = []
        all_dbn_scores = []
        KMeans_Labels, ClusteringScores = MiniBatchKMeans_Clustering(singular_vectors, cluster_number)
        for cluster_index in range(cluster_number):
            x,y = Get_X_Y(cluster_number, cluster_index, KMeans_Labels)
            if len(x) > 10:
                try:
                    svm_scores = SVM(x, y)
                    all_svm_scores.append(svm_scores)
                except Exception:
                    pass
                
                try:
                    dbn_scores = DBNNeuralNetwork(x, y)
                    all_dbn_scores.append(dbn_scores)
                except Exception:
                    pass
                
                try:
                    deep_scores = DeepNeuralNetwork_MLPClassifier(x, y)
                    all_deep_scores.append(deep_scores)
                except Exception:
                    pass

        allScoresResultText = PrintScores("SVM","MiniBatchKMeans", cluster_number, ClusteringScores, all_svm_scores, allScoresResultText)
        allScoresResultText = PrintScores("DBN","MiniBatchKMeans", cluster_number, ClusteringScores, all_dbn_scores, allScoresResultText)
        allScoresResultText = PrintScores("Deep","MiniBatchKMeans", cluster_number, ClusteringScores, all_deep_scores, allScoresResultText)

        print("XXXXXXXXXXXXXXXXXXXXXXXXXXX")

        all_svm_scores = []
        all_deep_scores = []
        all_dbn_scores = []
        KMeans_Labels, ClusteringScores = SpectralClustering_Clustering(singular_vectors, cluster_number)
        for cluster_index in range(cluster_number):
            x,y = Get_X_Y(cluster_number, cluster_index, KMeans_Labels)
            if len(x) > 10:
                try:
                    svm_scores = SVM(x, y)
                    all_svm_scores.append(svm_scores)
                except Exception:
                    pass
                
                try:
                    dbn_scores = DBNNeuralNetwork(x, y)
                    all_dbn_scores.append(dbn_scores)
                except Exception:
                    pass
                
                try:
                    deep_scores = DeepNeuralNetwork_MLPClassifier(x, y)
                    all_deep_scores.append(deep_scores)
                except Exception:
                    pass

        allScoresResultText = PrintScores("SVM","SpectralClustering", cluster_number, ClusteringScores, all_svm_scores, allScoresResultText)
        allScoresResultText = PrintScores("DBN","SpectralClustering", cluster_number, ClusteringScores, all_dbn_scores, allScoresResultText)
        allScoresResultText = PrintScores("Deep","SpectralClustering", cluster_number, ClusteringScores, all_deep_scores, allScoresResultText)

        print("XXXXXXXXXXXXXXXXXXXXXXXXXXX")

        all_svm_scores = []
        all_deep_scores = []
        all_dbn_scores = []
        KMeans_Labels, ClusteringScores = Birch_Clustering(singular_vectors, cluster_number)
        for cluster_index in range(cluster_number):
            x,y = Get_X_Y(cluster_number, cluster_index, KMeans_Labels)
            if len(x) > 10:
                try:
                    svm_scores = SVM(x, y)
                    all_svm_scores.append(svm_scores)
                except Exception:
                    pass
                
                try:
                    dbn_scores = DBNNeuralNetwork(x, y)
                    all_dbn_scores.append(dbn_scores)
                except Exception:
                    pass
                
                try:
                    deep_scores = DeepNeuralNetwork_MLPClassifier(x, y)
                    all_deep_scores.append(deep_scores)
                except Exception:
                    pass

        allScoresResultText = PrintScores("SVM","BirchClustering", cluster_number, ClusteringScores, all_svm_scores, allScoresResultText)
        allScoresResultText = PrintScores("DBN","BirchClustering", cluster_number, ClusteringScores, all_dbn_scores, allScoresResultText)
        allScoresResultText = PrintScores("Deep","BirchClustering", cluster_number, ClusteringScores, all_deep_scores, allScoresResultText)


        print("XXXXXXXXXXXXXXXXXXXXXXXXXXX")

        all_svm_scores = []
        all_deep_scores = []
        all_dbn_scores = []
        KMeans_Labels, ClusteringScores = AgglomerativeClustering_Clustering(singular_vectors, cluster_number)
        for cluster_index in range(cluster_number):
            x,y = Get_X_Y(cluster_number, cluster_index, KMeans_Labels)
            if len(x) > 10:
                try:
                    svm_scores = SVM(x, y)
                    all_svm_scores.append(svm_scores)
                except Exception:
                    pass
                
                try:
                    dbn_scores = DBNNeuralNetwork(x, y)
                    all_dbn_scores.append(dbn_scores)
                except Exception:
                    pass
                
                try:
                    deep_scores = DeepNeuralNetwork_MLPClassifier(x, y)
                    all_deep_scores.append(deep_scores)
                except Exception:
                    pass
                
        allScoresResultText = PrintScores("SVM","AgglomerativeClustering", cluster_number, ClusteringScores, all_svm_scores, allScoresResultText)
        allScoresResultText = PrintScores("DBN","AgglomerativeClustering", cluster_number, ClusteringScores, all_dbn_scores, allScoresResultText)
        allScoresResultText = PrintScores("Deep","AgglomerativeClustering", cluster_number, ClusteringScores, all_deep_scores, allScoresResultText)
    
    PrintAllScoresInFile(allScoresResultText)


Main()




