from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

GRAPHS_FOLDER = "bayes_graphs/"
data = pd.read_csv("base_aps_failure_trainingCla.csv")
X_train = np.array(data.drop("class",axis=1))
y_train = np.array(data["class"])
target_names = np.array(["0","1"])

res = pd.read_csv("base_aps_failure_testCla.csv")
X_test = np.array(res.drop("class",axis=1))
y_test = np.array(res["class"])
# split dataset into training/test portions
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

# PCA part
pca = PCA(n_components=2).fit(X)
X_pca = pca.transform(X)

pca = PCA(n_components=2).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

def plot_confusion_matrix(cnf_matrix, classesNames, normalize=False,cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
    """
    np.set_printoptions(precision=2)

    if normalize:
        soma = cnf_matrix.sum(axis=1)[:, np.newaxis]
        cm = cnf_matrix.astype('float') / soma
        title = "Normalized confusion matrix"
    else:
        cm = cnf_matrix
        title = 'Confusion matrix, without normalization'

    print(cm)

    plt.figure()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classesNames))
    plt.xticks(tick_marks, classesNames, rotation=45)
    plt.yticks(tick_marks, classesNames)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def run_all_bayes(X, y, X_train, y_train, X_test, y_test, option):
    if option == "GaussianNB":
        gnb = GaussianNB()
    elif option == "MultinomialNB":
        gnb = MultinomialNB()
    elif option == "BernoulliNB":
        gnb = BernoulliNB()
    gnb.fit(X_train,y_train)
    y_pred = gnb.predict(X_test)
    print("\nBayes classifier")
    #cnf_matrix = classification_report(y_test,y_pred,target_names=target_names)
    
    
    
    
    conf_mat = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(conf_mat, annot=True, fmt='d')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    
    print(confusion_matrix(y_test,y_pred, labels=range(2)))
    print("Accuracy score: %f" % (accuracy_score(y_test,y_pred)))
    print("ROC auc score: %f" % (roc_auc_score(y_test,y_pred)))
    #res = 0
    # Nao funciona no PCA
    # for el in range(0,len(X_test)):
    #     sol = X_test[el,0]
    #     if y_pred[el] != sol:
    #         res = res + 1 
    # print("Number of mislabeled points out of a total %d points : %d" % (len(X_test),res))
    if option == "GaussianNB":
        gnb = GaussianNB()
    elif option == "MultinomialNB":
        gnb = MultinomialNB()
    elif option == "BernoulliNB":
        gnb = BernoulliNB()
    gnb.fit(X,y)
    print("Cross-Validation (10-fold) score: %f" % (cross_val_score(gnb, X, y, cv=10).mean()))
    return accuracy_score(y_test,y_pred)

def draw_learning_curve(X, y, X_pca, filename):
    clf = GaussianNB()
    train_sizes,train_scores, test_scores = learning_curve(
        clf, X, y, cv=10, n_jobs=8)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    clf_pca = GaussianNB()
    train_sizes_pca,train_scores_pca, test_scores_pca = learning_curve(
        clf, X_pca, y, cv=10, n_jobs=8)
    train_scores_mean_pca = np.mean(train_scores_pca, axis=1)
    train_scores_std_pca = np.std(train_scores_pca, axis=1)
    test_scores_mean_pca = np.mean(test_scores_pca, axis=1)
    test_scores_std_pca = np.std(test_scores_pca, axis=1)

    f = plt.figure()
    plt.title("Learning Curve Naive Bayes - " + filename)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.gca().set_ylim([0.46,1.06])
    plt.grid()

    plt.plot(train_sizes, train_scores_mean, '.-', color="r",
             label="Training score Non-PCA")
    plt.plot(train_sizes_pca, train_scores_mean_pca, '.-', color="b",
             label="Training score with PCA")
    plt.plot(train_sizes, test_scores_mean, '.-', color="g",
             label="Cross-validation score Non-PCA")
    plt.plot(train_sizes_pca, test_scores_mean_pca, '.-', color="y",
             label="Cross-validation score with PCA")

    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))   
    #plt.show()
    f.savefig(GRAPHS_FOLDER+"lc_bayes_"+filename+".png",bbox_inches="tight")
    f.savefig(GRAPHS_FOLDER+"lc_bayes_"+filename+".pdf",bbox_inches="tight")

def run_non_pca_knn():  
    return run_all_bayes(X, y, X_train, y_train, X_test, y_test, "GaussianNB")


def run_pca_knn():
    return run_all_bayes(X_pca, y, X_train_pca,y_train,X_test_pca,y_test,"GaussianNB")

def draw_all_learning_curves():
    draw_learning_curve(X,y,X_pca,"default")

yaxis = [run_pca_knn()]
x = ["pca"]
width = 1/1.5
plt.bar(x, yaxis, width, color="blue")
plt.title("accuracy")
plt.gca().set_ylim([0.9,1])
plt.xlabel("accuracy score")
plt.ylabel("gaussian Naive Bayes")
fig = plt.gcf()
plotly_fig = tls.mpl_to_plotly(fig)
py.iplot(plotly_fig, filename='mpl-basic-bar')
#draw_all_learning_curves()

'''

DETERMINAR NUMERO DE COMPONENTES
acc = []
for n in range(2,11):
    sm = SMOTE(random_state=2)
    X_sm, y_sm = sm.fit_sample(X,y)
    X_train_sm, y_train_sm = sm.fit_sample(X_train,y_train)

    pca_test = PCA(n_components=n).fit(X_sm)
    X_sm_pca_test = pca_test.transform(X_sm)

    pca_test = PCA(n_components=n).fit(X_train_sm)
    X_train_sm_pca_test = pca_test.transform(X_train_sm)
    X_test_sm_pca_test = pca_test.transform(X_test)

    # pca_test = PCA(n_components=n).fit(X)
    # X_pca_test = pca_test.transform(X)

    # pca_test = PCA(n_components=n).fit(X_train)
    # X_train_pca_test = pca_test.transform(X_train)
    # X_test_pca_test = pca_test.transform(X_test)

    clf = GaussianNB()
    clf.fit(X_train_sm_pca_test,y_train_sm)
    y_pred = clf.predict(X_test_sm_pca_test)
    accu = accuracy_score(y_test,y_pred)
    acc.append(accu)
    print("Accuracy score for %d components: %f" % (n , (accu)))

x = [2,3,4,5,6,7,8,9,10]
x = pd.Series.from_array(x)
width = 1/1.5
plt.bar(x, acc, width, color="blue")
plt.title("accuracy")
plt.gca().set_ylim([0.95,1])
plt.xlabel('componentes')
plt.ylabel('accuracy score')
fig = plt.gcf()
plotly_fig = tls.mpl_to_plotly(fig)
py.iplot(plotly_fig, filename='mpl-basic-bar')
'''