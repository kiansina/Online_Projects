import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
#cancer
#print(cancer.DESCR) # Print the data set description
# You should write your whole answer within the function provided. The autograder will call
# this function and compare the return value against the correct solution value
def answer_zero():
    # This function returns the number of features of the breast cancer dataset, which is an integer.
    # The assignment question description will tell you the general format the autograder is expecting
    return len(cancer['feature_names'])

# You can examine what your function returns by calling it in the cell. If you have questions
# about the assignment formats, check out the discussion forums for any FAQs
#answer_zero()
def answer_one():



    # Your code here
    cols=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
          'mean smoothness', 'mean compactness', 'mean concavity',
          'mean concave points', 'mean symmetry', 'mean fractal dimension',
          'radius error', 'texture error', 'perimeter error', 'area error',
          'smoothness error', 'compactness error', 'concavity error',
          'concave points error', 'symmetry error', 'fractal dimension error',
          'worst radius', 'worst texture', 'worst perimeter', 'worst area',
          'worst smoothness', 'worst compactness', 'worst concavity',
          'worst concave points', 'worst symmetry', 'worst fractal dimension',
          'target']
    df=pd.DataFrame(cancer['data'])
    dff=pd.DataFrame(cancer['target'])
    dz=df.merge(dff,how='outer',left_index=True, right_index=True)
    dz.columns=cols
    return dz


#answer_one()

def answer_two():
    d1 = answer_one()
    d1=d1['target']
    c0=0
    c1=0
    for i in d1:
        if i==0:
            c0+=1
        elif i==1:
              c1+=1

    target=pd.Series([c0,c1],index=['malignant','benign'])
    # Your code here

    return target# Return your answer


#answer_two()

def answer_three():
    df = answer_one()
    X=df[df.columns[:-1]]
    y=df[df.columns[-1]]

    # Your code here

    return (X,y)

#answer_three()

from sklearn.model_selection import train_test_split

def answer_four():
    X, y = answer_three()
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)

    # Your code here

    return X_train, X_test, y_train, y_test
#answer_four()

from sklearn.neighbors import KNeighborsClassifier

def answer_five():
    X_train, X_test, y_train, y_test = answer_four()
    knn=KNeighborsClassifier(1)
    knn=knn.fit(X_train,y_train)

    # Your code here

    return knn# Return your answer
#answer_five()


def answer_six():
    df = answer_one()
    knn=answer_five()
    X_train, X_test, y_train, y_test = answer_four()
    means = df.mean()[:-1].values.reshape(1, -1)
    knn.fit(X_train,y_train)
    pdic=knn.predict(means)

    # Your code here

    return pdic# Return your answer
#answer_six()

def answer_seven():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    knn.fit(X_train,y_train)
    pdc=knn.predict(X_test)

    # Your code here

    return pdc# Return your answer

#answer_seven()

def answer_eight():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    knn.fit(X_train,y_train)
    SC=knn.score(X_test,y_test)

    # Your code here

    return SC# Return your answer
#answer_eight()

def accuracy_plot():
    import matplotlib.pyplot as plt

    #%matplotlib notebook

    X_train, X_test, y_train, y_test = answer_four()

    # Find the training and testing accuracies by target value (i.e. malignant, benign)
    mal_train_X = X_train[y_train==0]
    mal_train_y = y_train[y_train==0]
    ben_train_X = X_train[y_train==1]
    ben_train_y = y_train[y_train==1]

    mal_test_X = X_test[y_test==0]
    mal_test_y = y_test[y_test==0]
    ben_test_X = X_test[y_test==1]
    ben_test_y = y_test[y_test==1]

    knn = answer_five()
    knn.fit(X_train,y_train)

    scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y),
              knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]


    plt.figure()

    # Plot the scores as a bar chart
    bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])

    # directly label the score onto the bars
    for bar in bars:
        height = bar.get_height()
        plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2),
                     ha='center', color='w', fontsize=11)

    # remove all the ticks (both axes), and tick labels on the Y axis
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

    # remove the frame of the chart
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8);
    plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)
    
