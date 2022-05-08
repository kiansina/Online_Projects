import numpy as np
import pandas as pd

def answer_one():

    df=pd.read_csv('fraud_data.csv')
    A=len(df[df['Class']==1])/len(df)
    return A # Return your answer

# Use X_train, X_test, y_train, y_test for all of the following questions
from sklearn.model_selection import train_test_split

df = pd.read_csv('fraud_data.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

def answer_two():
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import accuracy_score, recall_score
    dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
    y_dummy_predictions = dummy_majority.predict(X_test)
    A=accuracy_score(y_test,y_dummy_predictions)
    B=recall_score(y_test,y_dummy_predictions)


    # Your code here

    return (A,B)# Return your answer


def answer_three():
    from sklearn.metrics import accuracy_score, recall_score, precision_score
    from sklearn.svm import SVC

    m=SVC().fit(X_train,y_train)
    y_p=m.predict(X_test)
    A=accuracy_score(y_test,y_p)
    B=recall_score(y_test,y_p)
    C=precision_score(y_test,y_p)
    # Your code here

    return (A,B,C) # Return your answer


def answer_four():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC
    from sklearn.metrics import precision_recall_curve
    m=SVC(C= 1e9, gamma= 1e-07).fit(X_train,y_train)
    y_scores_m = m.decision_function(X_test)>-220
    #precision, recall, thresholds = precision_recall_curve(y_test, y_scores_m)
    #closest_zero = np.argmin(np.abs(thresholds+220))
    #closest_zero_p = precision[closest_zero]
    #closest_zero_r = recall[closest_zero]
    confusion_mc = confusion_matrix(y_test, y_scores_m)
    # Your code here

    return confusion_mc # Return your answer


def answer_five():
    from sklearn.linear_model import LogisticRegression
    #import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    lr=LogisticRegression().fit(X_train,y_train)
    y_scores_lr=lr.decision_function(X_test)
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores_lr)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_scores_lr)
    #plt.figure()
    #plt.xlim([0.0, 1.01])
    #plt.ylim([0.0, 1.01])
    #plt.plot(precision, recall, label='Precision-Recall Curve')
    closest_zero = np.argmin(np.abs(precision-0.75))
    closest_zero_r = recall[closest_zero]
    #plt.plot(0.75, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
    #plt.xlabel('Precision', fontsize=16)
    #plt.ylabel('Recall', fontsize=16)
    #plt.axes().set_aspect('equal')
    #plt.show()
    closest_zero2 = np.argmin(np.abs(fpr_lr-0.16))
    closest_zero_p2 = tpr_lr[closest_zero2]

    # Your code here

    return (closest_zero_r, closest_zero_p2)# Return your answer


def answer_six():
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    #C=[0.01, 0.1, 1, 10, 100]
    grid_values={'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10, 100]}
    lr=LogisticRegression()
    OP = GridSearchCV(lr, param_grid = grid_values, scoring = 'recall')
    OP.fit(X_train, y_train)
    # Your code here

    return np.array([OP.cv_results_['mean_test_score'][x:x+2] for x in range(0, len(OP.cv_results_['mean_test_score']), 2)])# Return your answer

answer_six()
