from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def get_paper_classifiers(random_state=42):
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=random_state
        ),
        "SVM Linear": LinearSVC(
            dual=False,
            class_weight='balanced',
            max_iter=2000,
            random_state=random_state
        ),
        "SVM RBF": SVC(
            kernel='rbf',
            class_weight='balanced',
            max_iter=2000,
            random_state=random_state
        ),
        "Logistic Regression": LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=random_state
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=30,
            max_depth=3,
            random_state=random_state
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=30,
            random_state=random_state
        ),
        "Naive Bayes": GaussianNB(),
        "KNN (k=7)": KNeighborsClassifier(n_neighbors=7),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=15,
            class_weight='balanced',
            random_state=random_state
        ),
        "Multinomial NB": MultinomialNB()
    }
