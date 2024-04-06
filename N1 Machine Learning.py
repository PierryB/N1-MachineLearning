from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

# Carregando o Data Set
iris = load_iris()
X = iris.data
y = iris.target

# Binarização do target para a Curva ROC
y_bin = label_binarize(y, classes=[0, 1, 2])
n_classes = y_bin.shape[1]

# Dividindo em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.3, random_state=42, stratify=y)

# Definição e treinamento dos classificadores
classifiers = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# Armazenamento dos resultados
results = {}

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_score = clf.predict(X_test)
    
    # Cálculo das métricas
    accuracy = accuracy_score(y_test, y_score)
    precision = precision_score(y_test, y_score, average='macro')
    recall = recall_score(y_test, y_score, average='macro')
    
    # Armazenamento dos resultados
    results[name] = {'accuracy': accuracy, 'precision': precision, 'recall': recall}

    # Curva ROC (AUC precisa de scores de probabilidade, aqui usamos predict para simplificação)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Gráfico da Curva ROC para cada classe
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {name}')
    plt.legend(loc="lower right")
    plt.show()

# Exibindo os resultados
for name, metrics in results.items():
    print(f"{name}:")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}\n")
