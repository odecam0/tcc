from arpc_utils import p_gen

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from pdb import set_trace

# Data Splits

def data_split1(dataset, select=0):
    """
    Escolhe o participante com SELECT e retorna 75% dos dados deste
    participante como treino e 25% como avaliação
    """

    for df in p_gen(dataset, select=[select]):
        X = df.drop(columns=['atividade', 'intensidade', 'tempo', 'sensor', 'participante'])
        y = [i + j for i, j in zip(df['atividade'], df['intensidade'])]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    return [(X_train, y_train)], [(X_test, y_test)]

# Train models

def train_randomforest(train_data, train_label):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(train_data, train_label)
    return rf

# Eval models

def eval_randomforest(trained_model, eval_data, eval_label):
    prediction = trained_model.predict(eval_data)
    cmat = metrics.confusion_matrix(eval_label, prediction, labels=trained_model.classes_)
    return cmat
