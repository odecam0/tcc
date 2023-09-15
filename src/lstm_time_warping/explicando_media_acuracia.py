# Tornando os modulos ARPC acessíveis

import sys
from pathlib import Path
p = Path("../")
sys.path.append(str(p.resolve()))

import ARPC
import ie_data as ie
import numpy as np
from pprint import pprint

# Acessando um conjunto de experimentos que já foi realizado e foi
# salvado em um arquivo utilizando a biblioteca 'pickle'

test = ARPC.Arpc()
test = ie.load_group_of_experiments("dataaug_classified_arpos")

# Os experimentos são armazenados como uma lista encadeada de objetos
# da classe Arpc. O último experimento realizado possui uma referência
# ao penúltimo, que possui uma referência ao anti-penultimo, assim por
# diante até se chegar ao primeiro.

(test, test.past_exp, test.past_exp.past_exp)

# Há um método que retornar um 'iterable' com todas as referências de
# todos os experimentos encadeados em um objeto.

[i for i in test.exp_gen()]

# Cada experimento tem um nome

[i.name for i in test.exp_gen()]

# É possível criar uma função que, dado um nome, retorna o experimento da
# cadeia de experimentos com aquele nome

def get_experiment_by_name(arpo, name):
    for e in arpo.exp_gen():
        if e.name == name:
            exp = e
    return exp

# De acordo com a seguinte imagem: (find-tccdalstmfile "lstm-dataaug-comparisson.png")
# o experimento cujo nome é 'Split LOSO com aumento de dados' teve uma acurácia nula
# nas posturas 'Sentado Moderado' e 'Sentado Vigoroso', então este esperimento será
# investigado para averiguar se isto procede.

experimento = get_experiment_by_name(test, 'Split LOSO com aumento de dados')

# Cada experimento recebe, como etapa final no processo de classificação, uma lista com 2-tuplas
# contendo cada tupla, uma matriz de confusão, e uma lista com os labels associados à esta matriz.
# Desta forma, cada objeto Arpc agrupa uma sequência de experimentos relacionados. Neste caso temos
# uma matriz de confusão com dados de classificação para cada participante considerado, ou seja, 11
# matrizes de confusão.

# Quantidade de participantes        | Matriz de confusão                  | Lista com labels
( len(experimento.confusion_matrixes), experimento.confusion_matrixes[0][0], experimento.confusion_matrixes[0][1])

# O modulo arpc_metrics possui uma função que retorna a acurácia de um dado label de uma matriz de confusão
# que recebe a 2-tupla citada anteriormente e o indice do label como parâmetro, eis sua implementação:

def label_accuracy(cm, label:int):
    cm = cm[0]
    correct_predictions = cm.diagonal()[label]
    total_predictions   = cm[label, :].sum()
    
    return correct_predictions/total_predictions

# Podemos utiliza-la na primeira matriz de confusão do conjunto de experimentos em 'experimento' para
# conseguir todas as acurácias associadas à todos os labels.

primeira_matriz = experimento.confusion_matrixes[0]
quantidade_labels = len(primeira_matriz[1])
[label_accuracy(primeira_matriz, i) for i in range(quantidade_labels)]

# Realizando a mesma coisa para todas as matrizes do conjunto de experimentos, conseguimos uma matriz
# de acurácias, onde cada linha (no eixo 0) representa um experimento com um participante diferente,
# e cada coluna (no eixo 1) representa um label diferente.

matrizes = experimento.confusion_matrixes
matriz_experimento_label = [[label_accuracy(matrizes[j], i) for i in range(quantidade_labels)] for j in range(len(matrizes))]
pprint(matriz_experimento_label)

# Utilizando a função 'sum' do 'numpy', é possível somar os dados de uma matriz na direção de algum eixo, se especificarmos
# o eixo 0 estaremos somando todas as linhas da matriz, ficando com apenas uma linha possuindo a soma das acurácias de cada
# experimento com cada participanda, para cada label

soma_acuracias_experimentos = np.sum(matriz_experimento_label, axis=0)
soma_acuracias_experimentos

# Para que se tornem as médias das acurácias (aquilo que foi exibido no gráfico) basta que os valores
# sejam divididos pela quantidade de participantes.

soma_acuracias_experimentos = soma_acuracias_experimentos / 11
soma_acuracias_experimentos

# Para facilitar a visualização podemos criar um dicionário com o nome do label e sua acurácia somada, e
# exibilo utilizando a biblioteca pprint para que fique mais organizado.

labels = matrizes[0][1]
dicionario = {}
[dicionario.update({l:v}) for l,v in zip(labels, soma_acuracias_experimentos)]
pprint(dicionario)

