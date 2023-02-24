import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from pdb import set_trace

def plot_all(df, save=False, participantes=[]):
    atividades  = list(df.atividade.drop_duplicates())
    intensidades = list(df.intensidade.drop_duplicates())

    if len(participantes) == 0:
        participantes = list(df.participante.drop_duplicates())

    for participante in participantes:
        fig, axs = plt.subplots(len(atividades), len(intensidades), sharey=True)
        fig.suptitle('Aluno'+str(participante))

        for a, n1 in zip(atividades, range(len(atividades))):
            for i, n2 in zip(intensidades, range(len(intensidades))):

                if n1 != len(atividades) - 1:
                    axs[n1][n2].set_xticks([])

                sub_d_raw = df.loc[(df['atividade'] == a) &
                                    (df['intensidade'] == i) &
                                    (df['participante'] == participante)]

                axs[n1][n2].plot(sub_d_raw['tempo'], sub_d_raw['x'], linewidth=0.5, color='blue')
                axs[n1][n2].plot(sub_d_raw['tempo'], sub_d_raw['y'], linewidth=0.5, color='red')
                axs[n1][n2].plot(sub_d_raw['tempo'], sub_d_raw['z'], linewidth=0.5, color='green')

        for a, n1 in zip(atividades, range(len(atividades))):
            axs[n1][0].set_ylabel(a)

        for i, n2 in zip(intensidades, range(len(intensidades))):
            axs[len(atividades)-1][n2].set_xlabel(i)

        if save:
            plt.savefig('Aluno' + str(participante) +'.png')
        else:
            plt.show()


# arpo -> Arpc Object -> Activity Recognition Protocol Object
def plot_compare_side_err_bar(arpo, get_data_proc, show=True, gray=False, fig_ax=None, eb_width=None, legend=True):
    """
    Deve haver um padrão para os argumentos que GET_DATA_PROC deve receber e oque deve retornar
    Receberá ARPO, e deve retornar uma lista com listas que poossuem tuplas, com o valor da métrica
    e o limite superior e inferior do intervalo de confiança.
    """

    # Experiments
    values, err_bars, exp_labels = get_data_proc(arpo)

    # Plotting code
    labels = arpo.confusion_matrixes[0][1]
    if fig_ax:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots()
    plots = []

    exp_num = values.shape[0]
    x = np.arange(len(labels))
    width = .5 / exp_num

    # exps[i][1] deve ser um array com shape (2, N) com o valor que somado
    # ao valor de exps[i][0] vai resultar no limite inferior e superior do intervalo de confiança
    for i in range(exp_num):
        plot_kwargs = {'fmt':'o'}
        if gray:
            plot_kwargs['c'] = 'gray'
        if eb_width:
            plot_kwargs['elinewidth'] = eb_width
        if legend:
            plot_kwargs['label'] = exp_labels[i]

        this_x = x - (exp_num*width)/2 + i*width
        plots += [ax.errorbar(this_x, values[i], err_bars[i], **plot_kwargs)]

    ax.set_ylabel('Acurácias')
    ax.set_title('Acurácia média de experimentos agrupada por classe')
    ax.set_xticks(x, labels, rotation=45, ha='right')

    ax.legend()

    fig.tight_layout()

    if show:
        plt.show()

def get_compare_side_err_barr_data(arpo, depth, metric_func, get_label_func):
    exps = []
    exp_names = []
    for e in arpo.exp_gen(depth):
        exps += [[metric_func(e.confusion_matrixes, l) for l in get_label_func(arpo)]]
        exp_names += [e.name]

    exps = np.array(exps)

    return exps[:,:,0], exps[:,:,1:].transpose(0,2,1), exp_names


def plot_compare_2_set_of_exps(arpo, depth, metric_func, label_func):
    """
    DEPTH is used to get the second set of experiments,
    ARPO must have 2*DEPTH experiments
    The gray plot will be the experiments deepest in ARPO
    """

    for e in arpo.exp_gen(depth+1):
        arpo_g = e

    fig, ax = plt.subplots()

    def this_get_data(x):
        return get_compare_side_err_barr_data(x, depth, metric_func, label_func)

    plot_compare_side_err_bar(arpo_g, this_get_data, show=False, gray=True, eb_width=2., legend=False, fig_ax=(fig, ax))
    plot_compare_side_err_bar(arpo,   this_get_data, show=False, eb_width=1.,               fig_ax=(fig, ax))

    plt.show()

    
    
