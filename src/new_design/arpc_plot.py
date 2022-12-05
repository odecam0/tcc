import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def plot_all(df, save=False, participantes=[]):
    # atividades e intensidades foram declarados no início do arquivo
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
