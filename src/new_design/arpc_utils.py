import pandas as pd

def aip_gen(df, select=[]):
    """
    Generator for activity, intensity and participant pandas DataFrame
    Specific classes can be selected passing array of tuples with Participant,
    Activity and Intensity as strings in this order
    """
    atividades, intensidades, participantes = [], [], []

    # TODO : Deixar essa parte genérica para quaisquer nomes de colunas
    try:
        atividades    = df['atividade'   ].unique()
        intensidades  = df['intensidade' ].unique()
        participantes = df['participante'].unique()
    except KeyError as err:
        print('Uma das colunas do dataframe está com nome errado?')
        print('Na função aip_gen'                                 )
        print('Coluna que deu errado: ', err.args[0]              )

    if len(select) > 0:
        for s in select:
            yield df.loc[(df['atividade']    == s[1]) &
                         (df['intensidade']  == s[2]) &
                         (df['participante'] == s[0])]
        return

    for a in atividades:
        for i in intensidades:
            for p in participantes:
                yield df.loc[(df['atividade']    == a) &
                             (df['intensidade']  == i) &
                             (df['participante'] == p)]

def p_gen(df, select=[]):
    if len(select)==0:
        try:
            participantes = df['Participante'].unique()
        except KeyError as err:
            print('Uma das colunas do dataframe está com nome errado?')
            print('Na função aip_gen')
            print('Coluna que deu errado: ', err.args[0])

        for p in participantes:
            yield df.loc[df['Participante'] == p]
    else:
        for p in select:
            yield df.loc[df['Participante'] == p]

def rem_outliers(df):
    df_new = df.copy()

    eixos = ['x', 'y', 'z']

    for e in eixos:

        Q1 = df_new[e].quantile(0.25)
        Q3 = df_new[e].quantile(0.75)
        iQ = Q3 - Q1

        inf = Q1 - 1.5 * iQ
        sup = Q3 + 1.5 * iQ

        df_new[e] = np.where(
            df_new[e] > sup,
            sup,
            np.where(
                df_new[e] < inf,
                inf,
                df_new[e]
            )
        )

    return df_new

def get_accuracy(cmat):
    result = []

    n_classes = cmat.shape[0]

    for i in range(n_classes):
        n_class       = cmat[i].sum()
        n_right_pred  = cmat.diagonal()[i]
        result       += [n_right_pred / n_class]

    return result
