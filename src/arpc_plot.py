import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import re

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

def group_labels_by_first_word(labels):
    """
    Receives a simples list of labels and return them grouped and separated by the
    first word in the label, considering word written in camel case. LikeThisWordHere.
    Example:
    Input -> [ label0, ..., labelN ]
    Output \/
    {
      'FirstWord0': ['RestOfLabel0', ..., 'RestOfLabelM_1'],
       ...,
      'FirstWordN': ['RestOfLabel0', ..., 'RestOfLabelM_k']
    }
    """
    list_of_all_first_words = []
    for label in labels:
        first_word_match = re.search('[A-Z][a-z]+', label)
        first_word       = first_word_match.group()
        if first_word not in list_of_all_first_words:
            list_of_all_first_words += [first_word]
    
    # Initializing each group as an enpty list in the dictionary
    groups = {}
    for word in list_of_all_first_words:
        groups[word] = []
        
    # Filling each group in the dictionary
    for label in labels:
        first_word_match = re.search('[A-Z][a-z]+', label)
        first_word       = first_word_match.group()
        rest_of_label    = re.sub(first_word, '', label)
        groups[first_word] += [rest_of_label]
    
    return groups

def plot_compare_err_bar(arpo, get_data_proc, grouping_scheme_proc,
                         show=True, save_file_name=None, gray=False,
                         eb_width=1, legend=True, fig_gs=None,
                         no_background=False):
    """
    GET_DATA_PROC is a procedure receives an Arpc object as parameter and returns a 3-tuple with
    an array of means of some metric for each label, array of superior and inferior limits of error bar,
    and and array of experiment names. Like this:
        ( [ mean0, ... , meanN] ,
        [ [inf0, ..., infN], [sup0, ..., supN] ],
        [ <exp_name0, ..., exp_nameM>] )
    One such procedure is arpc_plot.get_compare_side_err_barr_data defined in this same module.

    LABEL_GROUPING_PROC will receive a simple list of labels and will return a list of lists with the
    same labels grouped. For example:
        Input --> [ label0, ..., label8]
        Output \/
    {
        'FirstWord0': ['RestOfLabel0', ..., 'RestOfLabelM_1'],
        ...,
        'FirstWordN': ['RestOfLabel0', ..., 'RestOfLabelM_k']
    }
    FIG_GS is a tuple like (<figure>, <gridspec>)
    RETURN_SUBGRIDS and SUB_GRIDS_PARAMS make it possible to acess the subgrids from other calls of this function
    """
    # Experiments data
    values, err_bars, exp_labels = get_data_proc(arpo)

    labels = arpo.confusion_matrixes[0][1]
    grouped_labels = grouping_scheme_proc(labels)

    exp_nums = values.shape[0]

    if not fig_gs:
        fig = plt.figure(figsize=(12,6))
        gs = plt.GridSpec(1, len(grouped_labels), figure=fig)
    else:
        fig, gs = fig_gs

    colors_error_bar = list(plt.cm.Dark2(np.linspace(0, 1, exp_nums)))

    # This loop will create a sub gridspec for each label group to make members of the group closer
    # for each gridspec created, it will create its children subplot, one for each full label of the group
    # and for each subplot, a plot will be made for each experiment. These are the 3 nested loops.
    for i, group in enumerate(grouped_labels):
        sub_gs = gs[0, i].subgridspec(1, len(grouped_labels[group]))

        # Adding title to each gridspec that separates a group
        ax = fig.add_subplot(sub_gs[:])
        ax.patch.set_alpha(0)
        ax.axis('off')
        ax.set_title(group, fontsize=15)

        for j, label in enumerate(grouped_labels[group]):

            ax = fig.add_subplot(sub_gs[0, j])
            if no_background:
                ax.patch.set_alpha(0)

            ax.spines[['right', 'top', 'left']].set_visible(False)
            ax.set_yticks([])
            ax.set_xticks([0.5], [label], fontsize=12)
            ax.set_xlim(-0.5, 1.5)
            ax.set_ylim(-0.05, 1)

            for exp in range(exp_nums):
                # To get space between the experiments in each 
                x = exp * (1/ (exp_nums - 1))

                # values comes as an array with 1 array for each experiment
                # each experiment array comes with <number of labels> floating numbers.
                # Like this:
                #  [ [exp0_value0, ... , exp0_valueN], ... , [expM_value0, ... , expM_valueN] ]
                # And we want to plot the (i * exp_nums + j)th value in the j_th axs of the i_th gridspec
                this_index = i * exp_nums + j
                this_value = values[exp, this_index]
                this_yerr = np.reshape(err_bars[exp, :, this_index], (2, 1))

                if gray:
                    c = 'gray'
                else:
                    c = colors_error_bar[exp]

                ax.errorbar(x, this_value, this_yerr, c=c, fmt='o', label=exp_labels[exp],
                            linewidth=eb_width)

    # Calling .legend only on the last axis.
    if legend:
        ax.legend() 


    # Here, as the final step, we create a subplot that covers all the parent gridspec, and use
    # it to set a title for the whole figure, and draw the left ticks of the entire figure, that
    # are colored based on how good is the accuracy, also drawing an horizontal line into the whole
    # figure, also colored.
    ax = fig.add_subplot(gs[:], facecolor=None) 

    # Create colormap from red to green, going through orange. And get 5 colors corresponding to
    # 0%, 25%, 50%, 75%, 100% of accuracy.
    red_to_green = LinearSegmentedColormap.from_list('GreenToRed', ['red', 'orange', 'green'])
    percentage_values = np.linspace(0, 1, 5, endpoint=True)
    red_to_green_values = red_to_green(percentage_values)

    [t.set_visible(False) for t in ax.yaxis.get_ticklines()]
    [t.set_color(red_to_green_values[i%5]) for i, t in enumerate(ax.yaxis.get_ticklabels())]
    [ax.axhline(y=y, color=c, linestyle='--', alpha=0.5, linewidth=0.7)
     for y, c in zip(percentage_values, red_to_green_values)]

    #             \/ Here the Title is hardcoded, it should be changed. !TODO!
    ax.set_title("Acurácia média de diferentes modelos por atividade e intensidades", y=1.08)
    ax.set_ylim(-0.05, 1) # Has to be the same as the previous subplots
    ax.set_yticks(percentage_values, ['$'+str(int(i*100))+'\%$' for i in percentage_values], fontweight=0.5)
    ax.set_xticks([])
    ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
    ax.patch.set_alpha(0) # transparent background

    if show:
        plt.show()

    if save_file_name:
        plt.savefig(save_file_name)

def get_compare_side_err_barr_data(arpo, depth, metric_func, get_label_func):
    """
    For each last DEPTH experiments in ARPO, a METRIC_FUNCtion is used to get
    a metric from the confusion matrixes of the experiment, separated by labels
    returned by GET_LABEL_FUNC.

    3-tuple is returned with an array of means of some metric for each label,
    array of superior and inferior limits of error bar, and and array of labels.

    Signature of METRIC_FUNC must be METRIC_FUNC(cms, label)
    where cms is an array with multiple confusion matrixes, and label
    is one of the possible labels within the confusion matrixes.
    It must return a 3-tuple with (mean, inferior_limit, superior_limit)
    inferior and superior limits being the amount to be added and subtracted
    to the mean to get the error bar.

    METRIC_FUNC can be, for an example: arpc_metrics.get_label_accuracy_mean
     (find-fline "./arpc_metrics.py" "def get_label_accuracy_mean(cms, label)")
    """
    exp_names = []

    exps = []
    # exps is built calculating metrics using METRIC_FUNC for each label and
    # putting its values into the array, taking the form of:
    #
    # [ [[ mean0, inf0, sup0 ], ..., [meanN, infN, supN]]   # <- First experiment
    #                          . . .
    #   [[ mean0, inf0, sup0 ], ..., [meanN, infN, supN]] ] # <- Last experiment
    for e in arpo.exp_gen(depth):
        exps += [[metric_func(e.confusion_matrixes, l) for l in get_label_func(arpo)]]
        exp_names += [e.name]

    # then numpy is used to rearrange the data in exp
    exps = np.array(exps)
    mean_values = exps[:,:,0]
    inf_sup_values = exps[:,:,1:].transpose(0,2,1)
    return mean_values, inf_sup_values, exp_names


def plot_compare_2_set_of_exps(arpo, depth, metric_func, label_func, file_name=None, show=True):
    """
    DEPTH is used to get the second set of experiments,
    ARPO must have 2*DEPTH experiments
    The gray plot will be the experiments deepest in ARPO
    """

    for e in arpo.exp_gen(depth+1):
        arpo_g = e

    fig = plt.figure(figsize=(12,6))
    gs = plt.GridSpec(1, 3, figure=fig)

    def this_get_data(x):
        return get_compare_side_err_barr_data(x, depth, metric_func, label_func)

    plot_compare_err_bar(arpo_g, this_get_data, group_labels_by_first_word,
                         show=False, gray=True, eb_width=2., legend=False, fig_gs=(fig,gs))
    plot_compare_err_bar(arpo,   this_get_data, group_labels_by_first_word,
                         show=False, eb_width=1., no_background=True, fig_gs=(fig,gs))

    if file_name:
        plt.savefig(file_name)
    else:
        plt.show()

    
    
