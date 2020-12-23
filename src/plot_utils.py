import seaborn as sns
import matplotlib.pyplot as plt

from pandas import DataFrame
from itertools import combinations_with_replacement, combinations
from numpy import zeros_like, triu_indices_from, nan
from husl import hex_to_husl
from matplotlib.colors import LinearSegmentedColormap

COLOURS = ['#BB5566',  '#004488', '#DDAA33', '#000000']
MARKERS = ['o', 'X', 'D', 'P', 'X']
cpalette = sns.color_palette(COLOURS)
cmap = LinearSegmentedColormap.from_list('high_contrast', COLOURS[:-1])
cmap_sequential = sns.light_palette(hex_to_husl(COLOURS[1]), input="husl", as_cmap=True)

FSIZELABELS = 18
FSIZETICKS = 16

def set_style():
    sns.set_palette(cpalette)
    sns.set_style('whitegrid', {'axes.spines.right': True,
                                'axes.spines.top': True,
                                'axes.edgecolor': 'k',
                                'xtick.color': 'k',
                                'ytick.color': 'k',
                                'grid.color':'0.7',
                                'font.family': 'serif',
                                'font.sans-serif': 'cm',
                                'text.usetex': True})

def plt_false_negatives(pred_fixed_key, pred_rot_key, model_names,
                        x='Test Set', hue='Cipher',
                        x_order=['Fixed Key', 'Rotate Key'], hue_filter=['caesar', 'columnar', 'substitution', 'vigenere']):

    fig, axis = plt.subplots(1, 2, figsize=(15,4), sharey='all', sharex='all')


    ### Plot model trained on fixed key
    ax = axis[0]

    false_negatives = pred_fixed_key[pred_fixed_key['Prediction Correct'] == False]
    pltdata = false_negatives.groupby([x, hue]).size()/pred_fixed_key.groupby([x, hue]).size()
    pltdata.name = 'Error Rate'
    pltdata = pltdata * 100
    pltdata = pltdata.reset_index()
    pltdata = pltdata[pltdata[hue].isin(hue_filter)]

    sns.barplot(x=x, hue=hue, y='Error Rate',
                data=pltdata, order=x_order, ax=ax)

    ax.legend(loc='upper center', bbox_to_anchor=(1, -0.1),
              ncol=4, fontsize=FSIZELABELS)

    # Formatting
    ax.set_title(model_names[0], fontsize=FSIZETICKS)

    # y-axis
    ax.set_ylabel('% False Negatives', fontsize=FSIZELABELS)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(FSIZETICKS)

    # x-axis
    ax.set_xlabel('')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(FSIZELABELS)

    ### Plot model trained on rotating key
    ax = axis[1]

    false_negatives = pred_rot_key[pred_rot_key['Prediction Correct'] == False]
    pltdata = false_negatives.groupby([x, hue]).size()/pred_rot_key.groupby([x, hue]).size()
    pltdata.name = 'Error Rate'
    pltdata = pltdata * 100
    pltdata = pltdata.reset_index()
    pltdata = pltdata[pltdata[hue].isin(hue_filter)]

    sns.barplot(x=x, hue=hue, y='Error Rate',
                data=pltdata, order=x_order, ax=ax)
    ax.get_legend().remove()

    # Formatting
    ax.set_title(model_names[1], fontsize=FSIZETICKS)

    # y-axis
    ax.set_ylabel('')

    # x-axis
    ax.set_xlabel('')
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(FSIZETICKS)

    ### Format figure
    fig.subplots_adjust(wspace=0.02)

    return fig


def plt_error(predictions, model_name):
    misclassified = predictions[predictions['Prediction Correct'] == False]

    fig, axis = plt.subplots(1, 2, figsize=(14, 4))

    ### False negatives
    ax = axis[0]
    pltdata = misclassified.groupby('Label').size()/predictions.groupby('Label').size()
    pltdata.name = 'Error Rate'
    pltdata = pltdata * 100
    pltdata = pltdata.reset_index()

    sns.barplot(x='Label', y='Error Rate',
                data=pltdata, ax=ax)

    ax.set_title(model_name, fontsize=FSIZELABELS)
    ax.set_ylabel('% Misclassified', fontsize=FSIZELABELS)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(FSIZETICKS)

    ax.set_xlabel('Label', fontsize=FSIZELABELS)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(FSIZETICKS)

    ### Confusion matrix
    ax = axis[1]
    # Get group combinations
    categories = ['cipher', 'manual', 'sensor']

    # Compute confusion matrix
    counts = misclassified.groupby(['Label', 'Prediction']).size()
    confusion = counts.to_frame().pivot_table(index='Prediction', columns='Label', values=0).fillna(0).astype(int)
    for c in categories:
        if c not in list(confusion):
            confusion[c] = 0


    sns.heatmap(confusion,
                square=True, annot=True,
                ax=ax, cmap=cmap_sequential,
                annot_kws={'size':FSIZETICKS, 'weight': 'bold'},  fmt='d')
    ax.set_title('Confusion matrix', fontsize=FSIZELABELS)
    ax.set_xlabel('Label', fontsize=FSIZELABELS)
    ax.set_xticklabels([x for x in ax.get_xticklabels()], fontsize=FSIZETICKS, rotation=0, ha='center')

    ax.set_ylabel('Prediction', fontsize=FSIZELABELS)
    ax.set_yticklabels([x for x in ax.get_yticklabels()], fontsize=FSIZETICKS, rotation=45, ha='right')

    return fig

def plt_character_dist(baseline_model):
    fig, ax = plt.subplots()

    for l, counts in baseline_model.frequency_counts.items():
        alphabet = counts.keys()
        x = [i for i in range(len(alphabet))]

        marginal = counts.values()
        ax.bar(x, marginal, width=.8, label=l)

    ax.legend(fontsize=FSIZETICKS)
    ax.set_title('Character probabilities', fontsize=FSIZELABELS)
    ax.set_xlabel('Character index', fontsize=FSIZELABELS)
    ax.set_ylabel('P[c | y]', fontsize=FSIZELABELS)

    return fig
