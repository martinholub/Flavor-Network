'''Make recommendations based on flavor profile

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

#this function prints out dishes similar to the query, either includding or exclusing dishes from its own cuisine, based on 'similar_cuisine'
def finddishes(yum_ingr2, yum_cos, idx, similar_cuisine=False):
    cuisine = yum_ingr2.iloc[idx]['cuisine']
    print('Dishes similar to', yum_ingr2.ix[idx, 'recipeName'], '('+yum_ingr2.ix[idx, 'cuisine']+')')
    match = yum_ingr2.iloc[yum_cos[idx].argsort()[-21:-1]][::-1]

    if not similar_cuisine:
        submatch = match[match['cuisine']!=cuisine]
    else:
        submatch = match
    for i in submatch.index:
        print(submatch.ix[i, 'recipeName'], '('+submatch.ix[i, 'cuisine']+')','(ID:'+str(i)+')')

def compute_metric(yum_ingrX, yum_cos, idx, alpha = 1.0, beta = 0.75):
    """tba

    Finds recipes that are of different taste but use similar ingredients

    yum_cos: measure of similarity in flavor space
    yum_ingr: binary ingredients matrix (recipes x ingredients)
    """
    yum_cos2 = pairwise_distances(yum_ingrX, metric = "jaccard")
    metric = alpha * yum_cos2[idx] - beta * yum_cos[idx]
    return(metric)

#this function plot top-20 dishes similar to the query
def plot_dishes(yum_ingr, yum_ingrX, yum_cos, idx = 1, num = 20, similar_cuisine = False):
    """
    tba
    """
    #reset index yum_ingr
    yum_ingr2 = yum_ingr.reset_index(drop=True)
    # compute metric
    metric = compute_metric(yum_ingrX, yum_cos, idx)
    # get recipes maximizing metric
    match = yum_ingr2.iloc[metric.argsort(), :]
    # could also drop duplicate recope names, but probably not needed
    # Optionally, drop recipes of the same cuisine
    cuisine = yum_ingr2.iloc[idx]['cuisine']
    if not similar_cuisine:
        match = match[match['cuisine']!=cuisine]
    #match = match[-(num+2):-2][::-1].copy()
    match = match[0:(num+1)].copy()
    newidx = match.index.get_values()
    match.loc[:, 'cosine'] = metric[newidx]
    match.loc[:, 'rank'] = range(1,1+len(newidx))

    xlim = [metric[newidx][1] - 0.1, metric[newidx][-1] + 0.1]

    label1, label2 =[],[]
    for i in match.index:
        label1.append(match.ix[i,'cuisine'])
        label2.append(match.ix[i,'recipeName'])

    fig = plt.figure(figsize=(10,10))
    ax = sns.stripplot(y='rank', x='cosine', data=match, jitter=0.05,
                       hue='cuisine',size=15,orient="h")
    ax.set_title(yum_ingr2.ix[idx,'recipeName']+'('+yum_ingr2.ix[idx,'cuisine']+')',fontsize=18)
    ax.set_xlabel('Recipe score',fontsize=18)
    ax.set_ylabel('Rank',fontsize=18)
    ax.yaxis.grid(color='white')
    ax.xaxis.grid(color='white')

    for label, y,x, in zip(label2, match['rank'],match['cosine']):
         ax.text(x+0.001,y-1,label, ha = 'left')
    ax.legend(loc = 'best',prop={'size':14})
    ax.set_ylim([match.shape[0],-1])
    ax.set_xlim(xlim)

    return match

if __name__ == '__main__':
    yum_ingr = pd.read_pickle('data/yummly_ingr.pkl')
    yum_ingrX = pd.read_pickle('data/yummly_ingrX.pkl')
    yum_tfidf = pd.read_pickle('data/yum_tfidf.pkl')
    #calculate cosine similarity
    yum_cos = cosine_similarity(yum_tfidf)
    #reset index yum_ingr
    yum_ingr2 = yum_ingr.reset_index(drop=True)

    #plot similar dishes for Fettucini Bolognese
    idx = 3900
    xlim = [0.91,1.0]
    plot_dishes(idx,xlim)
    #plot similar dishes for chicken tikka masala
    idx = 3315
    xlim = [0.88,1.02]
    plot_dishes(idx,xlim)
