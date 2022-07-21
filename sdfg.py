
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram,linkage
from scipy.cluster.hierarchy import cut_tree

fil=pd.read_csv("Country-data.csv")

fil.head()
fil=fil.drop(columns="country")

fil.head()
scaler = StandardScaler()
scaled = scaler.fit_transform(fil)
new_fil=pd.DataFrame(scaled,columns=fil.columns)
new_fil

link = linkage(new_fil,method='complete')
plt.figure(figsize=(10,7))
dendrogram(link)
plt.show()
new_fil["hierarchical_cluster_labels"]=cut_tree(link,n_clusters=4)
new_fil

from sklearn.decomposition import PCA
pca = PCA(n_components=4)

df_pca = pca.fit_transform(new_fil)
DF_PCA=pd.DataFrame(df_pca,columns=["PC1","PC2","PC3","PC4"])
DF_PCA["hierarchical_cluster_labels"]=cut_tree(link,n_clusters=4)
DF_PCA
import seaborn as sns
sns.scatterplot(x='gdpp',y='child_mort', hue='hierarchical_cluster_labels',data=new_fil).set(title='How Low GDP Rate Correspons to the Child Mortality Rate')
