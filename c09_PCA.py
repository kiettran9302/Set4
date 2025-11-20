""""
Foundations of Pattern Recognition and Machine Learning 2nd edition
Chapter 9 Figure 9.5
Author: Ulisses Braga-Neto
This code is distributed under the GNU LGPL license

PCA example using the softt magnetic alloy dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler as ssc

# Fix random state for reproducibility
np.random.seed(0)

SMA = pd.read_csv('./Soft_Magnetic_Alloy_Dataset.csv')

fn0 = SMA.columns[0:26]         # all feature names
fv0 = SMA.values[:,0:26]        # all feature values
rs0 = SMA['Coercivity (A/m)']   # select response

# pre-process the data
n_orig   = fv0.shape[0]                 # original number of training points
p_orig   = np.sum(fv0>0,axis=0)/n_orig  # fraction of nonzero components for each feature
noMS     = p_orig>0.05
fv1      = fv0[:,noMS]		            # drop features with less than 5% nonzero components
noNA     = np.invert(np.isnan(rs0))     # find available response values
SMA_feat = fv1[noNA,:]                  # filtered feature values
SMA_fnam = fn0[noMS]                    # filtered feature names
SMA_resp = rs0[noNA]                    # filtered response values

n,d = SMA_feat.shape # filtered data dimensions

# add random perturbation to the features
sg = 2
SMA_feat_ns = SMA_feat + np.random.normal(0,sg,[n,d])
SMA_feat_ns = (SMA_feat_ns + abs(SMA_feat_ns))/2 # clamp values at zero

# standardize data
SMA_feat_std = ssc().fit_transform(SMA_feat_ns)

# compute PCA
pca = PCA()
pr = pca.fit_transform(SMA_feat_std)

# PCA plots
def plot_PCA(X,Y,resp,thrs,nam1,nam2):
    Ihigh = resp>thrs[1]
    Imid = (resp>thrs[0])&(resp<=thrs[1])
    Ilow = resp<=thrs[0]  
    plt.style.use('seaborn-v0_8-deep')
    plt.xlabel(nam1+' principal component',fontsize=16)
    plt.ylabel(nam2+' principal component',fontsize=16)
    plt.scatter(X[Ilow],Y[Ilow],c='blue',s=16,marker='o',label='Low')
    plt.scatter(X[Imid],Y[Imid],c='green',s=16,marker='o',label='Mid')
    plt.scatter(X[Ihigh],Y[Ihigh],c='red',s=16,marker='o',label='High')
    plt.xticks(size='medium')
    plt.yticks(size='medium')
    plt.legend(fontsize=14,facecolor='white',markerscale=2,markerfirst=False,handletextpad=0)
    plt.show()

fig=plt.figure(figsize=(8,8))#,dpi=150)
plot_PCA(pr[:,0],pr[:,1],SMA_resp,[2,8],'First','Second')
# fig.savefig('c09_PCA-a.png',bbox_inches="tight",facecolor="white")
fig=plt.figure(figsize=(8,8))#,dpi=150)
plot_PCA(pr[:,0],pr[:,2],SMA_resp,[2,8],'First','Third')
# fig.savefig('c09_PCA-b.png',bbox_inches="tight",facecolor="white")
fig=plt.figure(figsize=(8,8))#,dpi=150)
plot_PCA(pr[:,1],pr[:,2],SMA_resp,[2,8],'Second','Third')
# fig.savefig('c09_PCA-c.png',bbox_inches="tight",facecolor="white")

# --- a --- 
# Plot the percentage of variance explained by each PC as a function of PC number. This
# is called the scree plot. Now plot the cumulative percentage of variance explained by
# the PCs as a function of PC number. How many PCs are needed to explain 95% of
# the variance? Hint: use the attribute explained variance ratio and the cusum()
# method.

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)
plt.figure(figsize=(10, 5))

# Scree Plot
plt.subplot(1, 2, 1)
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio * 100, 'bo-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('% Variance Explained')
plt.grid(True)

# Cumulative Variance Plot
plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance * 100, 'rs-', linewidth=2)
plt.axhline(y=95, color='k', linestyle='--', label='95% Threshold')
plt.title('Cumulative Variance Explained')
plt.xlabel('Principal Component')
plt.ylabel('Cumulative % Variance')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Check how many PCs are needed for 95% variance [cite: 22]
n_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"\nNumber of PCs needed to explain 95% variance: {n_95}")

#  -- b ---
# Print the loading matrix W (this is the matrix of eigenvectors, ordered by PC number
# from left to right). The absolute value of the coefficients indicate the relative importance
# of each original variable (row of W) in the corresponding PC (column of W)

loading_matrix = pca.components_.T
loading_df = pd.DataFrame(loading_matrix, index=SMA_fnam, columns=[f'PC{i+1}' for i in range(loading_matrix.shape[1])])
print("Loading Matrix (W):")
print(loading_df)   

# -- c ---
# Identify which two features contribute the most to the discriminating first PC and plot
# the data using these top two features. What can you conclude about the effect of these
# two features on the coercivity? This is an application of PCA to feature selection.

pc1_loadings = loading_matrix[:, 0]
top2_indices = np.argsort(np.abs(pc1_loadings))[-2:]
top2_features = SMA_fnam[top2_indices]
print(f"\nTop 2 features contributing to the first PC: {top2_features.tolist()}")