# %% [markdown]
# # First Model - kNN (iris)

# %%
import imp
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

# %%
# Load the Data
iris = load_iris()
# iris_data,iris_target = load_iris(return_X_y=True)

# %%
for key in iris.keys():
    print(f'Key:{key}')

# %%
# Convert BUNCH to Dataframe
iris_df = pd.DataFrame(iris['data'],columns=iris['feature_names'])
iris_df

# %%
iris_df.info()

# %%
# Add the target variable
iris_df["target"] = iris['target']

# %% [markdown]
# ##### !NOTE - Setting Random State allows to get same result everytime the function is run. 

# %%
# Split Data in train and test
iris_train,iris_test = train_test_split(iris_df,test_size=0.25,random_state=0)

# %%
grr = pd.plotting.scatter_matrix(iris_train.iloc[:,:4], c=iris_train['target'], figsize=(15, 15), marker='o',
hist_kwds={'bins': 20}, s=60, alpha=.8)

# %%
knn = KNeighborsClassifier(n_neighbors=1)
knn

# %%
knn.fit(iris_train.iloc[:,:4],iris_train['target'])

# %%
iris_test['predicted'] = knn.predict(iris_test.iloc[:,:4])

# %%
np.mean(iris_test['predicted']==iris_test['target'])

# %%
knn.score(iris_test.iloc[:,:4],iris_test['target'])

# %% [markdown]
# # Practice kNN - penguins (an alternative to iris)


