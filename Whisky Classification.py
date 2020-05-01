import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# In the text files, the entries on each row are seperated by commas. These type of files are known as "comma seperated files" or as "CSV" files.
whisky = pd.read_csv("whiskies.txt")    # To read CSV file we will use this function.
whisky["Region"] = pd.read_csv("region.txt")

# Now we will use corrleation using pandas module.
corr_flavors = pd.DataFrame.corr(flavors)
print(corr_flavors)

# We will plot the obtained data on the graph.
plt.figure(figsize=(10,10))
plt.pcolor(corr_flavors)
plt.colorbar()
plt.savefig("corr_flavors.pdf") # We are saving this in pdf format.

# We can also look correlation among whiskies across flavor.
corr_whisky = pd.DataFrame.corr(flavors.transpose())
plt.figure(figsize=(10,10))
plt.pcolor(corr_whisky)
plt.axis("tight")
plt.colorbar()
plt.savefig("corr_whisky.pdf") # We are saving this in pdf format.

# Now we will do clustering of whiskies by flavor profile.
# We had done this clustering sklearn machine learning module.
from sklearn.cluster.bicluster import SpectralCoclustering
model = SpectralCoclustering(n_clusters = 6, random_state = 0)
model.fit(corr_whisky)
model.rows_

# We have now reshuffled rows and columns.
whisky = whisky.ix[np.argsort(model.row_labels_)]
whisky = whisky.reset_index(drop=True)

# Now we gona recalculate correlation matrix.
correlations = pd.DataFrame.corr(whisky.iloc[:,2:14].transpose())

# Now we will convert this into numpy array.
correlations = np.array(correlations)

# We are ploting the obtained data on the graph.
plt.figure(figsize = (14,7))
plt.subplot(121)
plt.pcolor(corr_whisky)
plt.title(corr_whisky)
plt.title("Original")
plt.axis("tight")
plt.subplot(122)
plt.pcolor(correlations)
plt.title("Rearranged")
plt.axis("tight")                   # This property is used when there is some empty space left on graph.
plt.savefig("correlations.pdf")     # Saving the graph in pdf format to the local computer.
