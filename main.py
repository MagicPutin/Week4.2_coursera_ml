import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

data = pd.read_csv('Data/close_prices.csv')
Dow_jones_data = pd.read_csv('Data/djia_index.csv')

# 1st half of the task
x = data.loc[:, data.columns != 'date']
y = data['date']

pca = PCA(n_components=10)
pca.fit(x)

sum, count = 0, 0

for i in sorted(pca.explained_variance_ratio_, reverse=True):
    if sum < 0.9:
        count += 1
        sum += i

with open('Answers/task1.txt', 'w') as task:
    task.write(str(count))

# 2nd half
component = []
transformed_data = pca.transform(x)

for i in range(transformed_data.shape[0]):
    component.append(transformed_data[i][0])

y = Dow_jones_data['^DJI']
a = round(np.corrcoef(component, y)[0][1], 2)

with open('Answers/task2.txt', 'w') as ans:
    ans.write(str(a))

maximum = max(pca.components_[0])
count = 0
for i in pca.components_[0]:
    count += 1
    if i == maximum:
        true_count = count

print(data.axes[1][true_count] + ' -> ' + 'Visa')




