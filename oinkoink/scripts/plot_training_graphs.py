import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_pickle('~/Downloads/nn/new_dir14/8ply.pkl')
df.plot(y='Average loss', title='8ply', legend=False, color='b')
plt.xlabel('Generation')
plt.ylabel('RMSE value loss')
plt.savefig('8ply_loss.png')

df = pd.read_pickle('~/Downloads/nn/new_dir14/7ply.pkl')
fig, ax1 = plt.subplots()
ax = df['Average loss'].plot(ax=ax1, color='b', linewidth=1.0)
df['prior Average loss'].plot(secondary_y=True, ax=ax1, color='g')
ax.set_ylabel('RMSE value loss', fontsize=10)
ax.set_xlabel('Generation', fontsize=10)
plt.ylabel('Cross Entropy policy loss', fontsize=10, rotation=-90)
plt.savefig('7ply_loss.png')
