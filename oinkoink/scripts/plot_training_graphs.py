import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import sys


df = pd.read_pickle(sys.argv[1] + '/8ply.pkl')
df.plot(y='Average loss', title='8ply', legend=False, color='b')
plt.xlabel('Generation')
plt.ylabel('RMSE value loss')
plt.savefig(sys.argv[2] + '/8ply.png')

df = pd.read_pickle(sys.argv[1] + '/7ply.pkl')
fig, ax1 = plt.subplots()
ax = df['Average loss'].plot(ax=ax1, color='b', linewidth=1.0)
ax2 = df['prior Average loss'].plot(secondary_y=True, ax=ax1, color='g')
ax.set_ylabel('RMSE value loss', fontsize=10)
ax.set_xlabel('Generation', fontsize=10)
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, ['value loss', 'policy loss'], loc=0)
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
plt.title('7ply')
plt.ylabel('Cross Entropy policy loss', fontsize=10, rotation=-90, labelpad=15)
plt.savefig(sys.argv[2] + '/7ply.png')
