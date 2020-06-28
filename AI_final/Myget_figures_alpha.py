import matplotlib.pyplot as plt
import pickle
import numpy as np

num = [0.0,0.1,0.3,0.5,0.8,1.0]
place = 111
fig = plt.figure(figsize=(13,10))
color = ['lightpink', 'orange', 'seagreen', 'royalblue', 'skyblue', 'r']

for i in range(6):
    fname = './alpha/' + ('alpha=%.1f'%(num[i])) + '.pkl'
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    keys = list(data.keys())
 
    y = data[keys[0]]
    plt.plot(y,color[i],label='a=%.1f'%(num[i]))

plt.xlabel('# of episodes')
plt.ylabel('avgQV')
plt.title('Average of Q-values with various alpha')
plt.legend(loc='lower right')

outname = 'alpha.png'
fig.savefig(outname)