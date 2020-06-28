#!/usr/bin/env python
# coding: utf-8

# In[5]:


import matplotlib.pyplot as plt
import random
import pickle


# In[13]:
for num in [0.0,0.2,0.4,0.6,0.8,1.0]:
    fname = './' + ('alpha=%.1f'%(num)) + '.pkl'

    with open(fname, 'rb') as f:
        data = pickle.load(f)

    fig = plt.figure(figsize=(13,10))
    place = [221]
    color = ['lightpink', 'orange', 'seagreen', 'royalblue', 'skyblue']
    for i, (place,keys) in enumerate(zip(place,data.keys())):
        y = data[keys]
        ax = fig.add_subplot(place)
        ax.plot(y, color=random.choice(color))
        ax.title.set_text(keys)
    outname = 'alpha=%.1f'%(num) + '.png'
    fig.savefig(outname)


# In[ ]:




