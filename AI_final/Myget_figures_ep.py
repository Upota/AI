
import matplotlib.pyplot as plt
import random
import pickle


for num in [0.0,0.3,0.5,0.8,1.0]:
    fname = './' + ('epsilon=%.1f'%(num)) + '.pkl'

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
    outname = 'epsilon=%.1f'%(num) + '.png'
    fig.savefig(outname)