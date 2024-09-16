import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

## Plot performance for tower of hanoi starting from random intial states (but terminal)
## Load data
seed_indx = 1 
file_indx = 'RandState' 
# Create directory to store results
file_dir = os.path.dirname(os.path.abspath(__file__))
file_dir = os.path.join(file_dir,str(seed_indx),file_indx)

label_1 = 'Muzero'
label_2 = 'ResetLatentPol'
label_3 = 'ResetLatentVal'
lable_4 = 'ResetLatentRwd'
label_5 = 'ResetLatentVal_ResetLatentRwd'

labels = [label_1, label_2, label_3, lable_4, label_5]
results = []

for l in labels:
    results.append(torch.load(os.path.join(file_dir,l+'_actingAccuracy.pt')).numpy())

results = np.array(results)


font_s =7
mpl.rc('font', size=font_s)
plt.rcParams["font.family"] = "helvetica"
mpl.rcParams['xtick.labelsize'] = font_s 
mpl.rcParams['ytick.labelsize'] = font_s 

fig, axs = plt.subplots(nrows=1, ncols=len(labels), figsize=(7.5,4),
 gridspec_kw={'wspace': 0.32, 'hspace': 0.3, 'left': 0.05, 'right': 0.97, 'bottom': 0.15,
                                               'top': 0.95})
i=0
for d in results:
    axs[i].plot(d[:,0],d[:,1])
    axs[i].set_ylim([0, 70])
    axs[i].set_title(labels[i],fontsize=font_s)
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['top'].set_visible(False)
    axs[i].set_xlabel('N. simulations every real step \n (planning time)')
    if i == 0:
        axs[i].set_ylabel('Error')
    i+=1
plt.show()
#plt.savefig('/Users/px19783/Desktop/CerebLatentPlanning_accuracies', format='png', dpi=1200)
