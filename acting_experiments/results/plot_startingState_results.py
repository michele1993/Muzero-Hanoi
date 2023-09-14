import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


## Plot performance for tower of hanoi starting from pre-determined states at a fixed distance from the target
# e.g., ES: far from goal, MS: moderate, LS: close to goal.
## Load data

# Create directory to store results
root_dir = os.path.dirname(os.path.abspath(__file__))
seed_indx = 1 
root_dir = os.path.join(root_dir,str(seed_indx))

dir_1 = 'ES'
dir_2 = 'MS'
dir_3 = 'LS'

directories = [dir_1, dir_2, dir_3]

label_1 = 'Muzero'
label_2 = 'ResetLatentPol'
label_3 = 'ResetLatentVal'
lable_4 = 'ResetLatentRwd'
label_5 = 'ResetLatentVal_ResetLatentRwd'

labels = [label_1, label_2, label_3, lable_4, label_5]

# Init figure
font_s =7
mpl.rc('font', size=font_s)
plt.rcParams["font.family"] = "helvetica"
mpl.rcParams['xtick.labelsize'] = font_s 
mpl.rcParams['ytick.labelsize'] = font_s 

fig, axs = plt.subplots(nrows=len(directories), ncols=len(labels), figsize=(7.5,4),
 gridspec_kw={'wspace': 0.32, 'hspace': 0.3, 'left': 0.065, 'right': 0.97, 'bottom': 0.15,
                                               'top': 0.95})

# Iterate through directories to plot each row with different ablations
e = 0
for d in directories:
    results = []
    file_dir = os.path.join(root_dir,d)
    for l in labels:
        results.append(torch.load(os.path.join(file_dir,l+'_actingAccuracy.pt')).numpy())

    results = np.array(results)

    i=0
    for r in results:
        axs[e,i].plot(r[:,0],r[:,1])
        #axs[e,i].set_ylim([0, 70])
        axs[e,i].spines['right'].set_visible(False)
        axs[e,i].spines['top'].set_visible(False)
        if i == 0:
            axs[e,i].set_ylabel('Error')
        if e == 0:
            axs[e,i].set_title(labels[i],fontsize=font_s)
        if e == len(directories) -1:
            axs[e,i].set_xlabel('N. simulations every real step \n (planning time)')
        i+=1
    e+=1
plt.show()
#plt.savefig('/Users/px19783/Desktop/CerebLatentPlanning_StartingStates_accuracies', format='png', dpi=1200)
