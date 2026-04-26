import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import moral_foundations_config


def main():
    ################################################
    # Get data
    ################################################
    RAW_A = np.loadtxt('data/plain/current/2000/data_raw_A.txt',  delimiter=' ')

    # V is the basis for traits
    V_matrix = np.loadtxt('data/plain/current/2000/data_archetype_space_V.txt',  delimiter=' ')

    ################################################
    # Get all trait locations in archetype space
    ################################################

    # traits X chars (where V is the basis for traits)
    CONFIGS = np.matmul(RAW_A, V_matrix)
    
    # get first two components to plot together
    hero_fool = CONFIGS[:,0]
    demon_angel = CONFIGS[:,1]

    fig, ax = plt.subplots()
    ax.scatter(hero_fool, demon_angel, c='lightgray', label="All other traits")

    ################################################
    # Get moral foundation configs in archetype space
    ################################################

    color_map = {
    "Care": "orange",
    "Fairness": "deepskyblue",
    "Loyalty": "blue",
    "Authority": "firebrick",
    "Sanctity": "orchid",
    "Liberty": "seagreen"
    }

    for key, value_dict in moral_foundations_config.FOUNDATIONS.items():
        color = color_map[key]
        x = []
        y = []
        print(value_dict)
        for idx, valence in value_dict.items():
            x.append(CONFIGS[idx - 1,0] * valence)
            y.append(CONFIGS[idx - 1,1] * valence)
        ax.scatter(x,y,c=color, label=key)

    # remove bounding box
    plt.gca().set_frame_on(False)

    # remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # axis lines
    ax.axhline(y=0, xmin=-10, xmax=10, color='black')
    ax.axvline(x=0, ymin=-10, ymax=10, color='black')

    # axis labels
    ax.annotate('Hero', (13, -0.5), color='black', fontsize='medium', fontweight='bold')
    ax.annotate('Fool', (-13, -0.5), color='black', fontsize='medium', fontweight='bold')
    ax.annotate('Demon', (0, 13), color='black', fontsize='medium', fontweight='bold')
    ax.annotate('Angel', (0, -13), color='black', fontsize='medium', fontweight='bold')

    # set limits
    ax.set_xlim(-14,14)
    ax.set_ylim(-14,14)

    # show legend
    plt.legend(loc=(0,0))
    plt.show()

if __name__=="__main__":
    main()