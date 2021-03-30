import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
from scipy.signal import savgol_filter

def process_file(path):
    marker1 = 'bomb placement'
    marker2 = 'coins collected'
    marker1_elements = []
    marker2_elements = []
    with open(path, 'r') as f: 
        read = f.read()
    for line in read.split('\n'):
        if marker1 in line:
            marker1_elements.append(int(line.split(' ')[-1]))
        if marker2 in line:
            marker2_elements.append(int(line.split(' ')[-1]))
    print("bombs")
    print(sum(marker1_elements))
    print(len(marker1_elements))
    print("coins")
    print(sum(marker2_elements))
    print(len(marker2_elements))
    return marker1_elements, marker2_elements
    


def better_plot(m1,m2, m3,m4):
    window_length = 101
    polyorder = 3
    window_length = 101
    polyorder = 3
    plt.figure(figsize=(18,14))
    #print(len(m3))
    # We change the fontsize of minor ticks label 
    yhat1 = savgol_filter(m1, window_length, polyorder) 
    yhat2 = savgol_filter(m3, window_length, polyorder) 
    x=[i for i in range(len(m3))]
    
    # , label = "Average baseline"
    plt.plot(x, yhat1, label='No crates')
    plt.plot(x, yhat2, label='Crates')
    plt.legend(loc="upper right", fontsize=15)
    plt.xlabel("Games", fontsize=20)
    plt.xticks(np.arange(0, 1500, step=300), rotation=45, fontsize=15)
    plt.yticks(np.arange(0, 80, step=20), rotation=0, fontsize=15)

    plt.ylabel("Bombs",fontsize=20)
    plt.title("Number of bombs placed with crates", fontsize=25)
    plt.gcf().subplots_adjust(bottom=0.15)

    plt.show()

def plot(l1, l2):
    plt.style.use('seaborn-whitegrid')

    x=[i for i in range(len(l1))]
    y=l1
    #print(x)
    #print(y)
    #plt.scatter(x,y)
    #plt.plot(x,y)
    #plt.xticks(fontsize=20)
    #plt.xticks(fontsize=20)
    
    plt.title("Number of bombs placed", fontsize=15)
    #plt.xticks(x, fontsize = 15)
    #plt.yticks(y, fontsize = 15)
    plt.xlabel("Games", fontsize=15)
    plt.ylabel("Bombs", fontsize=15)
    #plt.xticks(np.arange(x), x, rotation=45, fontsize=40)
    #plt.yticks(np.arange(-0.015, 0.018, step=0.005), rotation=0, fontsize=40)
    #plt.show()
    #figure.tight_layout()
    plt.plot(x, y, 'o', color='blue', markersize=2,zorder=1)
    plt.show()
            
def main(): 
    path1 = 'log_for_plots/gandhi_escape.log'
    path2 = 'log_for_plots/gandhi_escape_0.75.log'
    #path2 = 'log_for_plots/new_bomb_runaway.log'
    #path2 = 'log_for_plots/placement_ghandi_0.75.log'

    #path = 'log_for_plots/placement_ghandi_0.75.log'
    #path = 'log_for_plots/new_gandhi.log'
    #path1 = 'log_for_plots/gandhi_crates.log'
    #path2= 'log_for_plots/gandhi_no_crates.log'
    path = 'agent_code/large_gandhi/logs/large_gandhi.log'
    _,_ = process_file(path)
    #m1, m2 = process_file(path1) 
    #m3, m4 = process_file(path2) 
    #print(len(m1))
    #print(len(m3))
    #print(len(m4))
    #better_plot(m1, m2, m3,m4)
    #plot(m1, m2)

if __name__ == "__main__":
    main()