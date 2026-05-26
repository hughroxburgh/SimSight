import numpy as np
import matplotlib.pyplot as plt

def _Get_Colours(num):
    """
    Used for colouring sightlines blue -> red.
    """

    if num < 3 :
        colours = ['blue','red']
    elif num == 3:
        colours = ['blue','forestgreen','red']
    elif num == 4:
        colours = ['blue','forestgreen','tab:orange','red']
    else:
        cmap = plt.cm.jet  
        cmaplist = np.array([cmap(i) for i in range(cmap.N)])
        colours = cmaplist[np.linspace(30,len(cmaplist)-30,num).astype(int)]
    return colours