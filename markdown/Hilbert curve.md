# Hilbert curve

The code is copied from an article [Information bottlenecks and dimensionality reduction in deep learning](https://towardsdatascience.com/information-bottlenecks-c2ee67015065?_branch_match_id=869252424490057064)

```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def hilbert_curve_expansion(str, n):
    k = 0
    while k < len(str):
        if str[k] == 'A':
            str = str[0:k] + '+BF-AFA-FB+' + str[(k+1):]
            k+=10
        elif str[k] == 'B':
            str = str[0:k] + '-AF+BFB+FA-' + str[(k+1):]
            k+=10
        k += 1
    if n > 1:
        return hilbert_curve_expansion(str,n-1)
    else:
        return str

def draw_curve(str):
    direction = np.array([[1,0]])
    edge = np.zeros([2,2])+.5
    P = np.array([[0,1],[-1,0]])
    
    for c in str:
        if c == '+':
            direction = np.matmul(direction,P)
        elif c == '-':
            direction = np.matmul(direction,-P)
        elif c == 'F':
            edge[0,:] = edge[1,:]
            edge[1,:] = edge[1,:] + direction
            plt.plot(edge[:,0],edge[:,1],'k')
    

# plotting parameters
default_dpi = mpl.rcParamsDefault['figure.dpi']
mpl.rcParams['figure.dpi'] = default_dpi*1.5

# plot hilbert curves with different number of recursions
for k in range(4):
    plt.subplot(241+k)
    curve = hilbert_curve_expansion('A',k+1)
    draw_curve(curve)
    plt.gca().set_aspect('equal', 'box')
    plt.xlim([0,2**(k+1)])
    plt.ylim([0,2**(k+1)])
    plt.xticks([])
    plt.yticks([])
    plt.title('N=' + str(k+1))

plt.show()

draw_curve(hilbert_curve_expansion('A',5))
plt.gca().set_aspect('equal', 'box')
plt.xlim([0,2**5])
plt.ylim([0,2**5])
plt.xticks([])
plt.yticks([])
plt.title('N=5')
plt.show()

draw_curve(hilbert_curve_expansion('A',6))
plt.gca().set_aspect('equal', 'box')
plt.xlim([0,2**6])
plt.ylim([0,2**6])
plt.xticks([])
plt.yticks([])
plt.title('N=6')
plt.show()
```
