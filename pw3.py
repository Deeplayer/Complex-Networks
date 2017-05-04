import math
import time
from igraph import *
import numpy as np

"""
igraph
"""
G = read('E:\PycharmProjects\ML\CN\model\/rb125.net')
# get the partition of reference (if any)
c_ref, c_ref_ = [], []
try:
    com_ref = open("E:\PycharmProjects\ML\CN\model\/rb125-3.clu", "r").read()
    for line in com_ref.split():
        c_ref.append(line)
    c_ref = c_ref[2:]
    c_ref = [int(i) for i in c_ref]
    if min(c_ref) > 0: c_ref_ = [int(i)-1 for i in c_ref]
    else: c_ref_ = c_ref
except:
    pass


N = G.vcount()                          # number of nodes
G.simplify(multiple=True, loops=True)   # removing self-loops and/or multiple edges (if any).

# get node coordinates (if any)
try:
    coord_x = G.vs['x']
    coord_y = G.vs['y']
    layout = [(coord_x[i], coord_y[i]) for i in range(len(coord_x))]
except:
    layout=G.layout("kk")

if G.is_weighted():
    ew =  G.es['weight']   # edge weights
else:
    ew = None

c = G.community_fastgreedy(weights=ew).as_clustering()       # Greedy Modularity (optimized)
#c = G.community_infomap(edge_weights=ew)                    # Infomap

plot(c, 'com.png', layout=layout)

if c_ref != []:
    nvi = compare_communities(c_ref_, c.membership, method='vi')/math.log(N)   # Normalized Variation of Information
    nmi = compare_communities(c_ref_, c.membership, method='nmi')              # Normalized Mutual Information
    ri = compare_communities(c_ref_, c.membership, method='rand')              # Rand index
    m_ref = G.modularity(c_ref_)
    print 'Normalized Variation of Information:', nvi
    print 'Normalized Mutual Information:', nmi
    print 'Rand Index:', ri
    print 'Modularity_ref:', m_ref

m = G.modularity(c)
print 'Modularity:', m

# save partitions in .clu
file = open('greedy_rb125-3.clu', 'w')
file.write('*Vertices %d\n' % N)
cm = c.membership
for i in range(N):
    if c_ref == []: file.write(str(cm[i]+1)+'\n')
    else:
        if min(c_ref) > 0: file.write(str(cm[i]+1)+'\n')
        else: file.write(str(cm[i])+'\n')
file.close()

