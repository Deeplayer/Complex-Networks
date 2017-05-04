

from igraph import *
import networkx as nx
import community, string, math
import matplotlib.pyplot as plt

"""
NetworkX
"""
G = nx.Graph(nx.read_pajek('E:\PycharmProjects\ML\CN\/real\zachary_unwh.net'))
N = len(G.nodes())

# get node coordinates (if any)
pos = {}
idx = {}
for i, j in G.node.items(): idx[i] = int(j['id'])
idx = sorted(idx.items(), key=lambda d:d[1])
for line in open('E:\PycharmProjects\ML\CN\/real\zachary_unwh.net'):
    if line.split()[0] != '*Vertices':
        if line.split()[0] == '*Edges':
            break
        elif len(line.split()) > 2:
            s = line.translate(None, string.punctuation.replace('.','').replace('(','').replace(')','')).split()
            pos[s[1]] = (float(s[2]), -float(s[3]))
        else:
            pos = nx.spring_layout(G)

# get the partition of reference (if any)
c_ref, c_ref_ = [], []
try:
    com_ref = open("E:\PycharmProjects\ML\CN\/real\zachary_unwh-real.clu", "r").read()
    for line in com_ref.split():
        c_ref.append(line)
    c_ref = c_ref[2:]
    c_ref = [int(i) for i in c_ref]
    if min(c_ref) > 0: c_ref_ = [int(i)-1 for i in c_ref]
    else: c_ref_ = c_ref
except:
    pass

part = community.best_partition(G)      # Louvain algorithm
c = [part[i[0]] for i in idx]           # community number
c_ = [part.get(node) for node in G.nodes()]
nx.draw(G, pos, cmap=plt.get_cmap('jet'), node_color=c_, node_size=150, with_labels=False)
plt.savefig('com.png')

if c_ref != []:
    nvi = compare_communities(c_ref_, c, method='vi')/math.log(N)   # Normalized Variation of Information
    nmi = compare_communities(c_ref_, c, method='nmi')              # Normalized Mutual Information
    ri = compare_communities(c_ref_, c, method='rand')              # Rand index
    print 'Normalized Variation of Information:', nvi
    print 'Normalized Mutual Information:', nmi
    print 'Rand Index:', ri

mod = community.modularity(part, G)
print "Modularity:", mod

# save partitions in .clu
file = open('Louvain_zachary_unwh.clu', 'w')
file.write('*Vertices %d\n' % N)
for i in range(N):
    if c_ref == []: file.write(str(c[i]+1)+'\n')
    else:
        if min(c_ref) > 0: file.write(str(c[i]+1)+'\n')
        else: file.write(str(c[i])+'\n')
file.close()

