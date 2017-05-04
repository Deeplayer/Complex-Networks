__Authors__ = 'Zineng Xu'
# 2017.3.5 #

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


G = nx.Graph(nx.read_pajek('E:\PycharmProjects\ML\CN\/real\/airports_UW.net'))
# print nx.is_directed(G)
# print nx.is_multigraphical(G)
# print len(G.nodes())                            # Number of nodes
# print len(G.edges())                            # Number of edges
degrees = sorted(nx.degree(G).values())
# print degrees
# print degrees[0], degrees[-1]                   # Minimum, maximum degree
# print 2.*len(G.edges())/len(G.nodes())          # Average degree
# print nx.average_clustering(G)                  # Average clustering coefficient
# print nx.degree_assortativity_coefficient(G)    # Assortativity
# print nx.average_shortest_path_length(G)        # Average (shortest) path length
# print nx.diameter(G)                            # Diameter


# degree distribution
h = nx.degree_histogram(G)
print h
#ccdf = [sum(h[k:]) for k in xrange(len(h))]
bins = np.linspace(np.log(degrees[0]), np.log(degrees[-1]+1), 15)
print bins
degrees_ = np.exp(bins)
print degrees_
degrees_[0] = degrees[0]
print degrees_
pdf_ = np.zeros(15)
for i in xrange(len(degrees_)):
    for j in xrange(len(h)):
        if j>=degrees_[i] and j<degrees_[i+1]:
            pdf_[i] += h[j]
print pdf_

plt.bar(bins, np.asarray(pdf_)/float(len(G.nodes())), width=0.4)
plt.yscale('log')
plt.xticks(bins, [round(i,2) for i in degrees_])
plt.xlabel('Degree')
plt.ylabel('Fraction of nodes')
plt.title('Network: airports_UW')
#nx.draw(G)
plt.show()

# CCDF
ccdf_ = [sum(h[int(np.ceil(k)):]) for k in degrees_]
print ccdf_
plt.bar(bins, np.asarray(ccdf_)/float(ccdf_[0]), width=0.4)
plt.yscale('log')
plt.xlabel('Degree')
plt.xticks(bins, [round(i,2) for i in degrees_])
plt.ylabel('CCDF')
plt.title('Network: airports_UW')
plt.show()

