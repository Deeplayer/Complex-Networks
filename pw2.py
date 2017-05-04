__Authors__ = 'Zineng Xu'
# 2017.3.9 #

import networkx as nx
import numpy as np
import random, math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


"""
ER network
"""
def ER_network(N, p):
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(0, N):
        for j in range(i+1, N):
            if random.random() < p or p == 1:
                G.add_edge(i,j)
    return G

# N = 500
# p = 0.05
# G = ER_network(N, p)
# nx.draw(G)
# plt.show()
#
# # plot distribution
# k = 2.*len(G.edges())/N             # experimental <k>
# k_ = p*(N-1)                        # theoretical <k>
# h = nx.degree_histogram(G)
# plt.plot(np.asarray(h)/float(sum(h)), 'o-', label='Experimental distribution')
# plt.axvline(x=k, ls='dashed', label='Experimental <k>=%f' % k)
# plt.plot([math.exp(-k_)*k_**i/math.factorial(i) for i in range(len(h))], 'o-', color='r', label='Theoretical distribution')
# plt.axvline(x=k_, color='r', ls='dashed', label='Theoretical <k>=%f' % k_)
# plt.legend(loc=2)
# plt.xlabel('Degree')
# plt.ylabel('P(k)')
# plt.title('ER: N=%d, p=%f' % (N,p))
# plt.show()


"""
Watts-Strogatz(WS) model
"""
def WS_model(N, k, p):
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(1, k/2+1):
        G.add_edges_from(zip(range(N), range(N)[i:]+range(N)[:i]))

    # rewiring
    for i in range(1, k/2+1):
        for x,y in zip(range(N), range(N)[i:]+range(N)[:i]):
            if random.random() < p or p == 1:
                y_ = random.choice(range(N))
                # avoid self-loops and multi-links
                if y_ != y and not G.has_edge(x, y_):
                    G.remove_edge(x, y)
                    G.add_edge(x, y_)

    return G

# N = 50
# p = 0.9
# k = 4      # k should be even integer
# G = WS_model(N, k, p)
# print nx.average_shortest_path_length(G)
# print nx.average_clustering(G)
# nx.draw_spring(G, node_size=100)
# plt.show()
#
# h = nx.degree_histogram(G)
# plt.plot(np.asarray(h)/float(sum(h)), '-o')
# plt.xlabel('Degree')
# plt.ylabel('P(k)')
# plt.title('N=%d, <k>=%d, p=%f' % (N, k, p))
# plt.show()

# k1 = 2.*len(G.edges())/N             # experimental <k>
# print k1
# k_ = k                               # theoretical <k>
# plt.plot(np.asarray(h)/float(sum(h)), 'o-', label='Experimental distribution')
# plt.axvline(x=k1, ls='dashed', label='Experimental <k>=%f' % k1)
# plt.plot([math.exp(-k_)*k_**i/math.factorial(i) for i in range(len(h))], 'o-', color='r', label='Theoretical distribution')
# plt.axvline(x=k_, color='r', ls='dashed', label='Theoretical <k>=%f' % k_)
# plt.legend(loc=2)
# plt.xlabel('Degree')
# plt.ylabel('P(k)')
# plt.title('N=%d, <k>=%d, p=%f' % (N, k, p))
# plt.show()



"""
BA preferential attachment model
"""
def BA_model(N, m0, m):
    G = nx.empty_graph(m0)
    for i in range(m0):     # generate a small connected net
        for j in range(m0):
            if j>i: G.add_edge(i,j)

    for h in range(m0,N):    # add new nodes
        G.add_node(h)
        k = nx.degree(G)
        ks = []
        for i in k: ks.append(k[i])
        ks.remove(ks[-1])
        ks = [sum(ks[:i+1]) for i in range(len(ks))]
        M = 1
        th = -1
        while M <= m:
            p = random.random()*ks[-1]
            for idx, i in enumerate(ks):
                if p < i+1 and p > i:
                    if th == idx:
                        break
                    else:
                        G.add_edge(h, idx)
                        th = idx
                        M += 1

    return G

#G = BA_model(100,5,10)
# G = nx.barabasi_albert_graph(100,10)
# pos = nx.spring_layout(G, k=0.2)
# nx.draw(G, pos, node_size=100)
# plt.show()

# h = nx.degree_histogram(G)
# h_ = [i for i in h if i>0]
# dd = np.asarray([math.log(float(i)/sum(h_)) for i in h_])
# d_ = np.asarray([math.log(i) for i in range(len(h)) if h[i]>0]).reshape((len(h_),1))
# lr = LinearRegression()
# lr.fit(d_, dd)
# gamma = -lr.coef_
#
# plt.scatter([math.exp(i) for i in d_], [math.exp(j) for j in dd],  color='black', label='Experimental distribution')
# plt.plot([math.exp(i) for i in d_], [math.exp(j) for j in lr.predict(d_)], color='blue', linewidth=2, label='Estimated distribution')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend(loc=1)
# plt.xlabel('Degree')
# plt.ylabel('P(k)')
# plt.title('Estimated gamma: %f' % gamma)
# plt.show()
#
# nx.draw_circular(G)
# plt.show()


"""
Configuration Model (CM)
"""
def CM(N, distribution=None, gamma=None, k_ave=None):
    G = nx.empty_graph(N)
    s = 1
    m = 0
    slots = []
    if distribution == 'ER':
        while s % 2 != 0 or m <= 0:
            degrees = np.random.poisson(k_ave, N)
            degrees = [int(i) for i in degrees]
            s = sum(degrees)
            m = np.min(degrees)
            #print m
        for i in range(N):
            slots += list(np.ones(degrees[i], dtype=int)*(i+1))

    elif distribution == 'SF':
        while s % 2 != 0 or m <= 0:
            degrees = (gamma+1)/np.random.power(gamma+1,N)   # k^(-gamma)
            degrees = [int(i) for i in degrees]
            s = sum(degrees)
            m = np.min(degrees)
        for i in range(N):
            slots += list(np.ones(degrees[i], dtype=int)*(i+1))

    # To remove parallel edges and self loops
    count = 1
    while count != 0:
        count, p1, p2 = 0, [], []
        random.shuffle(slots)
        for i in range(0, s-1, 2):
            if slots[i] == slots[i+1]:
                count += 1
            for j in range(i+2, s-1, 2):
                if abs(slots[i]-slots[i+1]) == abs(slots[j]-slots[j+1]) and (slots[i]+slots[i+1]) == (slots[j]+slots[j+1]):
                    count += 1

    for j in range(0, s-1, 2):
        G.add_edge(slots[j]-1, slots[j+1]-1)

    return G


N = 1000
k_ = 2
gamma = 3.0
dt = 'SF'
G = CM(N, distribution=dt, k_ave=k_, gamma=gamma)
# pos = nx.spring_layout(G, k=0.06)
# nx.draw(G, pos, node_size=100)
# plt.show()

if dt == 'ER':
    k = 2.*len(G.edges())/N             # experimental <k>
    h = nx.degree_histogram(G)
    plt.plot(np.asarray(h)/float(sum(h)), 'o-', label='Experimental distribution')
    plt.axvline(x=k, ls='dashed', label='Experimental <k>=%f' % k)
    plt.plot([math.exp(-k_)*k_**i/math.factorial(i) for i in range(len(h))], 'o-', color='r', label='Theoretical distribution')
    plt.axvline(x=k_, color='r', ls='dashed', label='Theoretical <k>=%f' % k_)
    plt.legend(loc=1)
    plt.xlabel('Degree')
    plt.ylabel('P(k)')
    plt.title('CM(ER): N=100, <k>=%d' % k_)
    plt.show()
elif dt == 'SF':
    h = nx.degree_histogram(G)
    h_ = [i for i in h if i>0]
    dd = np.asarray([math.log(float(i)/sum(h_)) for i in h_])
    d_ = np.asarray([math.log(i) for i in range(len(h)) if h[i]>0]).reshape((len(h_),1))
    lr = LinearRegression()
    lr.fit(d_, dd)
    gamma_ = -lr.coef_     # estimated gamma

    plt.scatter([math.exp(i) for i in d_], [math.exp(j) for j in dd],  color='black', label='Experimental distribution')
    plt.plot([math.exp(i) for i in d_], [math.exp(j) for j in lr.predict(d_)], color='blue', linewidth=2, label='Estimated distribution')
    plt.plot(1./np.asarray(range(len(h)))**gamma, color='red', linewidth=2, label='Theoretical distribution')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc=1)
    plt.xlabel('Degree')
    plt.ylabel('P(k)')
    plt.title('Estimated gamma: %f' % gamma_)
    plt.show()


pos = nx.spring_layout(G, k=0.06)
nx.draw(G, pos, node_size=100)
plt.show()

