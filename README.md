# Complex-Networks

CN Assignments

PW1: Structural descriptors of complex networks

Calculation of structural descriptors of complex networks

a) Numerical descriptors
 
  · Number of nodes
 
  · Number of edges
 
  · Minimum, maximum and average degree
 
  · Average clustering coefficient (average of the clustering coefficients of each node)
 
  · Assortativity
 
  · Average path length (average distance between all pairs of nodes)
 
  · Diameter (maximum distance between nodes in the network)
 
Put the results in table form for as many as possible of the networks, including at least the following ones:
 
 · toy/circle9.net
 
 · toy/star.net
 
 · toy/graph3+1+3.net
 
 · toy/grid-p-6x6.net
 
 · model/homorand_N1000_K4_0.net
 
 · model/ER1000k8.net
 
 · model/SF_1000_g2.7.net
 
 · model/ws1000.net
 
 · real/zachary_unwh.net
 
 · real/airports_UW.net
 
b) Plot the degree distributions (PDF, probability distribution function) and the complementary cumulative degree distributions (CCDF, complementary cumulative distribution function) for the following networks:
 
 · model/ER1000k8.net
 
 · model/SF_1000_g2.7.net
 
 · model/ws1000.net
 
 · real/airports_UW.net
 
For each network, choose the appropriate histogram form, either linear (e.g. figure 8.3 of the book) or log-scale histogram (e.g. figures 8.6 and 8.7 of the book).

For this activity, you may use any software available for the analysis of complex networks, such as:
 
 · NetworkX
 
 · igraph
 
 · Pajek
 
 · Gephi
 
 · Radatools
 
 · MatlabBGL
 
 · Cytoscape
 
Alternatively, you may implement yourself the calculation of (some of) the descriptors.

----------------------------------------------------------------
PW2: Models of complex networks

Implementation of models of complex networks.

Implement generators of complex networks for, at least, two of the following models of complex networks (sorted by increasing difficulty), which must include at least one of the two last ones (BA or CM):

· Erdös-Rényi (ER) networks, either G(N,K) or G(N,p) 

· Watts-Strogatz (WS) small-world model 

· Barabási & Albert (BA) preferential attachment model 

· Configuration Model (CM) 

The correction, the number and the difficulty of the implemented models will be taken into account.

It is not allowed to use already implemented network generators such as the ones in NetworkX or Pajek. You may use libraries implementing “network” or “graph” data types, to avoid unnecessary work, but not the algorithms for the generation of networks for these models.

The delivery must include:

 · Source code in text form, not in binary form (e.g. do not deliver python, R or Mathematica notebooks)

 · The executable (if any)

 · Networks generated for the selected models, with different sizes N and for different values of the parameters of the models:

 - ER: different values of "K" for G(N,K), or of "p" for G(N,p), e.g. p=0.00, 0.01, 0.02, 0.03, 0.05, 0.1

 - WS: different values of "p", including p=0, e.g. p=0.0, 0.1, 0.2, 0.5, 0.9, 1.0

 - BA: different values of "m" (number of edges that each new nodes forms with the existing nodes), e.g. m=1, 2, 4, 10

 - CM: different degree distributions: Poisson (ER), e.g. =2, 4; power-law (SF) with different exponents, e.g. gamma=2.2, 2.5, 2.7, 3.0

· Plots of some of the small size generated networks, e.g. N=50 (ER, WS), N=100 (BA, CM)

· Plots of the degree distributions, including the theoretical values (corresponding to the selected parameters) and the experimental ones (for the generated network)

· Estimation of the exponent for the empirical degree distributions of BA and CM (SF)

----------------------------------------------------------------
PW3: Community detection

Apply at least three different community detection algorithms for the attached networks. It is not necessary to implement them, you may use any freely available software. At least one of the algorithms must be based on the optimization of modularity (but not all of them), and you must use at least two different programs (i.e. do not use the same application all the time).

Some of the provided networks have a partition of reference, obtained from external information. In these cases, you have to compare your partitions with them, using at least the following standard measures: Jaccard Index, Normalized Mutual Information (arithmetic normalization), and Normalized Variation of Information. It is not necessary to implement the calculation of these indices, you may use any program (e.g. Radatools).

The objective is to compare the partitions obtained with the different algorithms, to try to conclude which is the best method you have found.

The delivery must include:

· Brief description of the algorithms and the programs used.

· Selected parameters for each algorithm and/or network, and the scripts used (if any).

· The obtained partitions, in Pajek format (*.clu)

· For each network and partition, a plot with color-coded communities. To facilitate the comparison of partitions:

- If the network contains coordinates for the nodes (e.g. airports_UW.net), use them to establish the position of the nodes. Otherwise, use the Kamada-Kawai (without separation of components) to distribute the nodes in the plane. Avoid circular layouts.

- The position of the nodes must not change for all the partitions of the same network.

- Put all the plots of the different partitions of the same network close to each other, i.e. do not group by algorithm, group by network.

· A table with the comparison measures between your partitions and the reference ones, grouped by network.

· A table with the modularity values of all the partitions (including the reference ones), grouped by network.

· Your conclusions about the advantages and quality of the used algorithms.

It is not expected that the obtained partitions be equal to the reference ones or between them, with some exceptions.

----------------------------------------------------------------
PW4: Community detection

Dynamics on complex networks

Monte Carlo simulation of an epidemic spreading dynamics in complex networks, using the SIS model in which each node represents an individual which can be in two possible state: Susceptible (S), i.e. healthy but can get infected: Infected (I), i.e. has the disease and can spread it to the neighbors.

We are interested in the calculation of the fraction of infected nodes, ρ, in the stationay state, as a function of the infection probability of the disease β (at least 51 values, Δβ=0.02), for different values of the recovery probability μ (e.g. 0.1, 0.5, 0.9). Try different networks (e.g. Erdös-Rényi, scale-free, real), different sizes (at least 500 nodes), average degrees, exponents, etc. Do not make all the combinations, about 10 plots ρ(β) are enough.

The delivery must include:

· Source code in text form, not in binary form (e.g. do not deliver python, R or Mathematica notebooks)

· The executable (if any)

· Networks used in Pajek format

· Results files

· Document with all the plots, pointing out “all” the parameters of the corresponding simulation.

Monte Carlo simulations may require a lot of computing time, thus it
