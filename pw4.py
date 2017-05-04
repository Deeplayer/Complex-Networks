import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


# parameters for SIS model
betas = [0.01*i for i in range(0, 102, 2)]    # 51 values: 0 ~ 1
mus = [0.1, 0.5, 0.9]

# extra parameters for Monte Carlo simulation
Nrep = 100
P0 = 0.2
Tmax = 1100
Ttrans = 1000

# read networks
G_SF = nx.Graph(nx.read_pajek('SF_500_g2.7.net'))
G_ER = nx.Graph(nx.read_pajek('ER1000k8.net'))
G_real = nx.Graph(nx.read_pajek('airports_UW.net'))

def Net2dict(net):
    graph = {}
    for node in net.nodes():
        attributes = {'neighbors': net.neighbors(node), 'state': 'None'}
        graph[node] = attributes
    return graph

G_SF = Net2dict(G_SF)
G_ER= Net2dict(G_ER)
G_real = Net2dict(G_real)

# create SIS model on networks
class SIS(object):
    def __init__(self, net, mu, beta, p0):
        self.net = net
        self.mu = mu
        self.beta = beta
        self.p0 = p0
        self.N = len(net)
        self.initial()

    def initial(self):
        self.infected_nodes = []
        self.susceptible_nodes = []
        for node in self.net:
            if np.random.random() < self.p0:
                self.net[node]['state'] = 'I'
                self.infected_nodes.append(node)
            else:
                self.net[node]['state'] = 'S'
                self.susceptible_nodes.append(node)

    def step(self):
        next_infected = []
        next_susceptible = []

        for node_i in self.infected_nodes:
            if np.random.random() < self.mu:
                next_susceptible.append(node_i)
            else:
                next_infected.append(node_i)

        for node_s in self.susceptible_nodes:
            neighbors = self.net[node_s]['neighbors']
            i, infected = 0, False
            while i < len(neighbors):
                if self.net[neighbors[i]]['state'] == 'I':
                    infected = np.random.random() < self.beta
                    if infected == True: break
                i += 1

            if infected:
                next_infected.append(node_s)
            else:
                next_susceptible.append(node_s)

        for node_i in next_infected:
            self.net[node_i]['state'] = 'I'

        for node_s in next_susceptible:
            self.net[node_s]['state'] = 'S'

        self.infected_nodes = next_infected
        self.susceptible_nodes = next_susceptible

        fraction_of_infection = float(len(self.infected_nodes)) / self.N

        return fraction_of_infection

# MonteCarlo simulation
def MonteCarlo(model, Nrep, Tmax, Ttrans):
    p_final = 0
    for i in range(Nrep):
        p_simulation = []
        for j in range(Tmax):
            p = model.step()
            p_simulation.append(p)

        stationary_p = p_simulation[Ttrans:]
        mean_p = sum(stationary_p)/len(stationary_p)
        p_final += mean_p
        # get the initial state for next rep
        model.initial()

    p_final = p_final/Nrep

    return p_final


# for beta in betas:
#     Ttrans = 4000
#     mu = 0.1
#     initial_model = SIS(G_real, mu, beta, P0)
#     ps = []
#     for i in range(Ttrans):
#         p = initial_model.step()
#         ps.append(p)
#     plt.plot([j for j in range(Ttrans)], ps)
#     print(beta)
#
# plt.xlabel('T_trans')
# plt.ylabel('P')
# plt.title('Real(N=3618), SIS(μ=0.1, P0=0.2)')
# plt.grid()
# plt.show()


# start simulation
i = 0
network_types = ['SF(N=500, γ=2.7)', 'ER(N=1000, <K>=8)', 'Real(N=3618)']
for net in [G_SF, G_ER, G_real]:
    print('Network type:', network_types[i])
    for mu in mus:
        p_sequence = []
        print(' mu:', mu)
        for beta in betas:
            print('  beta:', beta)
            start = time.time()
            initial_model = SIS(net, mu, beta, P0)
            p = MonteCarlo(initial_model, Nrep, Tmax, Ttrans)
            end = time.time()
            p_sequence.append(p)
            print('   p: %.2f' % p)
            print('   time: %.2f s' % (end-start))

        plt.plot(betas, p_sequence, 'o-')
        plt.xlabel('β')
        plt.ylabel('P')
        plt.title(network_types[i] + ', SIS(μ=%.1f, P0=%.1f)' % (mu, P0))
        plt.savefig(network_types[i][:2] + '_' + str(mu) + '.png')
        plt.close()

    i += 1
