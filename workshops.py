# import warnings
import numpy as np
import numba
from typing import Optional, Union
import networkx as nx
import random
from itertools import product
import pickle
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class SERModel:
    """
    Model simulating neurons connected via connection matrix. The neurons have three possible states:
        Susceptible (0)
        Excited (1)
        Refractory (-1)
    Using (-1, 0, 1) instead of (0, 1, 2) is slightly faster and easier to implement efficiently.

    Parameters:
        n_steps: int
            Number of time steps of the simulation.
        prob_spont_act: float 0-1
            Probability of spontaneous process S -> E
        prob_recovery: float 0-1
            Probability of recovery, process: R -> S
        threshold: float
            (for a weighted network, as we use)
            Minimum weighted sum over active neighbours to activate a node.
        prop_e: float 0-1
            Proportion of nodes in excited state in the initial state of the system. It can be set to zero,
            then the simulation need more time to "initialize" due to very low probability of spontaneous activation.

    """

    def __init__(
            self, *,
            n_steps: int,
            # prob_spont_act: float, # r1?
            # prob_recovery: float, # r2?
            threshold: float,
            connectome: np.ndarray,
            prop_e: float = 0.01,
            n_transient: int = 200,
    ) -> None:
        # self.n_steps = n_steps
        self.n_steps = n_steps + n_transient # for agreement with Ising model
        self.threshold = threshold
        self.prop_e = prop_e
        self.n_transient = n_transient
        self.connectome = connectome

        self.network_size = self.connectome.shape[0]

        self.prob_spont_act = 2 / self.network_size
        self.prob_recovery = self.prob_spont_act ** 0.2

    @staticmethod
    def init_state(*, n_nodes: int, prop_e: float) -> np.ndarray:
        """
        Prepare an initial state of the system with prop_e as a proportion of excited states.
        The routine assumes that in the initial state we have only Susceptible and Excited states.

        Parameters:
        :param n_nodes: int
            Number of nodes
        :param prop_e: float 0-1
            Proportion of excited nodes

        Returns:
        states: 1-D np.ndarray of length n_nodes

        Activity encoded as follows:
            Susceptible: 0
            Excited: 1
            Refractory: -1

        """

        if prop_e is None:
            raise ValueError("prop_e must be defined!")
        if prop_e > 1.0:
            raise ValueError("prop_e must be <=1, now it is given prop_e={}".format(prop_e))

        # Initialize vector (assuming -1 as Refractory):
        states = np.zeros(n_nodes, dtype=np.int)

        # Compute number of excited nodes:
        n_nodes_e = int(round(n_nodes * prop_e, 2))

        # Set states:
        states[: n_nodes_e] = 1
        np.random.shuffle(states)

        # Print a warning if not all possible states (-1, 0, 1) in the initial state
        # In fact we do not care about that...
        # if len((set(states))):
        #     warnings.warn("Warning: not all states are present in the initial state!")

        return states

    def simulate(self, *,
                 # adj_mat: np.ndarray,
                 states: Optional[np.ndarray] = None
                 ) -> np.ndarray:
        """
        Run the simulation of the brain model using given connection matrix.

        Parameters
        :param adj_mat: np.ndarray
            Connection (adjacency) matrix of the system.
        :param states: 1-D np.ndarray, optional
            Initial state of the system. If None, random initial state is generated using brain_model class parameters.

        Returns:
        act_mat: 2D np.ndarray (n_nodes, n_steps)
            Activity matrix (node vs time) with:
                Susceptible: 0
                Excited: 1
                Refractory: -1
        """

        # Initialize the state if needed:
        if states is None:
            states = SERModel.init_state(n_nodes=len(self.connectome), prop_e=self.prop_e)

        states = states.astype(self.connectome.dtype)  # cast for numba

        # external routine, necessary for numba acceleration
        return _run(
            adj_mat=self.connectome,
            states=states,
            n_steps=self.n_steps,
            prob_spont_act=self.prob_spont_act,
            prob_recovery=self.prob_recovery,
            threshold=self.threshold,
            n_transient=self.n_transient
        )

    def calc_Tc(self):
        mean_wi = self.connectome.sum(axis=0).mean()
        return mean_wi * self.prob_recovery / (1 + 2 * self.prob_recovery)


@numba.njit(fastmath=True)
def _run(
        adj_mat: np.ndarray,
        states: np.ndarray,
        n_steps: int,
        prob_spont_act: float,
        prob_recovery: float,
        threshold: float,
        n_transient: int = 0
        ) -> np.ndarray:
    """
    :param adj_mat: np.ndarray
        Connection matrix.
    :param states: np.ndarray
        Initial state of the system.
    :param n_steps: int
        Length of the time series.
    :param prob_spont_act: float 0-1
        Probability of spontaneous activation/
    :param prob_recovery: float 0-1
        Probability of relaxation from R to S state.
    :param threshold: float
        Activation threshold.
    :param n_transient: int
        Time of initial simulation which is discarded. If n_transient > 0 then the full time series will have
        (n_steps - n_transient) time steps in total.
    :return: np.ndarray
        2-D activity matrix (node vs time).
    """
    _dtype = adj_mat.dtype
    n_nodes = len(adj_mat)

    # Initialize activity matrix
    act_mat = np.zeros((n_nodes, n_steps), dtype=_dtype)
    act_mat[:, 0] = states

    # Evaluate all the stochastic transition probabilities in advance:
    spont_activated = np.random.random(act_mat.shape) < prob_spont_act
    recovered = np.random.random(act_mat.shape) < prob_recovery

    for t in range(n_steps - 1):
        # E -> R
        act_mat[act_mat[:, t] == 1, t + 1] = -1
        # R -> S stochastic:
        refrac = act_mat[:, t] == -1
        act_mat[refrac, t + 1] = act_mat[refrac, t] + recovered[refrac, t]

        # S -> E threshold + stochastic:
        # (act_mat[:, t] == 1).astype(_dtype) is a vector of 0s and 1s where we have active nodes!
        weighed_neigh_input = adj_mat.T @ (act_mat[:, t] == 1).astype(_dtype)
        susce = act_mat[:, t] == 0
        act_mat[susce, t + 1] += np.logical_or(weighed_neigh_input[susce] >= threshold,
                                               spont_activated[susce, t])

    return act_mat[:, n_transient:]


class Grid:
    '''
    Set of methods for creation of grid-like networkx graphs. Basic methods:
    
    grid_2d - to create a two-dimensional grid
    grid_2d_patch - to create a two-dimensional grid with a patch in the middle
    grid_2d_slice - to create a two-dimensional grid sliced in the middle
    '''
    def __init__(self):
        pass

    @staticmethod
    def grid_2d(size_x : int,
                size_y : int,
                periodic : bool = False):

        g = nx.grid_2d_graph(size_x, size_y, periodic=periodic)
        nx.set_node_attributes(g,
                               values=0,
                               name='subsystem')
        return g

    @staticmethod
    def grid_2d_patch(size_x : int,
                      size_y : int,
                      x0 : int,
                      y0 : int,
                      dx : int,
                      dy : int,
                      remove_frac : float = 1.0):

        g = Grid.grid_2d(size_x,size_y)

        # find edges on the patch boundary
        x1 = x0 + dx
        y1 = y0 + dy
        x0_bunch = [((x0-1,i),(x0,i)) for i in range(y0,y1+1)]
        x1_bunch = [((x1,i),(x1+1,i)) for i in range(y0,y1+1)]
        y0_bunch = [((i,y0-1),(i,y0)) for i in range(x0,x1+1)]
        y1_bunch = [((i,y1),(i,y1+1)) for i in range(x0,x1+1)]

        ebunch = x0_bunch + y0_bunch + x1_bunch + y1_bunch

        # randomize boundary edges
        rng = np.random.default_rng(42)
        ebunch_ixs = list(rng.permutation(range(len(ebunch))))
        ebunch = [ebunch[i] for i in ebunch_ixs]
        ixmax = int(remove_frac*len(ebunch))
        g.remove_edges_from(ebunch[:ixmax])
        print(f'removing {len(ebunch[:ixmax])} out of {len(ebunch)} edges on the boundary')
        # label subsystems

        patch_nodes = product(list(range(x0,x1+1)),
                              list(range(y0,y1+1)))
        nx.set_node_attributes(g,
                               values={n: 1 for n in patch_nodes},
                               name='subsystem')
        return g

    @staticmethod
    def grid_2d_sliced(size_x : int,
                       size_y : int,
                       slice_ix : int,
                       row = False,
                       remove_frac : float = 1.0):

        g = Grid.grid_2d(size_x,size_y)
        if not row:
            ebunch = [((slice_ix-1,i),(slice_ix,i)) for i in range(size_y)]
            patch_nodes = product(list(range(0,slice_ix)),
                                  list(range(0,size_y)))
        else:
            ebunch = [((i,slice_ix-1),(i,slice_ix)) for i in range(size_x)]
            patch_nodes = product(list(range(0,size_x)),
                                  list(range(0,slice_ix)))
        #randomize the slice
        rng = np.random.default_rng(42)
        ebunch_ixs = list(rng.permutation(range(len(ebunch))))
        ebunch = [ebunch[i] for i in ebunch_ixs]
        ixmax = int(remove_frac*len(ebunch))

        g.remove_edges_from(ebunch[:ixmax])
        print(f'removing {len(ebunch[:ixmax])} out of {len(ebunch)} edges on the boundary')
        # label subsystems
        nx.set_node_attributes(g,
                               values={n: 1 for n in patch_nodes},
                               name='subsystem')

        return g

    @staticmethod
    def nx_to_np(graph: nx.Graph, store: str = 'graph'):

        if store == 'adj':
            adj = nx.to_numpy_matrix(graph)
        subsystems = np.array([attr['subsystem'] for n,attr in graph.nodes.items()])
        nodes = np.array([list(n) for n,attr in graph.nodes.items()])

        if store == 'adj':
            return {'adj': adj,
                    'subsystems': subsystems,
                    'nodes': nodes}

        elif store == 'graph':
            return {'graph': graph,
                    'subsystems': subsystems,
                    'nodes': nodes}

    @staticmethod
    def save_graph(graph: nx.Graph, file: str, store: str = 'graph'):
        out = Grid.nx_to_np(graph, store = store)
        if store == 'adj':
            np.savez_compressed(file,**out)
        elif store == 'graph':
            with open(file,'wb') as fp:
                pickle.dump(out,fp)


class IsingModel:
    '''
    Ising model simulating spins connected via connectome network. The spins are in two possible states +1 and -1.

    Parameters:
        n_steps: int
            Number of time steps of the simulation.
        T: float
            Model temperature.
        network: nx.Graph or np.ndarray
            (weighted) Graph with which the spins interact.
        J: float (default = 1.0)
            Coupling parameter.
        init_type: str (default = uniform)
            Initial condition type. Possible values uniform, random_sym and random_asym.
        n_transient: int
            Drop the initial time steps to reach thermalization.
        n_sweep: int (default = None)
            How many sweeps per single time step. If None, the whole network is swept.
    '''
    def __init__(self,
                 n_steps: int,
                 T: float,
                 network: Union[nx.Graph, np.ndarray],
                 J: float=1.0,
                 init_type: str = 'uniform',
                 n_transient: int = 500,
                 n_sweep: int = None) -> None:
        self.n_steps = n_steps
        self.T = T
        self.J = J
        self.n_transient = n_transient
        self.n_sweep = n_sweep
        self.init_type = init_type
        
        if type(network) == np.ndarray:
            self.network = nx.from_numpy_array(network)
        elif type(network) == nx.Graph:
            self.network = network
        else:
            raise Error('unknown network type')
            
    def E(self,s: dict) -> float:
        E0 = 0
        for n in self.network.nodes:
            nn = np.array([s[n2] for _,n2 in self.network.edges(nbunch=n)]).sum()
            E0 -= self.J * s[n] * nn
        return E0

    def init_state(self) -> dict:
        if self.init_type == 'uniform':
            return {n: -1 for n in self.network.nodes}
        elif self.init_type == 'random_sym':
            return {n: 2*np.random.randint(2)-1 for n in self.network.nodes}
        elif self.init_type == 'random_asym':
            return {n: np.random.choice([-1,1],p=[0.75,0.25]) for n in self.network.nodes}
        else:
            print(f'unknown init_type={self.init_type}')
            return None
        
    def sweep(self,s: dict) -> None:
        if self.n_sweep:
            f = lambda x: random.sample(x,self.n_sweep)
        else:
            f = lambda x: x
            
        for n in f(self.network.nodes):
            nn = np.array([s[n2] for _,n2 in self.network.edges(nbunch=n)]).sum()

            new_s = -s[n]
            dE =- self.J * (new_s-s[n])*nn

            if dE <= 0.:
                s[n] = new_s
            elif np.exp(-dE/self.T) > np.random.rand():
                s[n] = new_s        

    def simulate(self) -> np.ndarray:
        def s_arr(s: dict): 
            return np.array(list(s.values()))
        # print(f'running sim grid={grid}; J={J}; T={T}')

        s = self.init_state()
        X = s_arr(s).reshape(-1,1)

        # print('thermalization...')
        for i in range(self.n_transient):
            self.sweep(s)    

        # print('simulation')
        # for i in tqdm(range(self.n_steps-1)):
        for i in range(self.n_steps-1):
            self.sweep(s)
            X = np.hstack((X,s_arr(s).reshape(-1,1)))
        return X

    @staticmethod
    def calc_Tc(J: float) -> float:
        '''
        calculate critical temperature
        '''
        return 2*J/np.log(np.sqrt(2)+1)

    @staticmethod
    def calc_mag(T: float,J: float) -> float:
        '''
        calculate Onsager magnetization
        '''
        onsager = (1 - 1/(np.sinh(2*J/T)**4))
        if onsager < 0: onsager = 0.
        return onsager**(1/8) 

def cluster_sizes(adj: np.ndarray, mask: np.ndarray) -> np.ndarray:
    adj0 = adj[mask][::, mask]
    G0 = nx.from_numpy_array(adj0)
    return np.array([len(c) for c in sorted(nx.connected_components(G0), key=len, reverse=True)])


def find_clusters(X: np.ndarray, adj: np.ndarray, n_clusters: int = 10, cluster_val = 1) -> np.ndarray:
    '''
    find cluster sizes (single mode)
    
    Parameters:
        X: np.ndarray
            2D data array with shape (n_neurons, n_timesteps).
        adj: np.ndarray
            Adjacency matrix.
        n_clusters: int (default = 10)
            Specify the number of largest clusters to track.
        cluster_val: int (default = 1)
            Calculate clusters of cluster_val's. 
            
    Returns:
        clusters: 2D np.ndarray
            Cluster sizes with shape (n_timesteps, n_clusters)
    '''
    
    clusters = None
    for Xslice in X.T:
        mask = Xslice == cluster_val
        cl = cluster_sizes(adj, mask)
        offset = n_clusters-len(cl)
        if offset > 0:
            cl = np.pad(cl, [(0, offset)])
        else:
            cl = cl[:n_clusters]

        clusters = cl if clusters is None else np.vstack((clusters, cl))

    return clusters


def batch_clusters(Xs: np.ndarray, adj: np.ndarray, n_clusters: int = 10) -> np.ndarray:
    '''
    find cluster sizes (batch mode)
    
    Parameters:
        Xs: np.ndarray
            3D data array with shape (n_thresholds, n_neurons, n_timesteps).
        adj: np.ndarray
            Adjacency matrix.
        n_clusters: int (default = 10)
            Specify the number of largest clusters to track.
            
    Returns:
        clusters: 3D np.ndarray
            Cluster sizes with shape (n_thresholds, n_timesteps, n_clusters)
    '''
    clusters = None
    for X in tqdm(Xs):
        # cl = find_clusters(X, adj, n_clusters).mean(axis=0)
        cl = find_clusters(X, adj, n_clusters)
        cl = np.expand_dims(cl, axis=0)
        clusters = cl if clusters is None else np.concatenate((clusters, cl), axis=0)

    return clusters

def watts_strogatz_connectome(number_of_nodes: int, k: int, p: float) -> np.ndarray:
    '''
    generate a weighted Watts-Strogatz connectome
    
    Parameters:
        number_of_nodes: int
            Specify number of nodes in the connectome.
        k: int
            Number of nearest neighbors for each node.
        p: float
            Rewiring probability.
    
    Returns:
        adjacency: 2D np.ndarray
        
    '''
    g = nx.watts_strogatz_graph(number_of_nodes, k, p)
    adjacency = nx.adjacency_matrix(g).todense().astype(float)
    adj_tri_up = np.triu(adjacency)
    number_of_weights = int(np.sum(adj_tri_up))
    weights = np.random.exponential(scale=1 / 12.5, size=number_of_weights)
    adj_tri_up[adj_tri_up > 0] = weights
    adjacency = adj_tri_up + adj_tri_up.T
    return adjacency

def plot_color_clusters(snapshot: np.ndarray, network: nx.Graph) -> None:
    """
    Starting from a 2D snapshot of Ising model (entries -1 and 1) finds the nodes that are in the largest (S1)
    and second largest (S2) clusters.

    Plots the initial state ("check") and the state with S1 in red and S2 in blue.

    """

    Lx, Ly = snapshot.shape
    state = snapshot.flatten()
    state_check = np.copy(state)

    ising_adj = nx.adjacency_matrix(network).todense()

    mask_up = 1.0 * (state == 1)
    mask_down = 1.0 * (state == -1)
    mask_mat_up = np.outer(mask_up.T, mask_up)
    mask_mat_down = np.outer(mask_down.T, mask_down)

    ising_adj_up = np.multiply(ising_adj, mask_mat_up)
    ising_adj_down = np.multiply(ising_adj, mask_mat_down)

    if np.all(ising_adj_up == 0.0): # i.e. graph not connected
        clusters_up = list(np.argwhere(mask_up == 1.0)) # finds the first one spin up
    else:
        g_up = nx.from_numpy_array(ising_adj_up) # connected graph
        clusters_up = list(nx.connected_components(g_up))

    if np.all(ising_adj_down == 0.0):
        clusters_down = list(np.argwhere(mask_down == 1.0))
    else:
        g_down = nx.from_numpy_array(ising_adj_down)
        clusters_down = list(nx.connected_components(g_down))

    all_clusters = clusters_up + clusters_down

    if len(all_clusters) < 2:
        print("Only 1 cluster in the system.")
        fig, ax = plt.subplots(1, 1)
        ax.imshow(state_check.reshape(Lx, Ly), cmap='gray')

    else:
        all_clusters = sorted(all_clusters, key=len, reverse=True)
        S1 = list(all_clusters[0])
        S2 = list(all_clusters[1])

        state[S1] = 3
        state[S2] = 2

        cmap = ListedColormap(["black", "None", "white", "blue", "red"], 5)

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10,3))
        ax0.imshow(state_check.reshape(Lx, Ly), cmap='gray')
        ax1.imshow(state.reshape(Lx, Ly), cmap=cmap, vmin=-1, vmax=3)
