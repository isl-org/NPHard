import ctypes
import networkx as nx
import numpy as np
import os
import sys
import scipy.sparse as sp

class reducelib(object):

    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.lib = ctypes.CDLL('%s/libreduce.so' % dir_path)

    def __CtypeNetworkX(self, g):
        edges = g.edges()
        e_list_from = (ctypes.c_int * len(edges))()
        e_list_to = (ctypes.c_int * len(edges))()

        if len(edges):
            a, b = zip(*edges)
            e_list_from[:] = a
            e_list_to[:] = b

        return (len(g.nodes()), len(edges), ctypes.cast(e_list_from, ctypes.c_void_p), ctypes.cast(e_list_to, ctypes.c_void_p))

    def __CtypeAdj(self, adj):
        adj = adj.tocoo()
        num_edge = adj.nnz
        num_node = adj.shape[0]
        e_list_from = (ctypes.c_int * num_edge)()
        e_list_to = (ctypes.c_int * num_edge)()
        edges = zip(adj.col, adj.row)
        if num_edge:
            a, b = zip(*edges)
            e_list_from[:] = a
            e_list_to[:] = b

        return (num_node, num_edge, ctypes.cast(e_list_from, ctypes.c_void_p), ctypes.cast(e_list_to, ctypes.c_void_p))

    def reduce_graph(self, adj):
        # g = nx.from_scipy_sparse_matrix(adj)
        n_nodes, n_edges, e_froms, e_tos = self.__CtypeAdj(adj)
        reduced_node = (ctypes.c_int * (n_nodes))()
        new_n_nodes = ctypes.c_int()
        new_n_edges = ctypes.c_int()
        reduced_xadj = (ctypes.c_int * (n_nodes+1))()
        reduced_adjncy = (ctypes.c_int * (2*n_edges))()
        mapping = (ctypes.c_int * (n_nodes))()
        reverse_mapping = (ctypes.c_int * (n_nodes))()
        crt_is_size = self.lib.Reduce(n_nodes, n_edges, e_froms, e_tos, reduced_node,
                                      ctypes.byref(new_n_nodes), ctypes.byref(new_n_edges),
                                      reduced_xadj, reduced_adjncy, mapping, reverse_mapping)
        # crt_is_size = self.lib.Reduce(n_nodes, n_edges, e_froms, e_tos, reduced_node)
        new_n_nodes = new_n_nodes.value
        new_n_edges = new_n_edges.value
        reduced_node = np.asarray(reduced_node[:])
        reduced_xadj = np.asarray(reduced_xadj[:])
        reduced_xadj = reduced_xadj[:new_n_nodes+1]
        reduced_adjncy = np.asarray(reduced_adjncy[:])
        reduced_adjncy = reduced_adjncy[:new_n_edges]
        mapping = np.asarray(mapping[:])
        reverse_mapping = np.asarray(reverse_mapping[:])
        reverse_mapping = reverse_mapping[:new_n_nodes]
        reduced_adj = sp.csr_matrix((np.ones(new_n_edges), reduced_adjncy, reduced_xadj), shape=[new_n_nodes, new_n_nodes])
        return reduced_node, reduced_adj, mapping, reverse_mapping, crt_is_size
        # return reduced_node[:], crt_is_size

    def local_search(self, adj, indset):
        n_nodes, n_edges, e_froms, e_tos = self.__CtypeAdj(adj)
        init_mis = (ctypes.c_int * (n_nodes))()
        final_mis = (ctypes.c_int * (n_nodes))()
        init_mis[:] = np.int32(indset)
        init_mis =  ctypes.cast(init_mis, ctypes.c_void_p)
        self.lib.LocalSearch(n_nodes, n_edges, e_froms, e_tos, init_mis, final_mis)
        indset = np.asarray(final_mis[:])
        return indset

    def test(self):
        self.lib.test()
