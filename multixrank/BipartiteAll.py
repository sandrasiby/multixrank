import networkx
import numpy
import scipy


class BipartiteAll:
    """Deals with list of bipartites"""

    def __init__(self, source_target_bipartite_dic, multiplexall):

        self.source_target_bipartite_dic = source_target_bipartite_dic
        self.multiplexall = multiplexall

        self.multiplex_layer_count_list2d = [len(multiplexall.layer_tuple)
                                             for multiplexall in
                                             multiplexall.multiplex_tuple]
        self.multiplexall_node_list2d = [multiplexall.nodes for
                                         multiplexall in
                                         multiplexall.multiplex_tuple]
        self.multiplex_dic = dict()
        for multiplexone_obj in self.multiplexall.multiplex_tuple:
            self.multiplex_dic[multiplexone_obj.key] = multiplexone_obj

        self._bipartite_matrix = None
        self._graph = None  # Networkx with all undirected edges in Bipartites
        self._digraph = None  # Networkx with all directed edges in Bipartites

    @property
    def graph(self):
        """Returns graph with all undirected edges in all bipartites"""

        if self._graph is None:
            self._graph = networkx.Graph()
            for edge_tple in self.source_target_bipartite_dic:
                bipartite_obj = self.source_target_bipartite_dic[edge_tple]
                if isinstance(bipartite_obj.networkx, networkx.Graph):
                    edge_data_lst = [(u, v, bipartite_obj.networkx[u][v]) for u, v in bipartite_obj.networkx.edges]
                    self._graph.add_edges_from(edge_data_lst)
        return self._graph

    @property
    def digraph(self):
        """Returns graph with all directed edges in all bipartites"""

        if self._digraph is None:
            self._digraph = networkx.DiGraph()
            for edge_tple in self.source_target_bipartite_dic:
                bipartite_obj = self.source_target_bipartite_dic[edge_tple]
                if isinstance(bipartite_obj.networkx, networkx.Graph):
                    edge_data_lst = [(u, v, bipartite_obj.networkx[u][v]) for u, v in bipartite_obj.networkx.edges]
                    #SS: remove edges that we don't want? but no layers here
                    self._digraph.add_edges_from(edge_data_lst)
        return self._digraph

    def update_edges(temp, i, j, two_multiplex_nodes):

        new_temp = temp.copy()
        multiplexone_obj1 = self.multiplexall.multiplex_tuple[i]
        multiplexone_obj2 = self.multiplexall.multiplex_tuple[j]
        multiplex_key1 = multiplexone_obj1.key
        multiplex_key2 = multiplexone_obj2.key
        # Get all nodes in the multiplexes
        two_multiplex_nodes = multiplexone_obj1.nodes + multiplexone_obj2.nodes

        if (multiplex_key1, multiplex_key2) in self.source_target_bipartite_dic:
            # Get the graph (all layers)
            bipartite_layer_obj = self.source_target_bipartite_dic[(multiplex_key1, multiplex_key2)]
            bipartite_layer_networkx = bipartite_layer_obj.networkx

        for r in range(temp.shape[0]):
            for c in range(temp.shape[1]):
                # Iterate through each element of temp
                # Each element is a matrix of interactions between nodes in MP1 and nodes in MP2
                # Each element represents interactions on a layer
                src_layer = self.multiplexall.multiplex_tuple[i].layer_tuple[r].key
                dst_layer = self.multiplexall.multiplex_tuple[j].layer_tuple[c].key
                edges_to_remove = []
                H = bipartite_layer_networkx.copy()

                for edge in H.edges:
                    src_l = edge['src_layer']
                    dst_l = edge['dst_layer']
                    # If the interaction does not exist on the layers, remove them
                    if (src_layer != src_l) or (dst_layer != dst_l):
                        edges_to_remove.append(edge)

                H.remove_edges_from(networkx.selfloop_edges(H))
                H.remove_edges_from(edges_to_remove)
                # Make new adjacency matrix of the layers
                B = networkx.to_scipy_sparse_matrix(H, nodelist=two_multiplex_nodes, format="csr")
                new_element = B[0:len(self.multiplexall_node_list2d[i]), len(self.multiplexall_node_list2d[i])::]
                new_temp[r, c] = new_element

        return new_temp            

    @property
    def bipartite_matrix(self):
        """"""

        #######################################################################
        #
        # Will add B bipartite multigraph matrix to each B_i_j with i!=j
        #
        #######################################################################

        if self._bipartite_matrix is None:

            multiplexall_supra_adj_matrix_list = []
            for i, multiplex_obj in enumerate(self.multiplexall.multiplex_tuple):
                multiplexall_supra_adj_matrix_list.append(multiplex_obj.supra_adj_matrixcoo)

            self._bipartite_matrix = numpy.zeros((len(self.multiplex_layer_count_list2d), len(self.multiplex_layer_count_list2d)), dtype=object)

            for i, multiplexone_obj1 in enumerate(self.multiplexall.multiplex_tuple):
                for j, multiplexone_obj2 in enumerate(self.multiplexall.multiplex_tuple):
                    multiplex_key1 = multiplexone_obj1.key
                    multiplex_key2 = multiplexone_obj2.key

                    if not (multiplex_key1 == multiplex_key2):
                        if (multiplex_key1, multiplex_key2) in self.source_target_bipartite_dic:
                            bipartite_layer_obj = self.source_target_bipartite_dic[(multiplex_key1, multiplex_key2)]
                            bipartite_layer_networkx = bipartite_layer_obj.networkx

                            bipartite_layer_networkx.remove_edges_from(networkx.selfloop_edges(bipartite_layer_networkx))
                            two_multiplex_nodes = multiplexone_obj1.nodes + multiplexone_obj2.nodes
                            B = networkx.to_scipy_sparse_matrix(bipartite_layer_networkx, nodelist=two_multiplex_nodes, format="csr")
                            self._bipartite_matrix[i, j] = B[0:len(self.multiplexall_node_list2d[i]), len(self.multiplexall_node_list2d[i])::]
                            self._bipartite_matrix[j, i] = B[len(self.multiplexall_node_list2d[i])::, 0:len(self.multiplexall_node_list2d[i])]
            
            for i in range(len(self.multiplex_layer_count_list2d)):
                for j in range(len(self.multiplex_layer_count_list2d)):
                    if i != j:
                        if isinstance(self._bipartite_matrix[i, j], int):
                            row = numpy.shape(multiplexall_supra_adj_matrix_list[i])[0]
                            col = numpy.shape(multiplexall_supra_adj_matrix_list[j])[1]
                            self._bipartite_matrix[i, j] = scipy.sparse.coo_matrix((row, col))
                        else:
                            temp = numpy.zeros((self.multiplex_layer_count_list2d[i],
                                               self.multiplex_layer_count_list2d[j]),
                                               dtype=object)
                            for k in range(self.multiplex_layer_count_list2d[i]):
                                temp[k] = self._bipartite_matrix[i, j]
                            #SS: we update here
                            new_temp = update_edges(temp, i, j)
                            self._bipartite_matrix[i, j] = scipy.sparse.bmat(temp, format='coo')

        return self._bipartite_matrix
