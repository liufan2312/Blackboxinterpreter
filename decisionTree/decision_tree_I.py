import node
import numpy as np

class decision_tree_I(object):

    def __init__(self, data, label, type_list, list_of_interpreter, max_split, minimal_data_size):
        self.data = data
        self.list_of_interpreter = list_of_interpreter
        self.max_split = max_split
        self.label = label
        self.root = node(data, label)
        self.minimal_data_size = minimal_data_size
        self.leaves = [self.root]
        self.nodes = [self.root]


    def build_tree(self):
        pass


    def build_tree_uniform_split(self):
        '''
        :return: list of leaves after max_depth random split
        '''
        n_s, n_f = self.data.shape
        data = self.data
        for i in range(self.max_split):
            #choose leaf node
            split = False
            while(split):
                leaf_ind = np.random.randint(0, len(self.leaves))
                col_ind = np.random.randint(0, n_f)
                values = data[:, col_ind]
                values.sort()
                if 3*self.minimal_data_size >= len(values)-3*self.minimal_data_size:
                    continue
                v_ind = np.random.randint(3*self.minimal_data_size, len(values)-3*self.minimal_data_size)
                value = values[v_ind]
                self.leaves[leaf_ind].split(col_ind, value)
                nd = self.leaves[leaf_ind]
                self.leaves.pop(leaf_ind)
                self.leaves.append(nd.left_kid)
                self.leaves.append(nd.right_kid)
                split = True
                self.nodes.append(nd.left_kid)
                self.nodes.append(nd.right_kid)



    def nodes(self):
        return self.ndoes


    def leaves(self):
        return self.leafs












