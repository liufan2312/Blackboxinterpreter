import node

class decision_tree_I(object):

    def __init__(self, data, label, list_of_interpreter, max_split):
        self.data = data
        self.list_of_interpreter = list_of_interpreter
        self.max_split = max_split
        self.label = label
        self.root = node(data)
        self.leafs = [self.root]

    def build_tree(self):
        pass


    def build_tree_uniform_split(self):
        '''
        :return: list of leafs after max_depth random split
        '''
        




    def leafs(self):
        return self.leafs











