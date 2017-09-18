import node

class decision_tree_I(object):

    def __init__(self, data, list_of_interpreter, max_depth):
        self.data = data
        self.list_of_interpreter = list_of_interpreter
        self.max_depth = max_depth
        self.root = node(data)
        self.leafs = [self.root];

    def _get_minimal_error(self, data):
        pass

    def build_tree(self):
        pass



    def build_tree_uniform_split(self):
        '''
        :return: list of leafs after max_depth random split
        '''
        


    def leafs(self):
        return self.leafs












