from decisionTree import decision_tree_I

class EmpericalErrorEvaluator(object):

    def __init__(self, data, data_set_feature_extractor, list_of_intepreters, max_split, emp_err_model):
        '''
        :param data: last column is the label or response
        :param data_set_feature_extractor: extract features from each partition which is static
        :param list_of_intepreters:
        '''
        self.data = data
        self.data_set_feature_extractor = data_set_feature_extractor
        self.data_set_feature_extractor = data_set_feature_extractor
        self.decision_tree_I = decision_tree_I(data, list_of_intepreters, max_depth)
        self.max_depth = max_depth

    def uniform_split(self):

