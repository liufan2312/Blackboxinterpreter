class node(object):
    def __init__(self, data):
        '''
        :param data:  data is a numpy array with (sample_size x number of columns, [data point features])
        the last column of the data is the label(or response)
        '''
        self.data = data
        self.left_kid = None
        self.right_kid = None

    def split(self, col_index, split_value):
        n_s, n_f = self.data.shape
        data = sorted(self.data, key = self.data[:,col_index])
        ind = data[:,col_index] <= split_value
        ind  = min([item for item in ind if item])
        if ind:
            self.left_kid = node(data[:,0:ind])
        if ind+1<n_f:
            self.right_kid = node(data[:,(ind+1):])



