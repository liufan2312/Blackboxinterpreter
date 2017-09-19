class node(object):

    def __init__(self, data, label, type_list):
        '''
        :param data:  data is a numpy array with (sample_size x number of columns, [data point features])
        the last column of the data is the label(or response)
        '''
        self.data = data
        self.label = label
        self.type_list = type_list
        self.left_kid = None
        self.right_kid = None


    def split(self, col_index, split_value):
        n_s, n_f = self.data.shape
        data = zip(self.data, self.label)
        data = sorted(self.data, key = self.data[0][:,col_index])
        ind = data[0][:,col_index] <= split_value
        ind  = min([item for item in ind if item])
        if ind:
            self.left_kid = node(data[0][:,0:ind], data[1][0:ind], self.type_list)
        if ind+1<n_f:
            self.right_kid = node(data[0][:,(ind+1):], data[1][(ind+1):], self.type_list)



