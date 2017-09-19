from dataSetFeatureExtractor import data_set_feature_extraction

class StaticFeatureIntepretator(object):

    def __init__(self, data, label, black_box_model, data_set_feature_extractor, emeprical_error_predictor, list_of_intepreters):
        '''
        :param data:  orginal data
        :param label: label
        :param black_box_model: the model to be intepretated,
        :param data_set_feature_extractor: given a data set extract the static features, only one paramter, static
        means the features only comes from the data not relavbant to intepretable models
        '''
        self.data = data
        self.label = label
        self.black_box_model = black_box_model
        self.black_box_label = black_box_model(data)
        self.data_set_feature_extractor = data_set_feature_extractor

    def fit_emperical_predictor(self):
        #estalish decision tree
        #call random partition several tunes
        #each time return leafs


