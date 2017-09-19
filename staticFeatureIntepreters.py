from decisionTree import decision_tree_I
import utilities
import dataSetFeatureExtractor
from decisionTree import node
class StaticFeatureIntepretator(object):

    def __init__(self, black_box_model,black_box_label, data_set_feature_extractor,
                 emeprical_error_predictor, list_of_intepreters,dis_continous_flag):
        '''
        :param data:  orginal data
        :param label: label
        :param black_box_model: the model to be intepretated,
        :param data_set_feature_extractor: given a data set extract the static features, only one paramter, static
        means the features only comes from the data not relavbant to intepretable models
        '''
        self.black_box_model = black_box_model
        self.black_box_label = black_box_label
        self.data_set_feature_extractor = data_set_feature_extractor
        self.emeprical_error_predictor = emeprical_error_predictor
        self.list_of_intepreters = list_of_intepreters
        self.dis_continous_flag = dis_continous_flag

    # fit the emperical predictor
    def fit(self, data, label, max_split, run_number):
        #estalish decision tree
        #call random partition several tunes
        #each time return leafs
        dataSetFeatureList = []
        minErrorList = []
        for i in range(run_number):
            tree = decision_tree_I(data, label, self.list_of_interpreter, max_split)
            tree.build_tree_uniform_split()
            for leaf in tree.nodes:
                feature = dataSetFeatureExtractor.extract_features(leaf)
                dataSetFeatureList.append(feature)
                error = utilities.get_minimal_error(feature, leaf.label, self.list_of_intepreters, self.dis_continous_flag)
                minErrorList.append(error)
        self.emeprical_error_predictor.fit(dataSetFeatureList,minErrorList)

    # predict data from empirical predictor
    def predict(self, data, label,typeList):
        feature = dataSetFeatureExtractor.extract_features(node(data,label,typeList))
        return self.emeprical_error_predictor.predict(feature)
