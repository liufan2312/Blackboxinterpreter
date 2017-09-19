from data_set_feature_extractor import DataSetFeatureExtractor
import numpy as np

x = np.ones((2, 2))
y = np.ones((2, 1))

x[0, 0] = np.inf

a = DataSetFeatureExtractor(x, y)
l = a.extract_features(['number_of_instances',
                        'log_number_of_instances',
                        'number_of_instances_with_missing_values',
                        'percentage_of_instances_with_missing_values',
                        'number_of_features_with_missing_values',
                        'percentage_of_features_with_missing_values'])
print l