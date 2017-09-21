from dataSetFeatureExtractor import DataSetFeatureExtractor
import numpy as np

x = np.random.rand(100, 10)
y = np.random.randint(2, size=(100, 1))

x[0, 0] = np.inf
x[1, 0] = np.nan

l = DataSetFeatureExtractor(x, y).extract_features(['number_of_instances',
                                                    'log_number_of_instances',
                                                    'number_of_classes',
                                                    'number_of_features',
                                                    'log_number_of_features',
                                                    'number_of_instances_with_missing_values',
                                                    'percentage_of_instances_with_missing_values',
                                                    'number_of_features_with_missing_values',
                                                    'percentage_of_features_with_missing_values',
                                                    'number_of_missing_values',
                                                    'percentage_of_missing_values',
                                                    'data_set_ratio',
                                                    'log_data_set_ratio',
                                                    'inverse_data_set_ratio',
                                                    'log_inverse_data_set_ratio',
                                                    'kurtosis_min',
                                                    'kurtosis_max',
                                                    'kurtosis_mean',
                                                    'kurtosis_std',
                                                    'skewness_min',
                                                    'skewness_max',
                                                    'skewness_mean',
                                                    'skewness_std',
                                                    'class_entropy',
                                                    'pca_fraction_of_components_for_95_percent_variance',
                                                    'pca_kurtosis_first_pc',
                                                    'pca_skewness_first_pc'])
print l
