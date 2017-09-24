from dataSetFeatureExtractor import DataSetFeatureExtractor
import numpy as np

x = np.random.rand(100, 10)
y = np.random.randint(2, size=100)

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
                                                    # 'number_of_numeric_features',
                                                    # 'number_of_categorical_features',
                                                    # 'ratio_numerical_to_nominal',
                                                    # 'ratio_nominal_to_numerical',
                                                    'data_set_ratio',
                                                    'log_data_set_ratio',
                                                    'inverse_data_set_ratio',
                                                    'log_inverse_data_set_ratio',
                                                    'class_probability_min',
                                                    'class_probability_max',
                                                    'class_probability_mean',
                                                    'class_probability_std',
                                                    # 'num_symbols',
                                                    # 'symbols_min',
                                                    # 'symbols_max',
                                                    # 'symbols_mean',
                                                    # 'symbols_std',
                                                    # 'symbols_sum',
                                                    'kurtosis_min',
                                                    'kurtosis_max',
                                                    'kurtosis_mean',
                                                    'kurtosis_std',
                                                    'skewness_min',
                                                    'skewness_max',
                                                    'skewness_mean',
                                                    'skewness_std',
                                                    'class_entropy',
                                                    'landmark_lda',
                                                    'landmark_naive_bayes',
                                                    'landmark_decision_tree',
                                                    'landmark_decision_node_learner',
                                                    'landmark_random_node_learner',
                                                    'landmark_1nn',
                                                    'pca_fraction_of_components_for_95_percent_variance',
                                                    'pca_kurtosis_first_pc',
                                                    'pca_skewness_first_pc'])
print len(l)
print l
