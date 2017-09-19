from dataSetFeatureExtractor import DataSetFeatureExtractor
import numpy as np

x = np.ones((2, 2))
y = np.ones((2, 1))

x[0, 0] = np.inf

l = DataSetFeatureExtractor(x, y).extract_features(['number_of_instances',
                                                    'log_number_of_instances'])
print l