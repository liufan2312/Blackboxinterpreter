
from collections import defaultdict, OrderedDict, deque
import copy
import sys


import numpy as np
import scipy.stats
from scipy.linalg import LinAlgError
import scipy.sparse as sps
import sklearn
import sklearn.metrics
import sklearn.model_selection
import sklearn.decomposition
from sklearn.utils import check_array
from sklearn.multiclass import OneVsRestClassifier


class DataSetFeatureExtractor(object):

    featureList = ['number_of_instances',
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
                   'number_of_numeric_features',
                   'number_of_categorical_features',
                   'ratio_numerical_to_nominal',
                   'ratio_nominal_to_numerical',
                   'data_set_ratio',
                   'log_data_set_ratio',
                   'inverse_data_set_ratio',
                   'log_inverse_data_set_ratio',
                   'class_occurrences',
                   'class_probability_min',
                   'class_probability_max',
                   'class_probability_mean',
                   'class_probability_std',
                   'num_symbols',
                   'symbols_min',
                   'symbols_max',
                   'symbols_mean',
                   'symbols_STD',
                   'symbols_sum',
                   'kurtosis_min',
                   'kurtosis_max',
                   'kurtosis_mean',
                   'kurtosis_std',
                   # 'Skewnesses',
                   'skewness_min',
                   'skewness_max',
                   'skewness_mean',
                   'skewness_std',
                   'class_entropy',
                   # @metafeatures.define("normalized_class_entropy")
                   # @metafeatures.define("attribute_entropy")
                   # @metafeatures.define("normalized_attribute_entropy")
                   # @metafeatures.define("joint_entropy")
                   # @metafeatures.define("mutual_information")
                   # @metafeatures.define("noise-signal-ratio")
                   # @metafeatures.define("signal-noise-ratio")
                   # @metafeatures.define("equivalent_number_of_attributes")
                   # @metafeatures.define("conditional_entropy")
                   # @metafeatures.define("average_attribute_entropy")
                   'landmark_lda',
                   'landmark_naive_bayes',
                   'landmark_decision_tree',
                   'landmark_decision_node_learner',
                   'landmark_random_node_learner',
                   'landmark_1nn',
                   'pca_fraction_of_components_for_95_percent_variance',
                   'pca_kurtosis_first_pc',
                   'pca_skewness_first_pc']

    def __init__(self, x, y):

        if type(x) is not np.ndarray or type(y) is not np.ndarray:
            print "scream!!!"

        self.x = x
        self.y = y
        self.numberOfInstances = x.shape[0]
        self.numberOfFeatures = x.shape[1]

        # PCA calculate can't take NaN entries, so we remove those
        mask = ~np.any(np.isnan(self.x), axis=1)
        self.x_clean = self.x[mask]
        self.y_clean = self.y[mask]

        if not sps.issparse(self.x):
            self.missing = ~np.isfinite(self.x)

            self.kurts = []
            self.skews = []

            for i in range(self.numberOfFeatures):
                self.kurts.append(scipy.stats.kurtosis(self.x[:, i]))
                self.skews.append(scipy.stats.skew(self.x[:, i]))

            # for i in range(self.numberOfFeatures):
            #     if not categorical[i]:
            #         kurts.append(scipy.stats.kurtosis(X[:, i]))

        else:
            data = [True if not np.isfinite(item) else False for item in self.x.data]
            self.missing = self.x.__class__((data, self.x.indices, self.x.indptr), shape=self.x.shape, dtype=np.bool)

            self.kurts = []
            self.skews = []

            x_new = self.x.tocsc()
            for i in range(x_new.shape[1]):
                start = x_new.indptr[i]
                stop = x_new.indptr[i + 1]
                self.kurts.append(scipy.stats.kurtosis(x_new.data[start:stop]))
                self.skews.append(scipy.stats.skew(x_new.data[start:stop]))

    def get_feature_list(self):

        return self.featureList

    def extract_features(self, feature_used):
        features = []
        for feature in feature_used:
            if feature in self.featureList:
                features.append(getattr(self, feature)())

            else:
                features.append(None)

        return features

    def number_of_instances(self):

        return float(self.numberOfInstances)

    def log_number_of_instances(self):

        return np.log(self.numberOfInstances)

    def number_of_classes(self):
        if len(self.y.shape) == 2:

            return np.mean([len(np.unique(self.y[:, i])) for i in range(self.y.shape[1])])

        else:

            return float(len(np.unique(self.y)))

    def number_of_features(self):

        return float(self.numberOfFeatures)

    def log_number_of_features(self):

        return np.log(self.numberOfFeatures)

    def number_of_instances_with_missing_values(self):
        if not sps.issparse(self.x):
            num_missing = self.missing.sum(axis=1)

            return float(np.sum([1 if num > 0 else 0 for num in num_missing]))

        else:
            new_missing = self.missing.tocsr()
            num_missing = [np.sum(new_missing.data[new_missing.indptr[i]:new_missing.indptr[i + 1]]) for i in range(new_missing.shape[0])]

        return float(np.sum([1 if num > 0 else 0 for num in num_missing]))

    def percentage_of_instances_with_missing_values(self):

        return DataSetFeatureExtractor.number_of_instances_with_missing_values(self) / \
               DataSetFeatureExtractor.number_of_instances(self)

    def number_of_features_with_missing_values(self):
        if not sps.issparse(self.x):
            num_missing = self.missing.sum(axis=0)

            return float(np.sum([1 if num > 0 else 0 for num in num_missing]))

        else:
            new_missing = self.missing.tocsc()
            num_missing = [np.sum(new_missing.data[new_missing.indptr[i]:new_missing.indptr[i + 1]]) for i in range(missing.shape[1])]

        return float(np.sum([1 if num > 0 else 0 for num in num_missing]))

    def percentage_of_features_with_missing_values(self):

        return DataSetFeatureExtractor.number_of_features_with_missing_values(self) / \
               DataSetFeatureExtractor.number_of_features(self)

    def number_of_missing_values(self):

        return float(self.missing.sum())

    def percentage_of_missing_values(self):

        return float(DataSetFeatureExtractor.number_of_missing_values(self)) / \
               (self.numberOfInstances * self.numberOfFeatures)

    # @staticmethod
    # def number_of_numeric_features(x, y):
    #     return len(categorical) - np.sum(categorical)

    # def NumberOfCategoricalFeatures(X, y, categorical):
    #     return np.sum(categorical)

    # def RatioNumericalToNominal(X, y, categorical):
    #     num_categorical = float(NumberOfCategoricalFeatures(X, y, categorical))
    #     num_numerical = float(NumberOfNumericFeatures(X, y, categorical))
    #     if num_categorical == 0.0:
    #        return 0.
    #     return num_numerical / num_categorical

    # def RatioNominalToNumerical(X, y, categorical):
    #     num_categorical = float(NumberOfCategoricalFeatures(X, y, categorical))
    #     num_numerical = float(NumberOfNumericFeatures(X, y, categorical))
    #     if num_numerical == 0.0:
    #         return 0.
    #     else:
    #         return num_categorical / num_numerical

    # Number of attributes divided by number of samples

    def data_set_ratio(self):

        return self.numberOfFeatures / float(self.numberOfInstances)

    def log_data_set_ratio(self):

        return np.log(DataSetFeatureExtractor.data_set_ratio(self))

    def inverse_data_set_ratio(self):

        return float(1 / DataSetFeatureExtractor.number_of_instances(self))

    def log_inverse_data_set_ratio(self):

        return np.log(DataSetFeatureExtractor.inverse_data_set_ratio(self))

    ##################################################################################
    # Statistical meta features
    # Only use third and fourth statistical moment because it is common to standardize
    # for the other two, see Engels & Theusinger, 1998 - Using a Data Metric for
    # Preprocessing Advice for Data Mining Applications.

    def kurtosis_min(self):
        minimum = np.nanmin(self.kurts) if len(self.kurts) > 0 else 0

        return minimum if np.isfinite(minimum) else 0

    def kurtosis_max(self):
        maximum = np.nanmax(self.kurts) if len(self.kurts) > 0 else 0

        return maximum if np.isfinite(maximum) else 0

    def kurtosis_mean(self):
        mean = np.nanmean(self.kurts) if len(self.kurts) > 0 else 0

        return mean if np.isfinite(mean) else 0

    def kurtosis_std(self):
        std = np.nanstd(self.kurts) if len(self.kurts) > 0 else 0

        return std if np.isfinite(std) else 0

    def skewness_min(self):
        minimum = np.nanmin(self.skews) if len(self.skews) > 0 else 0

        return minimum if np.isfinite(minimum) else 0

    def skewness_max(self):
        maximum = np.nanmax(self.skews) if len(self.skews) > 0 else 0

        return maximum if np.isfinite(maximum) else 0

    def skewness_mean(self):
        mean = np.nanmean(self.skews) if len(self.skews) > 0 else 0

        return mean if np.isfinite(mean) else 0

    def skewness_std(self):
        std = np.nanstd(self.skews) if len(self.skews) > 0 else 0

        return std if np.isfinite(std) else 0

    def class_entropy(self):
        labels = 1 if len(self.y.shape) == 1 else self.y.shape[1]
        new_y = self.y.reshape((-1, 1)) if labels == 1 else self.y

        entropies = []
        for i in range(labels):
            occurrence_dict = defaultdict(float)

            for value in new_y[:, i]:
                occurrence_dict[value] += 1
            entropies.append(scipy.stats.entropy([occurrence_dict[key] for key in occurrence_dict], base=2))

        return np.mean(entropies)

    ################################################################################
    # Bardenet 2013 - Collaborative Hyperparameter Tuning
    # K number of classes ("number_of_classes")
    # log(d), log(number of attributes)
    # log(n/d), log(number of training instances/number of attributes)
    # p, how many principal components to keep in order to retain 95% of the
    # dataset variance
    # skewness of a dataset projected onto one principal component...
    # kurtosis of a dataset projected onto one principal component

    def pca(self):
        if not sps.issparse(self.x_clean):
            pca = sklearn.decomposition.PCA(copy=True)
            rs = np.random.RandomState(42)
            indices = np.arange(self.x_clean.shape[0])

            try:
                rs.shuffle(indices)
                pca.fit(self.x_clean[indices])

                return pca

            except LinAlgError as e:
                pass
            # self.logger.warning("Failed to compute a Principle Component Analysis")
            return None

        else:
            rs = np.random.RandomState(42)
            indices = np.arange(self.x_clean.shape[0])
            xt = self.x_clean.astype(np.float64)

            for i in range(10):
                try:
                    rs.shuffle(indices)
                    truncated_svd = sklearn.decomposition.TruncatedSVD(
                        n_components=_new.shape[1] - 1, random_state=i,
                        algorithm="randomized")

                    truncated_svd.fit(xt[indices])

                    return truncated_svd

                except LinAlgError as e:
                    pass
            # self.logger.warning("Failed to compute a Truncated SVD")
            return None

    def pca_fraction_of_components_for_95_percent_variance(self):
        pca_ = DataSetFeatureExtractor.pca(self)
        if pca_ is None:

            return np.NaN

        sum_ = 0.
        idx = 0

        while sum_ < 0.95 and idx < len(pca_.explained_variance_ratio_):
            sum_ += pca_.explained_variance_ratio_[idx]
            idx += 1

        return float(idx) / float(self.x_clean.shape[1])

    def pca_kurtosis_first_pc(self):
        pca_ = DataSetFeatureExtractor.pca(self)
        if pca_ is None:

            return np.NaN

        components = pca_.components_
        pca_.components_ = components[:1]
        transformed = pca_.transform(self.x_clean)
        pca_.components_ = components

        kurtosis = scipy.stats.kurtosis(transformed)

        return kurtosis[0]

    def pca_skewness_first_pc(self):
        pca_ = DataSetFeatureExtractor.pca(self)
        if pca_ is None:

            return np.NaN

        components = pca_.components_
        pca_.components_ = components[:1]
        transformed = pca_.transform(self.x_clean)
        pca_.components_ = components

        skewness = scipy.stats.skew(transformed)
        
        return skewness[0]

'''
    @staticmethod
    def class_occurrences(x, y):
        if len(y.shape) == 2:
            occurrences = []
            for i in range(y.shape[1]):
                occurrences.append(self._calculate(x, y))
            return occurrences
        else:
            occurence_dict = defaultdict(float)
            for value in y:
                occurence_dict[value] += 1
            return occurence_dict

    def ClassProbabilityMin(X, y, categorical):
        occurences = ClassOccurences(X, y, categorical)

        min_value = np.iinfo(np.int64).max
        if len(y.shape) == 2:
            for i in range(y.shape[1]):
                for num_occurences in occurences[i].values():
                    if num_occurences < min_value:
                        min_value = num_occurences
        else:
            for num_occurences in occurences.values():
                if num_occurences < min_value:
                    min_value = num_occurences
        return float(min_value) / float(y.shape[0])

    def ClassProbabilityMax(X, y, categorical):
        occurences = ClassOccurences(X, y, categorical)
        max_value = -1

        if len(y.shape) == 2:
            for i in range(y.shape[1]):
                for num_occurences in occurences[i].values():
                    if num_occurences > max_value:
                        max_value = num_occurences
        else:
            for num_occurences in occurences.values():
                if num_occurences > max_value:
                    max_value = num_occurences
        return float(max_value) / float(y.shape[0])

    def ClassProbabilityMean(X, y, categorical):
        occurence_dict = ClassOccurences(X, y, categorical)

        if len(y.shape) == 2:
            occurences = []
            for i in range(y.shape[1]):
                occurences.extend(
                    [occurrence for occurrence in occurence_dict[
                        i].values()])
            occurences = np.array(occurences)
        else:
            occurences = np.array([occurrence for occurrence in occurence_dict.values()],
                                  dtype=np.float64)
        return (occurences / y.shape[0]).mean()

    def ClassProbabilitySTD(X, y, categorical):
        occurence_dict = ClassOccurences(X, y, categorical)

        if len(y.shape) == 2:
            stds = []
            for i in range(y.shape[1]):
                std = np.array(
                    [occurrence for occurrence in occurence_dict[
                        i].values()],
                    dtype=np.float64)
                std = (std / y.shape[0]).std()
                stds.append(std)
            return np.mean(stds)
        else:
            occurences = np.array([occurrence for occurrence in occurence_dict.values()],
                                  dtype=np.float64)
            return (occurences / y.shape[0]).std()
            
    ################################################################################
    # Reif, A Comprehensive Dataset for Evaluating Approaches of various Meta-Learning Tasks
    # defines these five metafeatures as simple metafeatures, but they could also
    # be the counterpart for the skewness and kurtosis of the numerical features

    def num_symbols(self, X, y, categorical):
        if not sps.issparse(self.x):
            symbols_per_column = []
            for i, column in enumerate(self.x.T):
                if categorical[i]:
                    unique_values = np.unique(column)
                    num_unique = np.sum(np.isfinite(unique_values))
                    symbols_per_column.append(num_unique)
            return symbols_per_column

        else:
            symbols_per_column = []
            new_X = X.tocsc()
            for i in range(new_X.shape[1]):
                if categorical[i]:
                    unique_values = np.unique(new_X.getcol(i).data)
                    num_unique = np.sum(np.isfinite(unique_values))
                    symbols_per_column.append(num_unique)
            return symbols_per_column

    def SymbolsMin(X, y, categorical):
        # The minimum can only be zero if there are no nominal features,
        # otherwise it is at least one
        # TODO: shouldn't this rather be two?
        minimum = None
        for unique in NumSymbols(X, y, categorical):
            if unique > 0 and (minimum is None or unique < minimum):
                minimum = unique
        return minimum if minimum is not None else 0

    def SymbolsMax(X, y, categorical):
        values = NumSymbols(X, y, categorical)
        if len(values) == 0:
            return 0
        return max(max(values), 0)

    def SymbolsMean(X, y, categorical):
        # TODO: categorical attributes without a symbol don't count towards this
        # measure
        values = [val for val in NumSymbols(X, y, categorical) if val > 0]
        mean = np.nanmean(values)
        return mean if np.isfinite(mean) else 0

    def SymbolsSTD(X, y, categorical):
        values = [val for val in NumSymbols(X, y, categorical) if val > 0]
        std = np.nanstd(values)
        return std if np.isfinite(std) else 0

    def SymbolsSum(X, y, categorical):
        sum = np.nansum(NumSymbols(X, y, categorical))
        return sum if np.isfinite(sum) else 0


    ################################################################################
    # Landmarking features, computed with cross validation
    # These should be invoked with the same transformations of X and y with which
    # sklearn will be called later on
    # from Pfahringer 2000
    # Linear discriminant learner

    def landmark_lda(self):
        if not sps.issparse(self.x):
            import sklearn.discriminant_analysis

            if len(self.y.shape) == 1 or self.y.shape[1] == 1:
                kf = sklearn.model_selection.StratifiedKFold(n_splits=10)
            else:
                kf = sklearn.model_selection.KFold(n_splits=10)

            accuracy = 0.
            try:
                for train, test in kf.split(self.x, self.y):
                    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()

                    if len(self.y.shape) == 1 or self.y.shape[1] == 1:
                        lda.fit(self.x[train], self.y[train])
                    else:
                        lda = OneVsRestClassifier(lda)
                        lda.fit(self.x[train], self.y[train])

                    predictions = lda.predict(self.x[test])
                    accuracy += sklearn.metrics.accuracy_score(predictions, self.y[test])
                return accuracy / 10
            except scipy.linalg.LinAlgError as e:
                # self.logger.warning("LDA failed: %s Returned 0 instead!" % e)
                return np.NaN
            except ValueError as e:
                # self.logger.warning("LDA failed: %s Returned 0 instead!" % e)
                return np.NaN

        else:
            return np.NaN

    def landmark_naive_bayes(self):
        if not sps.issparse(self.x):
            import sklearn.naive_bayes

            if len(self.y.shape) == 1 or self.y.shape[1] == 1:
                kf = sklearn.model_selection.StratifiedKFold(n_splits=10)
            else:
                kf = sklearn.model_selection.KFold(n_splits=10)

            accuracy = 0.
            for train, test in kf.split(self.x, self.y):
                nb = sklearn.naive_bayes.GaussianNB()

                if len(self.y.shape) == 1 or self.y.shape[1] == 1:
                    nb.fit(self.x[train], self.y[train])
                else:
                    nb = OneVsRestClassifier(nb)
                    nb.fit(self.x[train], self.y[train])

                predictions = nb.predict(self.x[test])
                accuracy += sklearn.metrics.accuracy_score(predictions, self.y[test])
            return accuracy / 10

        else:
            return np.NaN

    def landmark_decision_tree(self):
        if not sps.issparse(self.x):
            import sklearn.tree

            if len(self.y.shape) == 1 or self.y.shape[1] == 1:
                kf = sklearn.model_selection.StratifiedKFold(n_splits=10)
            else:
                kf = sklearn.model_selection.KFold(n_splits=10)

            accuracy = 0.
            for train, test in kf.split(self.x, self.y):
                random_state = sklearn.utils.check_random_state(42)
                tree = sklearn.tree.DecisionTreeClassifier(random_state=random_state)

                if len(self.y.shape) == 1 or self.y.shape[1] == 1:
                    tree.fit(self.x[train], self.y[train])
                else:
                    tree = OneVsRestClassifier(tree)
                    tree.fit(self.x[train], self.y[train])

                predictions = tree.predict(self.x[test])
                accuracy += sklearn.metrics.accuracy_score(predictions, self.y[test])
            return accuracy / 10

        else:
            return np.NaN

    def landmark_decision_node_learner(self):
        if not sps.issparse(self.x):
            import sklearn.tree

            if len(self.y.shape) == 1 or self.y.shape[1] == 1:
                kf = sklearn.model_selection.StratifiedKFold(n_splits=10)
            else:
                kf = sklearn.model_selection.KFold(n_splits=10)

            accuracy = 0.
            for train, test in kf.split(self.x, self.y):
                random_state = sklearn.utils.check_random_state(42)
                node = sklearn.tree.DecisionTreeClassifier(
                    criterion="entropy", max_depth=1, random_state=random_state,
                    min_samples_split=2, min_samples_leaf=1, max_features=None)
                if len(self.y.shape) == 1 or self.y.shape[1] == 1:
                    node.fit(self.x[train], self.y[train])
                else:
                    node = OneVsRestClassifier(node)
                    node.fit(self.x[train], self.y[train])
                predictions = node.predict(self.x[test])
                accuracy += sklearn.metrics.accuracy_score(predictions, self.y[test])
            return accuracy / 10

        else:
            return np.NaN

    def landmark_random_node_learner(self):
        if not sps.issparse(self.x):
            import sklearn.tree

            if len(self.y.shape) == 1 or self.y.shape[1] == 1:
                kf = sklearn.model_selection.StratifiedKFold(n_splits=10)
            else:
                kf = sklearn.model_selection.KFold(n_splits=10)
            accuracy = 0.

            for train, test in kf.split(self.x, self.y):
                random_state = sklearn.utils.check_random_state(42)
                node = sklearn.tree.DecisionTreeClassifier(
                    criterion="entropy", max_depth=1, random_state=random_state,
                    min_samples_split=2, min_samples_leaf=1, max_features=1)
                node.fit(self.x[train], self.y[train])
                predictions = node.predict(self.x[test])
                accuracy += sklearn.metrics.accuracy_score(predictions, self.y[test])
            return accuracy / 10

        else:
            return np.NaN

    # Replace the Elite 1NN with a normal 1NN, this slightly changes the
    # intuition behind this landmark, but Elite 1NN is used nowhere else...

    def landmark_1nn(self):
        import sklearn.neighbors

        if len(self.y.shape) == 1 or self.y.shape[1] == 1:
            kf = sklearn.model_selection.StratifiedKFold(n_splits=10)
        else:
            kf = sklearn.model_selection.KFold(n_splits=10)

        accuracy = 0.
        for train, test in kf.split(self.x, self.y):
            kNN = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
            if len(self.y.shape) == 1 or self.y.shape[1] == 1:
                kNN.fit(self.x[train], self.y[train])
            else:
                kNN = OneVsRestClassifier(kNN)
                kNN.fit(self.x[train], self.y[train])
            predictions = kNN.predict(self.x[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, self.y[test])
        return accuracy / 10
'''