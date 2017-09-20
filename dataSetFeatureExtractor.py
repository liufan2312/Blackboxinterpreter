
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
from sklearn.utils import check_array
from sklearn.multiclass import OneVsRestClassifier


class DataSetFeatureExtractor(object):

    featureList = ['number_of_instances',
                    'log_number_of_instances',
                    'number_of_classes',
                    'number_of_features',
                    'log_number_of_feature',
                    'missing_values',
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
                    'class_probability_STD',
                    'num_symbols',
                    'symbols_min',
                    'symbols_max',
                    'symbols_mean',
                    'symbols_STD',
                    'symbols_sum',
                    'kurtosis_ses',
                    'kurtosis_min',
                    'kurtosis_max',
                    'kurtosis_mean',
                    'kurtosis_STD',
                    # 'Skewnesses',
                    'skewness_min',
                    'skewness_max',
                    'skewness_mean',
                    'skewness_STD',
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
                    'landmark_LDA',
                    'landmark_naive_Bayes',
                    'landmark_decision_tree',
                    'landmark_decision_node_learner',
                    'landmark_random_node_learner',
                    'landmark_1NN',
                    'PCA',
                    'PCA_fraction_of_components_for_95_percent_variance',
                    'PCA_Kurtosis_first_PC',
                    'PCA_skewness_first_PC']

    def __init__(self, x, y):

        self.x = x
        self.y = y
        self.numberOfInstances = float(x.shape[0])
        self.numberOfFeatures = float(x.shape[1])
        self.missing = None

    def get_feature_list(self):
        return self.featureList

    def extract_features(self, feature_used):
        features = []
        for feature in feature_used:
            if feature in self.feature_list:
                features.append(getattr(self, feature)())

            else:
                features.append(None)
        return features

    def number_of_instances(self):
        return self.numberOfInstances

    def log_number_of_instances(self):
        return np.log(self.numberOfInstances)

    def number_of_classes(self):
        if len(self.y.shape) == 2:
            return np.mean([len(np.unique(self.y[:, i])) for i in range(self.y.shape[1])])
        else:
            return float(len(np.unique(self.y)))

    def number_of_features(self):
        return self.numberOfFeatures

    def log_number_of_features(self):
        return np.log(self.numberOfFeatures)

    def missing_values(self):
        if not sps.issparse(self.x):
            self.missing = ~np.isfinite(self.x)

        else:
            data = [True if not np.isfinite(item) else False for item in self.x.data]
            self.missing = self.x.__class__((data, self.x.indices, self.x.indptr), shape=self.x.shape, dtype=np.bool)

    def number_of_instances_with_missing_values(self, x, y):
        if not sps.issparse(x):
            missing = DataSetFeatureExtractor.missing_values(x, y)
            num_missing = missing.sum(axis=1)
            return float(np.sum([1 if num > 0 else 0 for num in num_missing]))

        else:
            missing = DataSetFeatureExtractor.missing_values(x, y)
            new_missing = missing.tocsr()
            num_missing = [np.sum(new_missing.data[new_missing.indptr[i]:new_missing.indptr[i + 1]]) for i in range(new_missing.shape[0])]

        return float(np.sum([1 if num > 0 else 0 for num in num_missing]))

    @staticmethod
    def percentage_of_instances_with_missing_values(x, y):
        return DataSetFeatureExtractor.number_of_instances_with_missing_values(x, y) / \
        DataSetFeatureExtractor.number_of_instances(x, y)

    @staticmethod
    def number_of_features_with_missing_values(x, y):
        if not sps.issparse(x):
            missing = DataSetFeatureExtractor.missing_values(x, y)
            num_missing = missing.sum(axis=0)
            return float(np.sum([1 if num > 0 else 0 for num in num_missing]))

        else:
            missing = DataSetFeatureExtractor.missing_values(x, y)
            new_missing = missing.tocsc()
            num_missing = [np.sum(new_missing.data[new_missing.indptr[i]:new_missing.indptr[i + 1]]) for i in range(missing.shape[1])]

        return float(np.sum([1 if num > 0 else 0 for num in num_missing]))

    @staticmethod
    def percentage_of_features_with_missing_values(x, y):
        return DataSetFeatureExtractor.number_of_features_with_missing_values(x, y) / \
        DataSetFeatureExtractor.number_of_features(x, y)

    @staticmethod
    def number_of_missing_values(x, y):
        return float(DataSetFeatureExtractor.missing_values(x, y).sum())

    @staticmethod
    def percentage_of_missing_values(x, y):
        return float(DataSetFeatureExtractor.number_of_missing_values(x, y)) / float(x.shape[0] * x.shape[1])

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
    @ staticmethod
    def data_set_ratio(x, y):
        return float(DataSetFeatureExtractor.number_of_features(x, y)) / \
        float(DataSetFeatureExtractor.number_of_instances(x, y))

    @staticmethod
    def log_data_set_ratio(x, y):
        return np.log(DataSetFeatureExtractor.data_set_ratio(x, y))

    @staticmethod
    def inverse_data_set_ratio(x, y):
        return float(1 / DataSetFeatureExtractor.number_of_instances(x, y))

    @staticmethod
    def log_inverse_data_set_ratio(x, y):
        return np.log(DataSetFeatureExtractor.inverse_data_set_ratio(x, y))
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
    #  be the counterpart for the skewness and kurtosis of the numerical features

    def NumSymbols(X, y, categorical):
        if not sps.issparse(X):
            symbols_per_column = []
            for i, column in enumerate(X.T):
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
    # Statistical meta features
    # Only use third and fourth statistical moment because it is common to
    # standardize for the other two
    # see Engels & Theusinger, 1998 - Using a Data Metric for Preprocessing Advice for Data Mining Applications.


    def Kurtosisses(X, y, categorical):
        if not sps.issparse(X):
            kurts = []
            for i in range(X.shape[1]):
                if not categorical[i]:
                    kurts.append(scipy.stats.kurtosis(X[:, i]))
            return kurts
        else:
            kurts = []
            X_new = X.tocsc()
            for i in range(X_new.shape[1]):
                if not categorical[i]:
                    start = X_new.indptr[i]
                    stop = X_new.indptr[i + 1]
                    kurts.append(scipy.stats.kurtosis(X_new.data[start:stop]))
            return kurts

    def KurtosisMin(X, y, categorical):
        kurts = Kurtosisses(X, y, categorical)
        minimum = np.nanmin(kurts) if len(kurts) > 0 else 0
        return minimum if np.isfinite(minimum) else 0

    def KurtosisMax(X, y, categorical):
        kurts = Kurtosisses(X, y, categorical)
        maximum = np.nanmax(kurts) if len(kurts) > 0 else 0
        return maximum if np.isfinite(maximum) else 0

    def KurtosisMean(X, y, categorical):
        kurts = Kurtosisses(X, y, categorical)
        mean = np.nanmean(kurts) if len(kurts) > 0 else 0
        return mean if np.isfinite(mean) else 0

    def KurtosisSTD(X, y, categorical):
        kurts = Kurtosisses(X, y, categorical)
        std = np.nanstd(kurts) if len(kurts) > 0 else 0
        return std if np.isfinite(std) else 0

    def Skewnesses(X, y, categorical):
        if not sps.issparse(X):
            skews = []
            for i in range(X.shape[1]):
                if not categorical[i]:
                    skews.append(scipy.stats.skew(X[:, i]))
            return skews

        else:
            skews = []
            X_new = X.tocsc()
            for i in range(X_new.shape[1]):
                if not categorical[i]:
                    start = X_new.indptr[i]
                    stop = X_new.indptr[i + 1]
                    skews.append(scipy.stats.skew(X_new.data[start:stop]))
            return skews

    def SkewnessMin(X, y, categorical):
        skews = Skewnesses(X, y, categorical)
        minimum = np.nanmin(skews) if len(skews) > 0 else 0
        return minimum if np.isfinite(minimum) else 0

    def SkewnessMax(X, y, categorical):
        skews = Skewnesses(X, y, categorical)
        maximum = np.nanmax(skews) if len(skews) > 0 else 0
        return maximum if np.isfinite(maximum) else 0

    def SkewnessMean(X, y, categorical):
        skews = Skewnesses(X, y, categorical)
        mean = np.nanmean(skews) if len(skews) > 0 else 0
        return mean if np.isfinite(mean) else 0

    def SkewnessSTD(X, y, categorical):
        skews = Skewnesses(X, y, categorical)
        std = np.nanstd(skews) if len(skews) > 0 else 0
        return std if np.isfinite(std) else 0

    # @metafeatures.define("cancor1")
    # def cancor1(X, y):
    #    pass

    # @metafeatures.define("cancor2")
    # def cancor2(X, y):
    #    pass

    ################################################################################
    # Information-theoretic metafeatures

    def ClassEntropy(X, y, categorical):
        labels = 1 if len(y.shape) == 1 else y.shape[1]
        if labels == 1:
            y = y.reshape((-1, 1))

        entropies = []
        for i in range(labels):
            occurence_dict = defaultdict(float)
            for value in y[:, i]:
                occurence_dict[value] += 1
            entropies.append(scipy.stats.entropy([occurence_dict[key] for key in occurence_dict], base=2))

        return np.mean(entropies)

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

    ################################################################################
    # Landmarking features, computed with cross validation
    # These should be invoked with the same transformations of X and y with which
    # sklearn will be called later on

    # from Pfahringer 2000
    # Linear discriminant learner

    def LandmarkLDA(X, y, categorical):
        if not sps.issparse(X):
            import sklearn.discriminant_analysis
            if len(y.shape) == 1 or y.shape[1] == 1:
                kf = sklearn.model_selection.StratifiedKFold(n_splits=10)
            else:
                kf = sklearn.model_selection.KFold(n_splits=10)

            accuracy = 0.
            try:
                for train, test in kf.split(X, y):
                    lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()

                    if len(y.shape) == 1 or y.shape[1] == 1:
                        lda.fit(X[train], y[train])
                    else:
                        lda = OneVsRestClassifier(lda)
                        lda.fit(X[train], y[train])

                    predictions = lda.predict(X[test])
                    accuracy += sklearn.metrics.accuracy_score(predictions, y[test])
                return accuracy / 10
            except scipy.linalg.LinAlgError as e:
                # self.logger.warning("LDA failed: %s Returned 0 instead!" % e)
                return np.NaN
            except ValueError as e:
                # self.logger.warning("LDA failed: %s Returned 0 instead!" % e)
                return np.NaN


        else:
            return np.NaN

    # Naive Bayes

    def LandmarkNaiveBayes(X, y, categorical):
        if not sps.issparse(X):
            import sklearn.naive_bayes

            if len(y.shape) == 1 or y.shape[1] == 1:
                kf = sklearn.model_selection.StratifiedKFold(n_splits=10)
            else:
                kf = sklearn.model_selection.KFold(n_splits=10)

            accuracy = 0.
            for train, test in kf.split(X, y):
                nb = sklearn.naive_bayes.GaussianNB()

                if len(y.shape) == 1 or y.shape[1] == 1:
                    nb.fit(X[train], y[train])
                else:
                    nb = OneVsRestClassifier(nb)
                    nb.fit(X[train], y[train])

                predictions = nb.predict(X[test])
                accuracy += sklearn.metrics.accuracy_score(predictions, y[test])
            return accuracy / 10

        else:
            return np.NaN

    def LandmarkDecisionTree(X, y, categorical):
        if not sps.issparse(X):
            import sklearn.tree

            if len(y.shape) == 1 or y.shape[1] == 1:
                kf = sklearn.model_selection.StratifiedKFold(n_splits=10)
            else:
                kf = sklearn.model_selection.KFold(n_splits=10)

            accuracy = 0.
            for train, test in kf.split(X, y):
                random_state = sklearn.utils.check_random_state(42)
                tree = sklearn.tree.DecisionTreeClassifier(random_state=random_state)

                if len(y.shape) == 1 or y.shape[1] == 1:
                    tree.fit(X[train], y[train])
                else:
                    tree = OneVsRestClassifier(tree)
                    tree.fit(X[train], y[train])

                predictions = tree.predict(X[test])
                accuracy += sklearn.metrics.accuracy_score(predictions, y[test])
            return accuracy / 10

        else:
            return np.NaN

    """If there is a dataset which has OneHotEncoded features it can happend that
    a node learner splits at one of the attribute encodings. This should be fine
    as the dataset is later on used encoded."""

    # TODO: use the same tree, this has then to be computed only once and hence
    #  saves a lot of time...

    def LandmarkDecisionNodeLearner(X, y, categorical):
        if not sps.issparse(X):
            import sklearn.tree

            if len(y.shape) == 1 or y.shape[1] == 1:
                kf = sklearn.model_selection.StratifiedKFold(n_splits=10)
            else:
                kf = sklearn.model_selection.KFold(n_splits=10)

            accuracy = 0.
            for train, test in kf.split(X, y):
                random_state = sklearn.utils.check_random_state(42)
                node = sklearn.tree.DecisionTreeClassifier(
                    criterion="entropy", max_depth=1, random_state=random_state,
                    min_samples_split=2, min_samples_leaf=1, max_features=None)
                if len(y.shape) == 1 or y.shape[1] == 1:
                    node.fit(X[train], y[train])
                else:
                    node = OneVsRestClassifier(node)
                    node.fit(X[train], y[train])
                predictions = node.predict(X[test])
                accuracy += sklearn.metrics.accuracy_score(predictions, y[test])
            return accuracy / 10

        else:
            return np.NaN

    def LandmarkRandomNodeLearner(X, y, categorical):
        if not sps.issparse(X):
            import sklearn.tree

            if len(y.shape) == 1 or y.shape[1] == 1:
                kf = sklearn.model_selection.StratifiedKFold(n_splits=10)
            else:
                kf = sklearn.model_selection.KFold(n_splits=10)
            accuracy = 0.

            for train, test in kf.split(X, y):
                random_state = sklearn.utils.check_random_state(42)
                node = sklearn.tree.DecisionTreeClassifier(
                    criterion="entropy", max_depth=1, random_state=random_state,
                    min_samples_split=2, min_samples_leaf=1, max_features=1)
                node.fit(X[train], y[train])
                predictions = node.predict(X[test])
                accuracy += sklearn.metrics.accuracy_score(predictions, y[test])
            return accuracy / 10

        else:
            return np.NaN

    # Replace the Elite 1NN with a normal 1NN, this slightly changes the
    # intuition behind this landmark, but Elite 1NN is used nowhere else...


    def Landmark1NN(X, y, categorical):
        import sklearn.neighbors

        if len(y.shape) == 1 or y.shape[1] == 1:
            kf = sklearn.model_selection.StratifiedKFold(n_splits=10)
        else:
            kf = sklearn.model_selection.KFold(n_splits=10)

        accuracy = 0.
        for train, test in kf.split(X, y):
            kNN = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
            if len(y.shape) == 1 or y.shape[1] == 1:
                kNN.fit(X[train], y[train])
            else:
                kNN = OneVsRestClassifier(kNN)
                kNN.fit(X[train], y[train])
            predictions = kNN.predict(X[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, y[test])
        return accuracy / 10

    ################################################################################
    # Bardenet 2013 - Collaborative Hyperparameter Tuning
    # K number of classes ("number_of_classes")
    # log(d), log(number of attributes)
    # log(n/d), log(number of training instances/number of attributes)
    # p, how many principal components to keep in order to retain 95% of the
    #     dataset variance
    # skewness of a dataset projected onto one principal component...
    # kurtosis of a dataset projected onto one principal component


    def PCA(X, y, categorical):
        if not sps.issparse(X):
            import sklearn.decomposition
            pca = sklearn.decomposition.PCA(copy=True)
            rs = np.random.RandomState(42)
            indices = np.arange(X.shape[0])
            for i in range(10):
                try:
                    rs.shuffle(indices)
                    pca.fit(X[indices])
                    return pca
                except LinAlgError as e:
                    pass
            # self.logger.warning("Failed to compute a Principle Component Analysis")
            return None

        else:
            import sklearn.decomposition
            rs = np.random.RandomState(42)
            indices = np.arange(X.shape[0])
            # This is expensive, but necessary with scikit-learn 0.15
            Xt = X.astype(np.float64)
            for i in range(10):
                try:
                    rs.shuffle(indices)
                    truncated_svd = sklearn.decomposition.TruncatedSVD(
                        n_components=X.shape[1] - 1, random_state=i,
                        algorithm="randomized")
                    truncated_svd.fit(Xt[indices])
                    return truncated_svd
                except LinAlgError as e:
                    pass
            # self.logger.warning("Failed to compute a Truncated SVD")
            return None

    def PCAFractionOfComponentsFor95PercentVariance(X, y, categorical):
        pca_ = PCA(X, y, categorical)
        if pca_ is None:
            return np.NaN
        sum_ = 0.
        idx = 0
        while sum_ < 0.95 and idx < len(pca_.explained_variance_ratio_):
            sum_ += pca_.explained_variance_ratio_[idx]
            idx += 1
        return float(idx) / float(X.shape[1])

    def PCAKurtosisFirstPC(X, y, categorical):
        pca_ = PCA(X, y, categorical)
        if pca_ is None:
            return np.NaN
        components = pca_.components_
        pca_.components_ = components[:1]
        transformed = pca_.transform(X)
        pca_.components_ = components

        kurtosis = scipy.stats.kurtosis(transformed)
        return kurtosis[0]

    # Skewness of first PC

    def PCASkewnessFirstPC(X, y, categorical):
        pca_ = PCA(X, y, categorical)
        if pca_ is None:
            return np.NaN
        components = pca_.components_
        pca_.components_ = components[:1]
        transformed = pca_.transform(X)
        pca_.components_ = components

        skewness = scipy.stats.skew(transformed)
        return skewness[0]
'''
