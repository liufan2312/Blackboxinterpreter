import numpy as np
class utilities(object):

    def __init__(self):
        pass

    @staticmethod
    def get_minimal_error(data, label, list_of_intepreters, dis_continous_flag):
        minimal_error = np.inf
        for model in list_of_intepreters:
            model.fit(data, label)
            label_hat = model.predict(data)
            if dis_continous_flag:
                err = np.sum(label == label_hat)/len(label)
            else:
                err = np.mean((label_hat-label)**2)
            if err<minimal_error:minimal_error = err

        return minimal_error


