from networks.FullConnectedNet import FCN
from networks.NinaProNet import NinaProNet
from networks.GengNet import GengNet
from networks.CNN import CNN
from networks.simplenet import SimpleLinearNN
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class NetFactory(object):
    @staticmethod
    def create(name):
        if name == 'FCN':
            return FCN
        if name == 'NinaProNet':
            return NinaProNet
        if name == 'SVM':
            return make_pipeline(StandardScaler(), SVC(gamma='auto'))
        if name == 'GengNet':
            return GengNet
        if name == 'CNN':
            return CNN
        if name == 'simplenet':
            return SimpleLinearNN
        # add your own networks here
        print('unsupported network:', name)
        exit()
