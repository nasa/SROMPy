import abc

'''
Abstract class providing a base class for RandomVariable and RandomVector.
'''


class RandomEntity(object):

    @abc.abstractmethod
    def get_dim(self):
        return
