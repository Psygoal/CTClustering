import numpy as np

class Jointdata(object):
    def gen_data(self, data, **kwargs):
        
        joint_data = self.gen_time_series_data(data,**kwargs)
            
        return joint_data
    
    def gen_time_series_data(self, data, lag_time = 1):

        pxy_data = np.concatenate((data[:-lag_time], data[lag_time:]), axis=-1)
        pyx_data = np.concatenate((data[lag_time:], data[:-lag_time]), axis=-1)

        return pxy_data,pyx_data
