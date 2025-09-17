import os
import pickle
import numpy as np
from tensorflow import keras
from scipy.signal import savgol_filter,find_peaks,peak_widths

from .flow import RealNVP
from .jointdata import Jointdata
from .estimating import inference_realnvp as inference
from .calcp import cal_pxy
from .merging import merging

class CTC:
    def __init__(self, dim, lag_time=1, Px_path="px.npy", Px_model_path='px_flow.h5', Pxy_model_path='pxy_flow.h5',\
                 transition_mat_path='transmat.npy'):
        self.dim = dim
        self.lag_time = lag_time
        self.Px_path = Px_path
        self.Px_model_path = Px_model_path
        self.Pxy_model_path = Pxy_model_path
        self.transition_mat_path = transition_mat_path
        
    def fit_predict(self,data=None,save=True,ctc_model_path='ctc.pkl',\
                    px_estimator_params=None,\
                    pxy_estimator_params=None,\
                    px_estimate_params=None,\
                    valley_finding_params=None,\
                    cal_transition_mat_params=None,\
                    merging_params=None):
        self.data = data
        
        px_estimator_params = px_estimator_params if px_estimator_params is not None else {}
        pxy_estimator_params = pxy_estimator_params if pxy_estimator_params is not None else {}
        px_estimate_params = px_estimate_params if px_estimate_params is not None else {}
        valley_finding_params = valley_finding_params if valley_finding_params is not None else {}
        cal_transition_mat_params = cal_transition_mat_params if cal_transition_mat_params is not None else {}
        merging_params = merging_params if merging_params is not None else {}
        
        # marginal probability
        if os.path.exists(self.Px_path):
            Px = np.load(self.Px_path)
            self.Px = Px
            lb = 0.1
            ub = 100 - lb
            Px_lb = np.percentile(Px,lb)
            Px_ub = np.percentile(Px,ub)
            Px_lb_idx = np.where(Px < Px_lb)[0]
            Px_ub_idx = np.where(Px > Px_ub)[0]
            Px[Px_lb_idx] = Px_lb
            Px[Px_ub_idx] = Px_ub
            self.Px = Px
            np.save(self.Px_path, Px)

            self.Px_lb = Px_lb
            self.Px_ub = Px_ub
            
        else:
            self._Px_estimator(**px_estimator_params)
            self._Px_estimate(**px_estimate_params)
        
        # probability valleys
        self._valley_finding(**valley_finding_params)
        indice = self.indice
        labels = self.labels
        
        # transition matrix
        if os.path.exists(self.transition_mat_path):
            self.transition_mat = np.load(self.transition_mat_path)
        else:
            self._Pxy_estimator(**pxy_estimator_params)
            self._cal_representative_metrics()
            self._cal_transition_mat(**cal_transition_mat_params)
            
        # merging result
        result = self._merging(**merging_params)
        if len(result) == 2:
            label_consistent = result[0]
            label_consistent = np.delete(label_consistent,result[1],axis=0)
            indice = np.delete(indice,result[1],axis=0)
            self.delete_indice = result[1]
            self.indice = indice # update indice
            
        else:
            label_consistent = result
                    
        # probability valleys
        valley_indice = np.zeros((len(indice)+1,2),dtype='int64')
        valley_indice[1:,0] = indice[:,1]
        valley_indice[:-1,1] = indice[:,0]
        valley_indice[-1,1] = len(self.Px)
        valley_indice = np.delete(valley_indice,np.where(valley_indice[:,0]==valley_indice[:,1])[0],axis=0)
        assert (valley_indice[:,0]<valley_indice[:,1]).all()
        self.valley_indice = valley_indice
        
        # assign labels
        aggregated_labels = [np.where(label_consistent==i)[0].copy() for i in range(label_consistent.max()+1)]
        subgraph_id_order = np.hstack(aggregated_labels)
        self.aggregated_labels = aggregated_labels
        self.subgraph_id_order = subgraph_id_order
        
        for n,m in enumerate(aggregated_labels):
            for i in m:
                labels[indice[i,0]:indice[i,1]] = n
                
        # deal with probability valleys       
        if valley_indice[0,0] == 0:
            labels[valley_indice[0,0]:valley_indice[0,1]] = labels[valley_indice[0,1]]
            for i in valley_indice[1:]:
                try:
                    if labels[i[0]-1] == labels[i[1]]:
                        labels[i[0]:i[1]] = labels[i[1]]
                    else:
                        labels[i[0]:i[1]] = -1
                except: # i[1] == len(Px)
                    labels[i[0]:i[1]] = labels[i[0]-1]

        else:
            for i in valley_indice:
                try:
                    if labels[i[0]-1] == labels[i[1]]:
                        labels[i[0]:i[1]] = labels[i[1]]
                    else:
                        labels[i[0]:i[1]] = -1
                except: # i[1] == len(Px)
                    labels[i[0]:i[1]] = labels[i[0]-1]
        self.labels = labels
        
        # save logic
        if save:
            if not hasattr(self, 'representative_structures'):
                self._cal_representative_metrics()
                
            if not hasattr(self, 'Pxy_estimator'):
                self._Pxy_estimator(**pxy_estimator_params)
                
            if not hasattr(self, 'Px_estimator'):
                self._Px_estimator(**px_estimator_params)
                
            self.save(ctc_model_path)
    
    def load(self,ctc_model_path,px_estimator_params=None,pxy_estimator_params=None):
        with open(ctc_model_path, 'rb') as f:
            ctc_params_dictionary = pickle.load(f)
        self._Px_estimator(**px_estimator_params)
        self._Pxy_estimator(**pxy_estimator_params)
        
        self.valley_indice = ctc_params_dictionary['valley_indice']
        self.indice = ctc_params_dictionary['indice']
        self.representative_P = ctc_params_dictionary['representative_P']
        self.representative_structures = ctc_params_dictionary['representative_structures']
        self.aggregated_labels = ctc_params_dictionary['aggregated_labels']
        self.Px_lb = ctc_params_dictionary['Px_lb']
        self.Px_ub = ctc_params_dictionary['Px_ub']
        print('CTC is successfully loaded')
        
    def save(self,ctc_model_path):
        # ctc parameters for prediction
        ctc_params_dictionary = {}
        ctc_params_dictionary['valley_indice'] = self.valley_indice
        ctc_params_dictionary['indice'] = self.indice
        ctc_params_dictionary['representative_P'] = self.representative_P
        ctc_params_dictionary['representative_structures'] = self.representative_structures
        ctc_params_dictionary['aggregated_labels'] = self.aggregated_labels
        ctc_params_dictionary['Px_lb'] = self.Px_lb
        ctc_params_dictionary['Px_ub'] = self.Px_ub
        
        with open(ctc_model_path, 'wb') as f:
            pickle.dump(ctc_params_dictionary, f)
        print('CTC model is saved at %s'%(ctc_model_path))
        
    def predict(self,x, inference_batch_size=200000):
        representative_structures = self.representative_structures
        representative_P = self.representative_P
        
        len_representative_structures = len(representative_structures)
        len_x = len(x)
        
        x_repeat = np.repeat(x,len_representative_structures,axis=0)
        representative_structures_repeat = np.repeat(representative_structures[None,...],len_x,axis=0)
        representative_structures_repeat = representative_structures_repeat.reshape((-1,representative_structures.shape[1]))
        
        data_pxy = np.hstack((x_repeat,representative_structures_repeat))
        data_pyx = np.hstack((representative_structures_repeat,x_repeat))
        
        # joint probability
        gauss_trans,log_Jacobian,d2p_dx2,Pxy1 = \
        inference(self.Pxy_estimator,data_pxy,batch_size=inference_batch_size)

        gauss_trans,log_Jacobian,d2p_dx2,Pxy2 = \
        inference(self.Pxy_estimator,data_pyx,batch_size=inference_batch_size)
        Pxy_square = (Pxy1 + Pxy2)
        
        # marginal probability of x
        gauss_trans,log_Jacobian,d2p_dx2,P_x_predict = inference(self.Px_estimator,x,batch_size=inference_batch_size)
        P_x_predict[P_x_predict<self.Px_lb] = self.Px_lb
        P_x_predict[P_x_predict>self.Px_ub] = self.Px_ub
        
        # transition matrix
        Pxy_reshape = Pxy_square.reshape((len_x,len_representative_structures))
        Pxy_reshape -= P_x_predict[...,None] + representative_P[None,...]
        
        PT = np.hstack([Pxy_reshape[:,i].mean(1,keepdims=True).copy() for i in self.aggregated_labels])
        prediction = PT.argmax(1)
        return prediction
    
    def _merging(self, ds_scale=1, bins=400, sigma=2, tolerance=0,\
                  min_cluster_size=100, patience=15,change_patient_tolerence=10):
        
        merging_result = merging(self.transition_mat, ds_scale, bins, sigma, tolerance,\
                                  min_cluster_size, patience, change_patient_tolerence)
        return merging_result
    
    def _cal_representative_metrics(self):
        data = self.data
        Px = self.Px
        indice = self.indice
        
        representative_structures = np.zeros((len(indice),data.shape[1]))
        representative_P = np.zeros(len(indice))

        for i,ind in enumerate(indice):
            representative_P[i] = Px[ind[0]:ind[1]].mean()
            representative_structures[i] = data[ind[0]:ind[1]][Px[ind[0]:ind[1]].argmax()]
            
        self.representative_P = representative_P
        self.representative_structures = representative_structures
        
    def _cal_transition_mat(self,inference_batch_size=1000000):
        transition_mat_path = self.transition_mat_path
        transition_mat = cal_pxy(self.Pxy_estimator, self.representative_structures, self.representative_P, batch=inference_batch_size)
        np.fill_diagonal(transition_mat,transition_mat.max(1)+1)
        np.save(transition_mat_path,transition_mat)
        
        self.transition_mat = transition_mat
        
        
    def _valley_finding(self,window_length=50, polyorder=7, peak_distance=50):
        Px = self.Px
        
        # smooth px
        sequence_length = len(Px)
        svg_px = savgol_filter(-Px,window_length,polyorder,0)
        peaks,_ = find_peaks(svg_px,height=np.percentile(svg_px,1),distance=peak_distance)
        results_half = peak_widths(svg_px, peaks, rel_height=0.33)
        
        # find stable segments
        left = np.int64(peaks - (peaks - results_half[2])*3)
        left = np.append(left,sequence_length)
        
        right =  np.int64((results_half[3] - peaks)*3 + peaks)
        right = np.insert(right,0,0)

        peaks = np.append(peaks,sequence_length)
        indice = np.concatenate([right[...,None],left[...,None]],axis=1)
        indice = np.delete(indice,np.where(indice[:,0]>=indice[:,1])[0],axis=0)
        
        # label
        labels = -1*np.ones(len(Px),dtype='int64')
         
        self.indice = indice
        self.labels = labels
    
    def _Px_estimator(self, output_dim=128, reg=1e-4, num_coupling_layers=12, learning_rate=3e-4, epochs=200,\
                       batch_size=2048, validation_split=0.3, inference_batch_size=200000):
        
        model_path = self.Px_model_path
        dim = self.dim
        
        # build estimator
        Px_estimator = RealNVP(dim,output_dim,reg,num_coupling_layers)
        _ = Px_estimator(np.zeros((32,dim)))
        
        # load or train
        if os.path.exists(model_path):
            print('========load the trained marignal probability estimator at %s========'%(model_path))
            Px_estimator.load_weights(model_path)
            
        else:
            print('========training marignal probability estimator========')
            data = self.data
            savebestmodel = keras.callbacks.ModelCheckpoint(model_path,save_best_only=True,save_weights_only=True,\
                                                            monitor='val_loss',verbose=1)
            Px_estimator.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
            history = Px_estimator.fit(data[np.random.permutation(np.arange(len(data)))], batch_size=batch_size,\
                                   epochs=epochs,verbose=1,callbacks=savebestmodel,validation_split=validation_split)
            Px_estimator.load_weights(model_path)
        
        # estimate Px
        self.Px_estimator = Px_estimator
        
    def _Px_estimate(self, inference_batch_size=200000):
        data = self.data
        Px_estimator = self.Px_estimator
        
        gauss_trans,log_Jacobian,d2p_dx2,Px = \
        inference(Px_estimator,data,batch_size=inference_batch_size)
        
        lb = 0.1
        ub = 100 - lb
        Px_lb = np.percentile(Px,lb)
        Px_ub = np.percentile(Px,ub)
        Px_lb_idx = np.where(Px < Px_lb)[0]
        Px_ub_idx = np.where(Px > Px_ub)[0]
        Px[Px_lb_idx] = Px_lb
        Px[Px_ub_idx] = Px_ub
        self.Px = Px
        np.save(self.Px_path, Px)
        
        self.Px_lb = Px_lb
        self.Px_ub = Px_ub
        
    def _Pxy_estimator(self, output_dim=256, reg=1e-4, num_coupling_layers=12, learning_rate=3e-4, epochs=200,\
                       batch_size=2048, validation_split=0.3):
        
        model_path = self.Pxy_model_path
        dim = self.dim
        
        Pxy_estimator = RealNVP(dim*2,output_dim,reg,num_coupling_layers)
        _ = Pxy_estimator(np.zeros((32,dim*2)))
        
        # load or train
        if os.path.exists(model_path):
            print('========load the trained joint probability estimator at %s========'%(model_path))
            Pxy_estimator.load_weights(model_path)
            
        else:
            # build estimator
            print('========training joint probability estimator========')
            data = self.data
            _ = Pxy_estimator(np.zeros((32,dim*2)))

            # gen joint data
            Data = Jointdata()
            joint_data = Data.gen_data(data,lag_time=self.lag_time)
            data_pxy,data_pyx = joint_data
            joint_data_training = np.vstack((data_pxy,data_pyx))
        
            
            savebestmodel = keras.callbacks.ModelCheckpoint(model_path,save_best_only=True,save_weights_only=True,\
                                                            monitor='val_loss',verbose=1)
            Pxy_estimator.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
            history = Pxy_estimator.fit(joint_data_training[np.random.permutation(np.arange(len(joint_data_training)))], batch_size=batch_size,\
                                   epochs=epochs,verbose=1,callbacks=savebestmodel,validation_split=validation_split)
            Pxy_estimator.load_weights(model_path)
            
        self.Pxy_estimator = Pxy_estimator
