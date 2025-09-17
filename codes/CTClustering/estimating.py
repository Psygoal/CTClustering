from tqdm import tqdm
import numpy as np


def inference(AEflow,train_data,batch_size=100000):
    realnvp_model = AEflow.get_layer(index=-1)
    input_dims = AEflow.get_layer('real_nvp').input_shape[1]
    
    rec = np.zeros(train_data.shape[:])
    encoded_vec = np.zeros((train_data.shape[0],input_dims))
    gauss_trans = np.zeros((train_data.shape[0],input_dims))
    log_Jacobian = np.zeros(train_data.shape[0])
    d2p_dx2 = np.zeros((train_data.shape[0],input_dims))
    probability_density = np.zeros(train_data.shape[0])

    batch_step = train_data.shape[0]/batch_size
    index = np.arange(train_data.shape[0])

    for i in tqdm(range(int(batch_step))):

        batch_index = index[i*batch_size:(i+1)*batch_size]
        x_batch_train = train_data[batch_index]

        rec_,encoded_vec_,_,_ = AEflow(x_batch_train)

        gauss_trans_,log_Jacobian_ = realnvp_model(encoded_vec_)

        probability_density_ = realnvp_model.distribution.log_prob(gauss_trans_) + log_Jacobian_


        rec[batch_index] = rec_
        encoded_vec[batch_index] = encoded_vec_
        gauss_trans[batch_index] = gauss_trans_
        log_Jacobian[batch_index] = log_Jacobian_
        probability_density[batch_index] = probability_density_

    if batch_step>int(batch_step):
        batch_index = index[int(batch_step)*batch_size:]
        x_batch_train = train_data[batch_index]

        rec_,encoded_vec_,_,_ = AEflow(x_batch_train)

        gauss_trans_,log_Jacobian_ = realnvp_model(encoded_vec_)

        probability_density_ = realnvp_model.distribution.log_prob(gauss_trans_) + log_Jacobian_


        rec[batch_index] = rec_
        encoded_vec[batch_index] = encoded_vec_
        gauss_trans[batch_index] = gauss_trans_
        log_Jacobian[batch_index] = log_Jacobian_
        probability_density[batch_index] = probability_density_

    return rec,encoded_vec,gauss_trans,log_Jacobian,d2p_dx2,probability_density

def inference_realnvp(realnvp_model,train_data,batch_size=100000):
    
    input_dims = train_data.shape[1]
    gauss_trans = np.zeros((train_data.shape[0],input_dims))
    log_Jacobian = np.zeros(train_data.shape[0])
    d2p_dx2 = np.zeros((train_data.shape[0],input_dims))
    probability_density = np.zeros(train_data.shape[0])

    batch_step = train_data.shape[0]/batch_size
    index = np.arange(train_data.shape[0])

    for i in tqdm(range(int(batch_step))):

        batch_index = index[i*batch_size:(i+1)*batch_size]
        x_batch_train = train_data[batch_index]

        gauss_trans_,log_Jacobian_ = realnvp_model(x_batch_train)

        probability_density_ = realnvp_model.distribution.log_prob(gauss_trans_) + log_Jacobian_

        gauss_trans[batch_index] = gauss_trans_
        log_Jacobian[batch_index] = log_Jacobian_
        probability_density[batch_index] = probability_density_

    if batch_step>int(batch_step):
        batch_index = index[int(batch_step)*batch_size:]
        x_batch_train = train_data[batch_index]

        gauss_trans_,log_Jacobian_ = realnvp_model(x_batch_train)

        probability_density_ = realnvp_model.distribution.log_prob(gauss_trans_) + log_Jacobian_

        gauss_trans[batch_index] = gauss_trans_
        log_Jacobian[batch_index] = log_Jacobian_
        probability_density[batch_index] = probability_density_

    return gauss_trans,log_Jacobian,d2p_dx2,probability_density

