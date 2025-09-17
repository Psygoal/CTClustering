from tqdm import tqdm
import numpy as np

def cal_pxy(realnvp_model, representative_structures, representative_P, batch=200000):
    feature_number = representative_structures.shape[1]

    num_repstru = len(representative_structures)

    pxy_mat = np.zeros((num_repstru,num_repstru))
    indices = np.triu_indices(num_repstru,k=1)

    pxy_length = len(indices[0])
    pxy = np.zeros(pxy_length)
    batch_steps = pxy_length/batch

    for batch_step in tqdm(range(int(batch_steps))):
        constructed_pxy = np.zeros((batch,feature_number*2))
        batch_id = np.arange(batch_step*batch,(batch_step+1)*batch)

        repstru_indice0 = representative_structures[indices[0][batch_id]]
        repstru_indice1 = representative_structures[indices[1][batch_id]]

        constructed_pxy = np.hstack((repstru_indice0,repstru_indice1))
        constructed_pxy_reverse = np.hstack((repstru_indice1,repstru_indice0))

        gauss_trans_,log_Jacobian_ = realnvp_model(constructed_pxy)
        pxy1 = realnvp_model.distribution.log_prob(gauss_trans_) + log_Jacobian_

        gauss_trans_,log_Jacobian_ = realnvp_model(constructed_pxy_reverse)
        pxy2 = realnvp_model.distribution.log_prob(gauss_trans_) + log_Jacobian_


        pxy[batch_step*batch:(batch_step+1)*batch] = (pxy1+pxy2)

    if batch_steps>int(batch_steps):
        constructed_pxy = np.zeros((pxy_length-(batch_step+1)*batch,feature_number*2))
        batch_id = np.arange((batch_step+1)*batch,pxy_length)

        repstru_indice0 = representative_structures[indices[0][batch_id]]
        repstru_indice1 = representative_structures[indices[1][batch_id]]

        constructed_pxy = np.hstack((repstru_indice0,repstru_indice1))
        constructed_pxy_reverse = np.hstack((repstru_indice1,repstru_indice0))

        gauss_trans_,log_Jacobian_ = realnvp_model(constructed_pxy)
        pxy1 = realnvp_model.distribution.log_prob(gauss_trans_) + log_Jacobian_

        gauss_trans_,log_Jacobian_ = realnvp_model(constructed_pxy_reverse)
        pxy2 = realnvp_model.distribution.log_prob(gauss_trans_) + log_Jacobian_
        pxy[(batch_step+1)*batch:] = (pxy1+pxy2)

    pxy_mat[indices] = pxy
    pxy_mat += pxy_mat.T
    pxy_mat -= representative_P[...,None]
    pxy_mat -= representative_P[None,...]
    diag_indice = np.diag_indices(num_repstru)
    pxy_mat[diag_indice] = 0
    
    return pxy_mat

