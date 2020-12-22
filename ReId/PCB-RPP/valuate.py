import scipy.io
import torch
import numpy as np
import time
import os
import  argparse

parser = argparse.ArgumentParser(description='Evaluating')
parser.add_argument('--result_mat', default='./Results/RPP_H_result.mat', type=str, help='save result dir')
args = parser.parse_args()

RES = []
######################################################################
#*************************** evaluate *******************************#
def evaluate(qf,ql,qc,gf,gl,gc):
    query = qf
    score = np.dot(gf,query)
    RES.append(max(score))
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # 分出正样本和负样本集：不同camera下的相同的人为正样本集，相同camera下相同的人与-1 label的为负样本集
    # setdiff1d: 差异
    # intersect1d: 并集
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)
    
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) 
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp
#*************************** evaluated ******************************#
######################################################################

######################################################################
#************************ compute index *****************************#
def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    cmc[rows_good[0]:] = 1

    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc
#************************ computed index ****************************#
######################################################################

######################################################################
#************************* main eval ********************************#
def eval_main():
    result = scipy.io.loadmat(args.result_mat)
    query_feature = result['query_feature']
    query_cam = result['query_cam'][0]
    query_label = result['query_label'][0]
    gallery_feature = result['gallery_feature']
    gallery_cam = result['gallery_cam'][0]
    gallery_label = result['gallery_label'][0]

    multi = os.path.isfile('multi_query.mat')
   
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    indexS = 100
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        # print(i, CMC_tmp[0])
        indexS = indexS + 1
        if indexS == 100:
            break

    CMC = CMC.float()
    CMC = CMC/len(query_label) #average CMC
    print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/len(query_label)))
#************************** main eval  ******************************#
######################################################################

eval_main()