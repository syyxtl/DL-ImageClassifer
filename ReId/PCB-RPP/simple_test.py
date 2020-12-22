import os, time
import torch
import numpy as np
from utils import load_data_ot
from model import PCB, PCB_test
from torch.autograd import Variable

RPP = False
feature_H = True
######################################################################
#************************* GPU concern  *****************************#
str_ids = "0".split(',')
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
use_gpu = torch.cuda.is_available()
#************************* GPU concern  *****************************#
######################################################################

######################################################################
#************************* load model *******************************#
def load_network(network):
    save_path = os.path.join("./Models/", "PCB", 'net_%s.pth'% "last")
    network.load_state_dict(torch.load(save_path), False)
    return network
#************************* loaded model *****************************#
######################################################################

######################################################################
#************ extract feature from a trained model ******************#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip

def extract_feature(model, dataloaders):
    features = torch.FloatTensor()

    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size() #batch_size, chaanel, height, weight
        if feature_H:
            ff = torch.FloatTensor(n, 256, 6).zero_() # we have six parts
        else:
            ff = torch.FloatTensor(n, 2048, 6).zero_()  # we have six parts

        for i in range(2):
            if(i == 1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            outputs = model(input_img)
            f = outputs.data.cpu()
            ff = ff+f

        # feature size (n,2048,6)
        # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
        # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
            # fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        ff = ff.view(ff.size(0), -1)
        features = torch.cat((features, ff), 0)
    return features
#************ extract feature from a trained model ******************#
######################################################################

def simple_test_main():
    print('-------test-----------')
    datas, images = load_data_ot()

    model_structure = PCB(751)
    if RPP:
        model_structure = model_structure.convert_to_rpp()

    model = load_network(model_structure)
    model = PCB_test(model, feature_H)

    model = model.eval()
    if use_gpu:
        model = model.cuda()

    process_start_time = time.time()

    query_feature = extract_feature(model, datas['two'])
    gallery_feature = extract_feature(model, datas['one'])
    
    one, two = gallery_feature, query_feature[0]
    for o in one:
        score = np.dot(o, two)
        # print("SCORE:", score)

    process_stop_time = time.time()

    diff_time = process_stop_time - process_start_time
    print(diff_time)
#************************** main test  ******************************#
######################################################################

simple_test_main()

# no RPP: 33 1.67-3.75
# RPP: 33 1.81-4.69