import os
from torch.utils.data import Dataset
import numpy as np
# from .utils import *
from utils import *
from utils import SplitComb
from utils import LossLoader
from utils import Config
from torch.utils.data import DataLoader
from loss import Calibration_loss
# from loss import SkeRecall, SkePrecision
import argparse
import torch
parser = argparse.ArgumentParser(description='Neuron 3D segmentation model')
parser.add_argument('--config', '-c', metavar='CONFIG', default='demo.yml',
                    help='configs')

class NeuronDataset(Dataset):
    def __init__(self,config):
        data_file = config.data['data_id']
        data_dir = config.data['data_dir']
        self.data_dir = data_dir
        with open (data_file,'r') as f:
            self.cases = f.readlines()
        self.cases = [f.split('\n')[0] for f in self.cases]
        self.img = [os.path.join(data_dir, f + config.data['img_subfix']) for f in self.cases]
        self.lab = [os.path.join(data_dir, f + config.data['lab_subfix']) for f in self.cases]
        self.roi_files = np.load(config.data['RoI_file'], allow_pickle=True).item()  # a dict
        self.roi_margin = config.data['ROI_margin']
        self.split_comb = SplitComb(config)

    def __getitem__(self,idx):
        img = tiff2array(self.img[idx])[np.newaxis]
        try:
            lab = tiff2array(self.lab[idx])[np.newaxis]
        except:
            lab = np.load(self.data_dir + self.cases[idx] + '_mask.npy')[np.newaxis]

        lab = lab/255.
        roi = self.roi_files[self.cases[idx]]
        img,lab = roi_data(img,lab,roi, self.roi_margin)

        crop_img, nswh, pad = self.split_comb.split(img)
        crop_lab, _, _ = self.split_comb.split(lab)
        return crop_img, crop_lab, nswh, self.cases[idx], lab, pad

    def __len__(self):
        return len(self.cases)

def run(x,model):
    b = x.size()[0]
    for i in range(b):
        x[i,:,:,:,:], _ = model.forward(x[i:(i+1),:,:,:,:])
    return x.detach()

def main(config):
    id  = 2
    # parameters
    em_names = config.net['em']
    emlist = []
    for ems in em_names:
        emlist.append(LossLoader.load(ems, config))
    model = torch.jit.load(config.net['model_file'])
    model = model.cuda(id)
    # prepare dataset
    data = NeuronDataset(config)
    data_loader = DataLoader(data, batch_size= 1,shuffle=False, num_workers=0, pin_memory=True, collate_fn=lambda x: x)

    # for inference
    em_avg = Averager()
    model.eval()
    with torch.no_grad():
        for idx, temp in enumerate(data_loader):
            data, target, zhw, name, fullab, _ = temp[0]
            data = torch.from_numpy(data).float().cuda(id)
            logit = run(data, model)
            fullab = torch.from_numpy(fullab).cuda(id)
            zhw = torch.from_numpy(zhw)
            # the prediction
            comb_pred = data_loader.dataset.split_comb.combine(logit, zhw)
            comb_pred = binary(comb_pred)
            # evaluation
            em_list = []
            if emlist is not None:
                for em_fun in emlist:
                    em_list.append(em_fun(comb_pred, fullab).detach())
                em_list = tuple([l.cpu().numpy() for l in em_list])
                em_avg.update(em_list)
                info = 'end %d out of %d, name %s, ' % (idx, len(data_loader), name)
                for lid, l in enumerate(em_list):
                    info += 'em %d: %.4f, ' % (lid, l)
                print(info)
        # print(em_avg.val())
        em_average = em_avg.val()
        print('Average F1 score:',em_average[0])
        print('Average Recall:', em_average[1])
        print('Average Precision:', em_average[2])





if __name__ == '__main__':
    global args
    args = parser.parse_args()
    config = Config.load(args.config)
    main(config)