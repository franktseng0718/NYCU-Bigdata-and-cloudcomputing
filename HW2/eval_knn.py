import os
import sys
import argparse
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
import torch.nn.functional as F
import utils
import vision_transformer as vits
#param
data_path = 'hw2/test'
batch_size_per_gpu = 16
num_workers = 8
arch = 'vit_small'
patch_size = 8
pretrained_weights = 'checkpoints/checkpoint0720.pth' 
embed_dim = 192
out_dim = 512
use_bn_in_head = False
norm_last_layer = False

# init
generate_embedding = True

@torch.no_grad()
def KNN(emb, cls, batch_size, Ks=[1, 10, 50, 100]):
    """Apply KNN for different K and return the maximum acc"""
    preds = []
    mask = torch.eye(batch_size).bool().to(emb.device)
    mask = F.pad(mask, (0, len(emb) - batch_size))
    for batch_x in torch.split(emb, batch_size):
        dist = torch.norm(
            batch_x.unsqueeze(1) - emb.unsqueeze(0), dim=2, p="fro")
        now_batch_size = len(batch_x)
        mask = mask[:now_batch_size]
        dist = torch.masked_fill(dist, mask, float('inf'))
        # update mask
        mask = F.pad(mask[:, :-now_batch_size], (now_batch_size, 0))
        pred = []
        for K in Ks:
            knn = dist.topk(K, dim=1, largest=False).indices
            knn = cls[knn].cpu()
            pred.append(torch.mode(knn).values)
        pred = torch.stack(pred, dim=0)
        preds.append(pred)
    preds = torch.cat(preds, dim=1)
    accs = [(pred == cls.cpu()).float().mean().item() for pred in preds]
    return max(accs)

@torch.no_grad()
def extract_feature(data_path='hw2/test'):
    transform = pth_transforms.Compose([
        pth_transforms.Resize(96, interpolation=3),
        #pth_transforms.CenterCrop(96),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_eval = ReturnIndexDataset(data_path, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset_eval,
        #sampler=sampler,
        batch_size=batch_size_per_gpu,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    '''
    model = utils.MultiCropWrapper(vits.__dict__[arch](patch_size=patch_size, num_classes=0), vits.DINOHead(
        embed_dim,
        out_dim,
        use_bn=use_bn_in_head,
        norm_last_layer=norm_last_layer,
    ))
    '''
    model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
    print(f"Model {arch} {patch_size}x{patch_size} built.")
    metric_logger = utils.MetricLogger(delimiter="  ")
    model.cuda()

    #load pretrained weight
    state_dict = torch.load(pretrained_weights, map_location="cpu")
    state_dict = state_dict['teacher']
    print(k for k in state_dict.keys())
   
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    
    msg = model.load_state_dict(state_dict, strict=False)
    print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    model.eval()
    emd_dim = model.embed_dim
    #extract feature
    features = torch.empty(0)
    for samples, index in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda()
        index = index.cuda()
        feats = model(samples).cpu()
        feats = torch.cat((feats, torch.zeros((feats.size(0), out_dim - emd_dim))), dim=1)
        #print(features.shape)
        features = torch.cat((features, feats), dim=0)
    features = features.to(dtype=torch.float32).numpy()
    dataset_eval.class_to_idx
    #print(dataset_eval.imgs)
    labels = torch.tensor([s[1]  for s in dataset_eval.imgs]).numpy()
    #save features
    if data_path == 'hw2/test':
        np.save('features_test', features)
        np.save('labels_test', labels)
    else:
        np.save('features_train', features)
    return features, labels
     
class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx

if __name__ == '__main__':
    cudnn.benchmark = True

    # generate test_data embedding
    if os.path.isfile('features_test.npy'):
        features = np.load('features_test.npy')
        labels = np.load('labels_test.npy')
    else:
        features, labels = extract_feature()

    #generate train_data embedding
    if os.path.isfile('features_train.npy'):
        features_train = np.load('features_train.npy')
    elif not generate_embedding:
        pass
    else:
        features_train, labels_train = extract_feature(data_path = 'hw2/unlabeled')
    
    print(features_train.shape)
    #print(features_train.dtype)
    print(features)
    print(features.shape)
    print(features.dtype)
    print(labels.shape)
    features = torch.from_numpy(features)
    labels = torch.from_numpy(labels)
    acc = KNN(features, labels, batch_size=16)
    print("Accuracy: %.5f" % acc)
