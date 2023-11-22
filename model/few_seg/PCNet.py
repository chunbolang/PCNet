import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd._functions import tensor
from torch.nn import BatchNorm2d as BatchNorm        
from torch.cuda.amp import autocast as autocast
import numpy as np
from model.backbone.layer_extrator import layer_extrator
from model.util.PSPNet import OneModel as PSPNet


def con_sim(x, y):
    return 1. / (1 + 2*np.exp(-5*x)) * 1. / (1 + np.exp(-5*y))
    # This is the function that calculates the weights

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005  
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area  
    return supp_feat

class OneModel(nn.Module):
    def __init__(self, args, cls_type=None):
        super(OneModel, self).__init__()
        
        self.zoom_factor = args.zoom_factor
        self.shot = args.shot
        self.ppm_scales = args.ppm_scales

        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.pretrained = args.pretrain
        self.classes = 2
        self.fp16 = args.fp16
        self.backbone = args.backbone
        self.index = 10 
        self.pro_num = 5  
        self.data_list = [None]*self.index   
        self.mean_list = [None]*self.index   
        self.std_list =[None]*self.index
        self.mean = []
        self.std = []
        
        if self.pretrained:
            BaseNet = PSPNet(args)
            weight_path = 'initmodel/PSPNet/{}/split{}/{}/best.pth'.format(args.dataset,  args.split, args.backbone) # 'PSPNet' or 'DeepLabv3plus'
            new_param = torch.load(weight_path, map_location=torch.device('cpu'))['state_dict']
            print('load <base> weights from: {}'.format(weight_path))
            try: 
                BaseNet.load_state_dict(new_param)
            except RuntimeError:                   # 1GPU loads mGPU model
                for key in list(new_param.keys()):
                    new_param[key[7:]] = new_param.pop(key)
                BaseNet.load_state_dict(new_param)
            
            self.layer0, self.layer1, self.layer2, \
                self.layer3, self.layer4 = BaseNet.layer0, BaseNet.layer1, BaseNet.layer2, BaseNet.layer3, BaseNet.layer4

        else:
            self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = layer_extrator(backbone=args.backbone, pretrained=True)

        reduce_dim = 256
        if self.backbone == 'vgg' :
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512       

        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),  
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),                 
            nn.Conv2d(reduce_dim, self.classes, kernel_size=1)
        )                 

        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                  
        )
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                   
        )
        self.down_pro = nn.Sequential(
            nn.Conv2d(reduce_dim*6, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.re_pro = nn.Sequential(
            nn.Conv2d(reduce_dim*2, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.pyramid_bins = self.ppm_scales
        self.avgpool_list = []
        for bin in self.pyramid_bins:
            if bin > 1:
                self.avgpool_list.append(
                    nn.AdaptiveAvgPool2d(bin)
                )


        factor = 1
        mask_add_num = 1
        self.init_merge = []
        self.beta_conv = []
        self.inner_cls = []        
        for bin in self.pyramid_bins:
            self.init_merge.append(nn.Sequential(
                nn.Conv2d(reduce_dim*2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))                      
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))            
            self.inner_cls.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),                 
                nn.Conv2d(reduce_dim, self.classes, kernel_size=1)
            ))            
        self.init_merge = nn.ModuleList(self.init_merge) 
        self.beta_conv = nn.ModuleList(self.beta_conv)
        self.inner_cls = nn.ModuleList(self.inner_cls)                             

        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim*len(self.pyramid_bins), reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),                          
        )              
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),                             
        )                        

        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.alpha_conv = []
        for idx in range(len(self.pyramid_bins)-1):
            self.alpha_conv.append(nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU()
            ))     
        self.alpha_conv = nn.ModuleList(self.alpha_conv)

    def get_optim(self, model, lr_dict, LR):
        optimizer = torch.optim.SGD(
            [
            {'params': model.down_query.parameters()},
            {'params': model.down_supp.parameters()},
            {'params': model.init_merge.parameters()},
            {'params': model.alpha_conv.parameters()},
            {'params': model.beta_conv.parameters()},
            {'params': model.inner_cls.parameters()},
            {'params': model.res1.parameters()},
            {'params': model.res2.parameters()},        
            {'params': model.cls.parameters()},
            {'params': model.down_pro.parameters()},
            {'params': model.re_pro.parameters()}
            ],

            lr=LR, momentum=lr_dict['momentum'], weight_decay=lr_dict['weight_decay'])
        
        return optimizer


    def forward(self, x, s_x, s_y, y=None, cat_idx=None):
        with autocast(enabled= self.fp16):
            x_size = x.size()
            # assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
            h = x_size[2]
            w = x_size[3]

            #   Query Feature
            with torch.no_grad():
                query_feat_0 = self.layer0(x)                # [4, 128, 119, 119]
                query_feat_1 = self.layer1(query_feat_0)     # [4, 256, 119, 119]
                query_feat_2 = self.layer2(query_feat_1)     # [4, 512, 60, 60]
                query_feat_3 = self.layer3(query_feat_2)     # [4, 1024, 60, 60]
                query_feat_4 = self.layer4(query_feat_3)     # [4, 2048, 60, 60]
                if self.backbone == 'vgg' :
                    query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2),query_feat_3.size(3)), mode='bilinear', align_corners=True)

            query_feat = torch.cat([query_feat_3, query_feat_2], 1)  # [4, 1536, 60, 60]
            query_feat = self.down_query(query_feat)                 # [4, 256, 60, 60]

            #   Support Feature     
            supp_feat_list = []
            final_supp_list = []
            mask_list = []
            mask_FG_list = []
            mask_BG_list = []
            supp_feat_fg_list = []
            supp_feat_bg_list = []
            repro_list = []
            supp_out_list = []
            for i in range(self.shot):
                mask = (s_y[:,i,:,:] == 1).float().unsqueeze(1)
                mask_list.append(mask)

                with torch.no_grad():
                    supp_feat_0 = self.layer0(s_x[:,i,:,:,:])
                    supp_feat_1 = self.layer1(supp_feat_0)
                    supp_feat_2 = self.layer2(supp_feat_1)
                    supp_feat_3 = self.layer3(supp_feat_2)   
                    mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)
                    supp_feat_4 = self.layer4(supp_feat_3*mask)  
                    
                    if self.backbone == 'vgg' :
                        supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2),supp_feat_3.size(3)), mode='bilinear', align_corners=True)
                final_supp_list.append(supp_feat_4)
                supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)    # [8,768,30,30]
                supp_feat = self.down_supp(supp_feat)   # [8,256,30,30]
                supp_origin = supp_feat.clone()

                mask_FG = mask # [8,256,1,1]
                mask_BG = 1-mask_FG
                mask_FG_list.append(mask_FG)
                mask_BG_list.append(mask_BG)
                supp_feat_fg = Weighted_GAP(supp_feat, mask_FG)     # [4, 256, 1, 1]
                supp_feat_bg = Weighted_GAP(supp_feat, mask_BG)
                supp_feat_fg_list.append(supp_feat_fg)
                supp_feat_bg_list.append(supp_feat_bg)

                supp_feat_fg = F.interpolate(supp_feat_fg, size=(supp_feat.size(2), supp_feat.size(3)), mode='bilinear', align_corners=True)
                probability = nn.CosineSimilarity(1, eps=1e-7)(supp_feat, supp_feat_fg).unsqueeze(1)

                iter_now = 0    # progressive parsing: get the parsed prototype "supp_feat"
                iter_max = 4
                while iter_now <= iter_max:
                    mask_FG_now = mask_FG.clone()
                    mask_FG_now[probability>=0.7]=1 
                    mask_FG_now[probability<0.7]=0
                    mask_FG=mask_FG - mask_FG_now
                    # mask_FG=0  TP TN
                    # mask_FG<0  FP
                    # mask_FG>0  FN
                    mask_FG[mask_FG<0]=0

                    mask_BG = 1 - mask_FG

                    mask_FG_list.append(mask_FG)
                    mask_BG_list.append(mask_BG)
                    supp_feat_fg = Weighted_GAP(supp_feat, mask_FG)
                    supp_feat_bg = Weighted_GAP(supp_feat, mask_BG)
                    supp_feat_fg_list.append(supp_feat_fg)
                    supp_feat_bg_list.append(supp_feat_bg)
                    supp_feat_fg = F.interpolate(supp_feat_fg, size=(supp_feat.size(2), supp_feat.size(3)), mode='bilinear', align_corners=True)
                    probability = nn.CosineSimilarity(1, eps=1e-7)(supp_feat, supp_feat_fg).unsqueeze(1)
                    iter_now += 1
                    
                supp_feat_fg = torch.cat([supp_feat_fg_list[0], supp_feat_fg_list[1], supp_feat_fg_list[2], supp_feat_fg_list[3], supp_feat_fg_list[4],supp_feat_fg_list[5]], 1) 
                supp_feat = self.down_pro(supp_feat_fg)  # supp_feat_fg  [8,256,1,1]

           
                with torch.no_grad():
                    supp_out = nn.CosineSimilarity(1, eps=1e-7)(supp_feat, supp_origin).unsqueeze(1)
                    supp_out1 = F.interpolate(supp_out, size=(s_y.size(2), s_y.size(3)), mode='bilinear', align_corners=True)
                    supp_out2 = 1-supp_out1
                    supp_out = torch.cat([supp_out2,supp_out1],dim=1)
                    supp_out_list.append(supp_out)
                # data storage when training
                if self.training:
                    cat = cat_idx[0]  
                    for i in range(self.index):  
                        i_index = np.argwhere(cat==i) 
                        for j in range(i_index.shape[0]): 
                            new_feat = supp_feat[j,:,:,:]  
                            old_feat = self.data_list[i] 
                            if old_feat is None : 
                                new_feat = new_feat.unsqueeze(0)  
                                self.data_list[i] = new_feat
                            elif old_feat.shape[0]<self.pro_num:  
                                new_feat = new_feat.unsqueeze(0) 
                                old_feat = torch.cat([new_feat,old_feat],0) 
                                self.data_list[i] = old_feat
                            else: # update the prototype library
                                old_feat_mean = old_feat.mean(0).unsqueeze(0) 
                                cat_feat = torch.cat([new_feat.unsqueeze(0),old_feat],0) 
                                similarity = F.cosine_similarity(old_feat_mean, cat_feat,dim=1) 
                                similarity = similarity.squeeze() 
                                min_similarity_index = similarity.argmin() 

                                arr1 = cat_feat[0:min_similarity_index.item(),:,:,:] 
                                arr2 = cat_feat[min_similarity_index.item()+1:,:,:,:]
                                old_feat = torch.cat([arr1,arr2],dim=0) 
                                self.data_list[i] = old_feat

                    num = 0
                    for i in range(self.index):  
                        old_feat = self.data_list[i]  
                        if old_feat is None:  
                            num += 1
                            self.mean_list[i] = None
                            self.std_list[i] = None
                        elif old_feat.shape[0] == 1:
                            self.mean_list[i] = old_feat.mean(0).unsqueeze(0)
                            self.std_list[i] = torch.zeros([1,256,1,1]).cuda()
                        else:
                            self.mean_list[i] = old_feat.mean(0).unsqueeze(0)  
                            self.std_list[i] = old_feat.std(0).unsqueeze(0)
                    if num == 0:
                        self.mean = torch.cat(self.mean_list)
                        self.std = torch.cat(self.std_list)
                # data distillation when testing
                else:
                    repro_list = []
                    pro_list = []
                    weight = []
                    mean_sim = F.cosine_similarity(supp_feat, self.mean, dim=1)
                    mean_sim = mean_sim.squeeze()
                    max_sim_index = mean_sim.argmax()
                    chosen_mean = self.mean[max_sim_index,:,:,:]
                    chosen_std = self.std[max_sim_index,:,:,:]
                    feat = supp_feat.squeeze(0)
                    for i in range(5):
                        repro = torch.normal(feat, chosen_std).unsqueeze(0)
                        repro_list.append(repro)
                        pro = F.interpolate(repro, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True)
                        pro_list.append(pro)
                    repro = torch.cat(repro_list)
                    pro = torch.cat(pro_list)
                    vs = F.cosine_similarity(supp_feat, repro, dim=1)
                    vs = vs.squeeze()
                    vs = vs.cpu().numpy()
                    rs = F.cosine_similarity(query_feat, pro, dim=1)
                    rs = rs.mean(-1).mean(-1)
                    rs = rs.cpu().numpy()
                    sim = con_sim(vs,rs)
                    supp_feat = 0.7 * supp_feat # weights used in integration
                    
                    for j in range(5):
                        # weight.append(0.1) # if use average weights
                        weight.append(sim[j]/np.sum(sim))
                    for k in range(5):
                        supp_feat += 0.3*weight[k]*repro_list[k]
                supp_feat_list.append(supp_feat)


            corr_query_mask_list = []
            cosine_eps = 1e-7
            for i, tmp_supp_feat in enumerate(final_supp_list):  # if k-shot
                resize_size = tmp_supp_feat.size(2)
                tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

                tmp_supp_feat_4 = tmp_supp_feat * tmp_mask                    
                q = query_feat_4
                s = tmp_supp_feat_4
                bsize, ch_sz, sp_sz, _ = q.size()[:]  

                tmp_query = q
                tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)  
                tmp_query_norm = torch.norm(tmp_query, 2, 1, True)  

                tmp_supp = s         
                tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1)    
                tmp_supp = tmp_supp.contiguous().permute(0, 2, 1)    
                tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)  

                similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)  
                similarity = similarity.max(1)[0].view(bsize, sp_sz*sp_sz)   
                similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps) 
                corr_query = similarity.view(bsize, 1, sp_sz, sp_sz) 
                corr_query = F.interpolate(corr_query, size=(query_feat_3.size()[2], query_feat_3.size()[3]), mode='bilinear', align_corners=True) 
                corr_query_mask_list.append(corr_query)  
            corr_query_mask = torch.cat(corr_query_mask_list, 1).mean(1).unsqueeze(1) 
            corr_query_mask = F.interpolate(corr_query_mask, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True)  

            if self.shot > 1:              # k-shot
                supp_feat = supp_feat_list[0]
                for i in range(1, len(supp_feat_list)):
                    supp_feat += supp_feat_list[i]
                supp_feat /= len(supp_feat_list) 

            out_list = []
            pyramid_feat_list = []

            for idx, tmp_bin in enumerate(self.pyramid_bins):
                if tmp_bin <= 1.0:
                    bin = int(query_feat.shape[2] * tmp_bin)
                    query_feat_bin = nn.AdaptiveAvgPool2d(bin)(query_feat)
                else:
                    bin = tmp_bin
                    query_feat_bin = self.avgpool_list[idx](query_feat)
                supp_feat_bin = supp_feat.expand(-1, -1, bin, bin)
                corr_mask_bin = F.interpolate(corr_query_mask, size=(bin, bin), mode='bilinear', align_corners=True)  
                merge_feat_bin = torch.cat([query_feat_bin, supp_feat_bin, corr_mask_bin], 1)   
                merge_feat_bin = self.init_merge[idx](merge_feat_bin)                           

                if idx >= 1:
                    pre_feat_bin = pyramid_feat_list[idx-1].clone()
                    pre_feat_bin = F.interpolate(pre_feat_bin, size=(bin, bin), mode='bilinear', align_corners=True)  
                    rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                    merge_feat_bin = self.alpha_conv[idx-1](rec_feat_bin) + merge_feat_bin  

                merge_feat_bin = self.beta_conv[idx](merge_feat_bin) + merge_feat_bin   
                inner_out_bin = self.inner_cls[idx](merge_feat_bin)                     
                merge_feat_bin = F.interpolate(merge_feat_bin, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True)  
                pyramid_feat_list.append(merge_feat_bin)   
                out_list.append(inner_out_bin)             

            query_feat = torch.cat(pyramid_feat_list, 1)     
            query_feat = self.res1(query_feat)               
            query_feat = self.res2(query_feat) + query_feat  
            out = self.cls(query_feat)                       
            

            #   Output Part
            if self.zoom_factor != 1:
                out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

            if self.training:
                act_map = nn.Softmax(1)(out)
                alpha = self.GAP(act_map[:,1].unsqueeze(1))
                main_loss = self.criterion(out, y.long()) 
                supp_loss = 0                                      # supp_loss is the progressive parsing loss
                for i in range(self.shot):
                    supp_out = supp_out_list[i]
                    mask_use = mask_list[i].squeeze(1)
                    supp_loss += self.criterion(supp_out, mask_use.long())
                supp_loss /= len(mask_list)
                mask_y = (y==1).float().unsqueeze(1)
                alpha_1 = self.GAP(mask_y)
                beta = (alpha - alpha_1)**2 

                aux_loss = - (1-alpha)*torch.log(alpha) - beta * torch.log(1-beta)
                return out.max(1)[1], main_loss, 0.1* torch.mean(aux_loss), 0.5*supp_loss
            else:
                return out
