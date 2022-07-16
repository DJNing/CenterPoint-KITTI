from .detector3d_template import Detector3DTemplate
import torch
import torch.nn as nn
from ...ops.pointnet2.pointnet2_batch import domain_fusion as df
import os
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_coder_utils, box_utils, loss_utils, common_utils
import ipdb
from ...vis_tools.vis_tools import *
import numpy as np

class IASSD_GAN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset, tb_log=None):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.attach_module_topology = ['backbone_3d']
        self.shared_module_topology = ['point_head']
        self.tb_log = tb_log
        self.class_names = model_cfg.get('CLASS_NAMES', None)
        
        self.debug = self.model_cfg.get('DEBUG', False)
        self.attach_model_cfg = model_cfg.get('ATTACH_NETWORK')
        self.attach_model_cfg.BACKBONE_3D['num_class'] = num_class
        if model_cfg.get('DISABLE_ATTACH'):
            self.attach_model = None
        else:
            attach_model = self.build_attach_network()
            self.attach_model = attach_model[0]
        # self.module_list += attach_model
        if self.model_cfg.get('CROSS_OVER', None) is None:
            self.cross_over_cfg = None
            self.transfer = feat_gan(None, 1, [[-1,-1]]) # use center feature only
        else:
            self.cross_over_cfg = self.model_cfg.CROSS_OVER
            self.transfer = DomainCrossOver(self.cross_over_cfg.MLPS, channel_in=self.cross_over_cfg.CH_IN, \
                relu=True, bn=True)

        # shared_head_cfg = self.model_cfg.SHARED_HEAD
        shared_head = self.build_shared_head()
        if len(shared_head) == 0:
            self.shared_head = None
        else:
            self.shared_head = shared_head[0]

        print('building IA-SSD-GAN')

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        # for i in range(self.full_len):
        #     batch_dict = self.module_list[i](batch_dict)
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            transfer_loss = self.get_transfer_loss(batch_dict)
            disp_dict['det_loss'] = loss.item()
            loss = (transfer_loss + loss) / 2
            

            ret_dict = {
                'loss': loss,
                'gan_loss': transfer_loss
            }
            disp_dict['gan_loss'] = transfer_loss.item()
            disp_dict['tatal_loss'] = loss.item()
            return ret_dict, tb_dict, disp_dict
        else:
            
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            if self.debug:
                loss, tb_dict, disp_dict = self.get_training_loss()
                transfer_loss = self.get_transfer_loss(batch_dict)
                batch_dict['sa_ins_labels'] = tb_dict['sa_ins_labels']
                # selected lidar points
                # selected lidar points labels
                # radar points labels
                # radar points classification
                pass
            # recall_dicts['batch_dict'] = batch_dict
            return pred_dicts, recall_dicts

    def freeze_attach(self):
        pass

    # def get_shared_head_loss(self, share_head_dict):
    #     head_loss_pt, tb_dict = self.shared_head.get_loss(share_head_dict)
    #     return head_loss_pt, tb_dict
    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    def get_domain_cross_loss(self, x):
        lidar_original = x['lidar_original']
        radar_original = x['radar_original']
        lidar_recover = x['lidar_recover']
        radar_recover = x['radar_recover']
        batch_size = x['batch']['batch_size']
        lidar_center = x['att']['centers']
        _, lidar_center, _ = self.break_up_pc(lidar_center)
        radar_center = x['batch']['centers']
        _, radar_center, _ = self.break_up_pc(radar_center)
        # xyz = xyz.view(batch_size, -1, 3)
        lidar_center = lidar_center.view(batch_size, -1, 3)
        radar_center = radar_center.view(batch_size, -1, 3)

        lidar_shared_feat = x['lidar_shared'].permute(0,2,1) # [B, C, N] -> [B, N, C]
        radar_shared_feat = x['radar_shared'].permute(0,2,1)
        # lidar_xyz = x['lidar_xyz']
        # radar_xyz = x['radar_xyz']
        lidar_xyz = lidar_center
        radar_xyz = radar_center    

        rec_lidar_loss = nn.functional.mse_loss(lidar_original, lidar_recover, reduction='mean')
        rec_radar_loss = nn.functional.mse_loss(radar_original, radar_recover, reduction='mean')
        # recover loss
        rec_loss = (rec_lidar_loss + rec_radar_loss)/2
        
        # matching loss
        self_idx, _ = df.ball_point(1, radar_xyz, radar_xyz, 1)
        cross_idx, mask = df.ball_point(1, lidar_xyz, radar_xyz, 1) # this should get the one and only result
        mask = mask.unsqueeze(-1).unsqueeze(-1)
        self_feat = df.index_points_group(radar_shared_feat, self_idx)
        cross_feat = df.index_points_group(lidar_shared_feat, cross_idx)
        self_coord = df.index_points_group(radar_xyz, self_idx)
        cross_coord = df.index_points_group(lidar_xyz, cross_idx)
        self_pts = torch.cat((self_coord, self_feat), dim=-1) * mask
        cross_pts = torch.cat((cross_coord, cross_feat), dim=-1) * mask

        # draw matching to tensorboard
        gt_boxes = x['batch']['gt_boxes']

        fig = draw_match(cross_coord, self_coord, \
            cross_idx, self_idx, mask, draw_match=True, \
                bbox=gt_boxes, c_names=self.class_names)
        self.tb_log.add_figure('matching', fig)

        plt.close()
        fig = draw_match(lidar_xyz, radar_xyz, cross_idx, self_idx, mask, draw_match=False)
        self.tb_log.add_figure('raw_pointcloud', fig)
        plt.close()
        draw_match_in_one(cross_coord, self_coord, cross_idx, self_idx, mask, self.tb_log, draw_match=True)
        if torch.isnan(self_pts).sum() > 0:
            print('idx error in self_pts')
            raise RuntimeError
        elif torch.isnan(cross_pts).sum() > 0:
            print('idx error in cross_pts')
            raise RuntimeError
        matching_loss = nn.functional.mse_loss(self_pts, cross_pts, reduction='mean')        

        cross_over_loss = (rec_loss + matching_loss)/2

        # ============= save files for debug ==================
        if self.debug:
            # save point match to transfer dict
            x['radar_idx'] = self_idx
            x['lidar_idx'] = cross_idx
            x['mask'] = mask

            pass
        # ============= save files for debug ==================
        return cross_over_loss

    def positive_mask(self, self_pts, cross_pts, self_idx, cross_idx, input_dict):
        '''
        *_pts: [B, N, C]
        *_idx: [B, N, 1]
        '''
        target_cfg = self.model_cfg.TARGET_CONFIG
        gt_boxes = input_dict['gt_boxes']
        if gt_boxes.shape[-1] == 10:   #nscence
            gt_boxes = torch.cat((gt_boxes[..., 0:7], gt_boxes[..., -1:]), dim=-1)

        targets_dict_center = {}
        # assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        # assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)
        batch_size = input_dict['batch_size']      
        if target_cfg.get('EXTRA_WIDTH', False):  # multi class extension
            extend_gt = box_utils.enlarge_box3d_for_class(
                gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=target_cfg.EXTRA_WIDTH
            ).view(batch_size, -1, gt_boxes.shape[-1])
        else:
            extend_gt = gt_boxes
        extend_gt_boxes = box_utils.enlarge_box3d(
            extend_gt.view(-1, extend_gt.shape[-1]), extra_width=target_cfg.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        
        # assign box idx to points
        box_idx_self_list = []
        box_idx_cross_list = []
        for k in range(batch_size):
            self_pts_single = self_pts[k:k+1, :, :]
            cross_pts_single = cross_pts[k:k+1, :, :]
            bbox_get_pts = torch.clone(gt_boxes)
            bbox_get_pts[:,:,6] = -bbox_get_pts[:,:,6] # this works from time to time
            box_idxs_of_self_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                self_pts_single.unsqueeze(dim=0), bbox_get_pts[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
            box_idxs_of_cross_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                cross_pts_single.unsqueeze(dim=0), bbox_get_pts[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0) 
            box_idx_self_list += [box_idxs_of_self_pts]
            box_idx_cross_list += [box_idxs_of_cross_pts]
        # check if self_idx and corresponding cross_idx is in the same box

        # return bool mask for the previous step
        pass

    def get_transfer_loss(self, batch_dict):
        attach_dict = {
            'points': torch.clone(batch_dict['attach']),
            'batch_size': batch_dict['batch_size']
        }
        # torch.clone(batch_dict)
        # attach_dict['points'] = attach_dict['attach']
        
        attach_dict = self.attach_model(attach_dict)
        transfer_dict = {
            'att': attach_dict,
            'batch': batch_dict
        }
        if self.cross_over_cfg is None:
            transfer_loss = self.transfer(transfer_dict)
        else:
            # gat domain cross-over loss
            # print('calculating domain cross over loss')
            self.transfer(transfer_dict)
            cross_over_loss = self.get_domain_cross_loss(transfer_dict)
            
            # construct dict for shared_head
            radar_shared_feat = transfer_dict['radar_shared']
            share_head_dict = {}
            for key in attach_dict.keys():
                if key in batch_dict:
                    share_head_dict[key] = batch_dict[key]
            share_head_dict.pop('centers_features')
            share_head_dict['gt_boxes'] = batch_dict['gt_boxes']
            _, c, _ = radar_shared_feat.shape
            share_head_dict['centers_features'] = radar_shared_feat.permute(0,2,1).contiguous().view(-1, c)
            share_head_dict = self.shared_head(share_head_dict)
            share_head_loss, shared_tb_dict = self.shared_head.get_loss(share_head_dict)
            transfer_loss = (share_head_loss + cross_over_loss)/2
            if self.debug:
                batch_dict['radar_idx'] = transfer_dict['radar_idx']
                batch_dict['lidar_idx'] = transfer_dict['lidar_idx']
                batch_dict['mask'] = transfer_dict['mask']
                batch_dict['lidar_centers'] = transfer_dict['att']['centers']
                batch_dict['lidar_preds'] = transfer_dict['att']['sa_ins_preds']

        return transfer_loss

    def build_shared_head(self):
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': 4,
            'num_point_features': self.model_cfg.SHARED_HEAD.NUM_POINT_FEATURES,
            'grid_size': self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size,
            'is_attach': False
        }

        for module_name in self.shared_module_topology:
            self.model_cfg.SHARED_HEAD['DEBUG'] = self.debug
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict,
                custom_cfg=self.model_cfg.SHARED_HEAD
            )
            full_module_name = 'shared_' + module_name
            self.add_module(full_module_name, module)
        return model_info_dict['module_list']


    def build_attach_network(self):
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': 4,
            'num_point_features': 4,
            'grid_size': self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size,
            'is_attach': True
        }
        for module_name in self.attach_module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            full_module_name = 'attach_' + module_name
            self.add_module(full_module_name, module)
        return model_info_dict['module_list']

    def load_ckpt_to_attach(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']
        
        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])

        update_model_state = {}
        for key, val in model_state_disk.items():
            attach_key = 'attach_' + key
            if attach_key in self.state_dict() and self.state_dict()[attach_key].shape == model_state_disk[key].shape:
                update_model_state[attach_key] = val
                logger.info('Update weight %s: %s' % (key, str(val.shape)))

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        # for key in state_dict:
        #     if key not in update_model_state:
        #         logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))
        
    def freeze_attach(self, logger):
        for name, param in self.named_parameters():
            if 'attach' in name:
                param.requires_grad = False
                logger.info('Freeze param in ' + name)
            


    def get_training_loss(self):
        disp_dict = {}
        loss_point, tb_dict = self.point_head.get_loss()
        
        loss = loss_point
        return loss, tb_dict, disp_dict

class feat_gan(nn.Module):
    def __init__(self, mlps, nsample, transfer_layer, contrastive=True):
        super().__init__()
        self.nsample = nsample

        self.mlps = mlps # use to build mlp
        self.contrastive = contrastive
        self.transfer_layer = transfer_layer
        if not contrastive:
            self.mlp_module = nn.ModuleList()


        pass

    def forward(self, x):
        attach_dict = x['att']
        batch_dict = x['batch']

        att_xyzs = attach_dict['encoder_xyz']
        bat_xyzs = batch_dict['encoder_xyz']
        att_feats = attach_dict['encoder_features']
        bat_feats = batch_dict['encoder_features']
        gan_loss = []
        # get required layers
        for tl in self.transfer_layer:
            att_xyz = att_xyzs[tl[0]]
            bat_xyz = bat_xyzs[tl[1]]
            att_feat = att_feats[tl[0]].permute(0, 2, 1) # B, N, C
            bat_feat = bat_feats[tl[1]].permute(0, 2, 1)
            
            # print(att_feat.shape)
            # print(bat_feat.shape)

            # find correspondence
            bat_idx, _ = df.ball_point(1, bat_xyz, bat_xyz, 1)
            att_idx, mask = df.ball_point(1, att_xyz, bat_xyz, 1)
            # print(mask.sum())
            # test = index_points(att_xyz, att_idx)
            group_att_feat = df.index_points_group(att_feat, att_idx) # [B, N, k, C]
            group_bat_feat = df.index_points_group(bat_feat, bat_idx) 
            group_att_xyz = df.index_points_group(att_xyz, att_idx) # [B, N, k, 3]
            group_bat_xyz = df.index_points_group(bat_xyz, bat_idx) # [B, N, k, 3]
            # if torch.isnan(group_att_feat).sum() + torch.isnan(group_bat_feat).sum() > 0:
            #     ipdb.set_trace()
            # if torch.isnan(group_att_xyz).sum() + torch.isnan(group_bat_xyz).sum() > 0:
            #     ipdb.set_trace()
            # group_att_xyz = df.index_points_group()
            
            group_att_points = torch.cat((group_att_xyz, group_att_feat), dim=-1) # [B, N, k, 3+C]
            group_bat_points = torch.cat((group_bat_xyz, group_bat_feat), dim=-1) # [B, N, k, 3+C]
            
            B, N = mask.shape
            mask = mask.reshape([B, N, 1, 1])
            _, _, _, C = group_att_points.shape
            mask = mask.repeat([1, 1, 1, C])
            group_att_points = group_att_points * mask
            group_bat_points = group_bat_points * mask
            # ipdb.set_trace()
            # print(nn.functional.mse_loss(group_att_feat, group_bat_feat, reduction='mean'))
            # break
            if self.contrastive:
                gan_loss += [nn.functional.mse_loss(group_att_points, group_bat_points, reduction='mean')]
            else:
                raise NotImplementedError
        
        loss = sum(gan_loss) / len(gan_loss)
        if torch.isnan(loss):
            loss = gan_loss[-1]

        # loss = loss / len(gan_loss)
        # print(loss)
        return loss

    def negative_log_contrastive(self, pts0, pts1, mask):
        xyz0 = pts0[:, :, :, :3]
        xyz1 = pts1[:, :, :, :3]
        feat0 = pts0[:, :, :, 3:]
        feat1 = pts1[:, :, :, 3:]

        pass

    def triplet_loss(self, pts0, pts1, mask):
        
        feat0 = pts0[:, :, :, 3:]
        feat1 = pts1[:, :, :, 3:]
        # gather positive 

        # gather negative
        
        pass
                    
    def weighted_mse(self, pts0, pts1):
        xyz0 = pts0[:, :, :, :3]
        xyz1 = pts1[:, :, :, :3]
        feat0 = pts0[:, :, :, 3:]
        feat1 = pts1[:, :, :, 3:]
        xyz_mse = nn.functional.mse_loss(xyz0, xyz1, reduction='mean')
        feat_mse = nn.functional.mse_loss(feat0, feat1, reduction='mean')
        loss = xyz_mse + 5 * feat_mse
        return loss
    

class DomainCrossOver(nn.Module):
    def __init__(self, mlps, channel_in, relu=True, bn=True):
        super().__init__()
        self.mlps = mlps
        self.channel_in = channel_in
        self.lidar_shared_mlp = CrossOverBlock(mlps, channel_in, relu=relu, bn=bn)
        self.radar_shared_mlp = CrossOverBlock(mlps, channel_in, relu=relu, bn=bn)
        self.lidar_unique_mlp = CrossOverBlock(mlps, channel_in, relu=relu, bn=bn)
        self.radar_unique_mlp = CrossOverBlock(mlps, channel_in, relu=relu, bn=bn)
        recover_mlp, recover_ch_in = self.build_recover_mlps()
        recover_mlp[0] = mlps[-1] * 2 
        self.radar_recover_mlp = CrossOverBlock(recover_mlp, recover_ch_in)
        self.lidar_recover_mlp = CrossOverBlock(recover_mlp, recover_ch_in)
        
    def forward(self, x):
        '''
        feat_dict:lidar_feat & radar_feat
        '''
        attach_dict = x['att']
        batch_dict = x['batch']
        att_feats = attach_dict['encoder_features']
        bat_feats = batch_dict['encoder_features']
        lidar_feat = att_feats[-1]
        radar_feat = bat_feats[-1]
        att_xyzs = attach_dict['encoder_xyz']
        bat_xyzs = batch_dict['encoder_xyz']
        # att_xyzs = attach_dict['centers']
        # bat_xyzs = batch_dict['centers']
        
        lidar_xyz = att_xyzs[-1]
        radar_xyz = bat_xyzs[-1]
        if torch.isnan(radar_xyz).sum() > 0:
            raise RuntimeError('Nan occurs in domain cross over!')
        
        # lidar_xyz = batch_dict['lidar_xyz'] # psuedo
        # radar_xyz = batch_dict['radar_xyz'] # psuedo

        shared_lidar = self.lidar_shared_mlp(lidar_feat) # [B, C, N]
        unique_lidar = self.lidar_unique_mlp(lidar_feat)
        shared_radar = self.radar_shared_mlp(radar_feat)
        unique_radar = self.radar_unique_mlp(radar_feat)

        lidar_recover = self.lidar_recover_mlp(torch.cat((shared_lidar, unique_lidar), dim=1))
        radar_recover = self.radar_recover_mlp(torch.cat((shared_radar, unique_radar), dim=1))

        x['lidar_original'] = lidar_feat
        x['radar_original'] = radar_feat
        x['lidar_recover'] = lidar_recover
        x['radar_recover'] = radar_recover
        x['radar_shared'] = shared_radar
        x['lidar_shared'] = shared_lidar
        x['lidar_xyz'] = lidar_xyz
        x['radar_xyz'] = radar_xyz
        
        return

    def get_cross_loss(self, x):
        lidar_shared_feat = x['lidar_shared']
        radar_shared_feat = x['radar_shared']
        lidar_xyz = x['lidar_xyz']
        radar_xyz = x['radar_xyz']
        # get 1nn 
        pass

    def build_recover_mlps(self):
        recover_mlps = self.mlps[::-1]
        recover_ch_in = 2 * self.mlps[-1] # we cat two vectors to recover the original one
        for i, v in enumerate(recover_mlps):
            if v < recover_ch_in:
                recover_mlps[i] = recover_ch_in

        return recover_mlps, recover_ch_in


class CrossOverBlock(nn.Module):
    def __init__(self, mlps, channel_in, relu=True, bn=True):
        '''
        mlps: list of output channel
        channel_in: input channel
        relu: whether to use relu
        bn: whether to use bn
        '''
        super().__init__()
        self.last_channel = mlps[-1]
        self.relu = relu
        self.bn = bn
        in_chs = [channel_in]
        self.mlp = []
        for idx, ch_out in enumerate(mlps):
            self.mlp.append(
                mlp_bn_relu(
                    in_chs[idx],
                    ch_out,
                    relu=self.relu,
                    bn=self.bn
                )
            )
            in_chs += [ch_out]
        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x):
        return self.mlp(x)
        

class mlp_bn_relu(nn.Module):
    def __init__(self, ch_in, ch_out, relu=True, bn=True):
        super().__init__()
        
        temp_list = [nn.Conv1d(ch_in, ch_out, 1, 1)]
        
        if bn:
            temp_list += [nn.BatchNorm1d(ch_out)]

        if relu:
            temp_list += [nn.ReLU()]

        self.net = nn.Sequential(*temp_list)

    def forward(self, x):
        return self.net(x)

def draw_match(lidar, radar, lidar_idx, radar_idx, mask, draw_match=True, \
    bbox=None, c_names=None):
    # draw the first batch
    lidar_pts = lidar[0, :, :].cpu().numpy().reshape([-1, 3])
    radar_pts = radar[0, :, :].cpu().numpy().reshape([-1, 3])
    mask_match = mask[0, :, :].cpu().numpy().reshape([-1])
    match_idx = np.where(mask_match == 1)[0]
    # l_idx = lidar_idx[0, :, :].cpu().numpy()[match_idx, :].reshape([-1])
    # r_idx = radar_idx[0, :, :].cpu().numpy()[match_idx, :].reshape([-1])
    # x, y only
    # l_pts = lidar_pts[l_idx, :2]
    # r_pts = radar_pts[r_idx, :2]
    l_pts = lidar_pts[match_idx, :2]
    r_pts = radar_pts[match_idx, :2]
    fig, ax1, ax2 = draw_two_pointcloud(lidar_pts, radar_pts, 'lidar', 'radar')
    if draw_match:
        for i in range(mask_match.sum()):
            xyA = l_pts[i, :]
            xyB = r_pts[i, :]
            draw_cross_line(xyA, xyB, fig, ax1, ax2)

    if bbox is not None:
        raw_bbox = bbox[0, :, :].cpu().numpy()
        rec_list = boxes2rec(raw_bbox, c_names)
        for rec in rec_list:
            ax1.add_patch(rec)
        rec_list = boxes2rec(raw_bbox, c_names)
        for rec in rec_list:
            ax2.add_patch(rec)
    return fig

def draw_match_in_one(lidar, radar, lidar_idx, radar_idx, mask, tb_log, draw_match=True):
    # draw the first batch
    lidar_pts = lidar[0, :, :].cpu().numpy().reshape([-1, 3])
    radar_pts = radar[0, :, :].cpu().numpy().reshape([-1, 3])
    mask_match = mask[0, :, :].cpu().numpy().reshape([-1])
    match_idx = np.where(mask_match == 1)[0]
    # l_idx = lidar_idx[0, :, :].cpu().numpy()[match_idx, :].reshape([-1])
    # r_idx = radar_idx[0, :, :].cpu().numpy()[match_idx, :].reshape([-1])
    # # x, y only
    # l_pts = lidar_pts[l_idx, :2]
    # r_pts = radar_pts[r_idx, :2]
    l_pts = lidar_pts[match_idx, :2]
    r_pts = radar_pts[match_idx, :2]
    fig = plt.figure(dpi=150)
    ax = fig.add_subplot()

    ax.scatter(l_pts[:,0], l_pts[:,1], c='cyan', s=10)
    ax.scatter(r_pts[:,0], r_pts[:,1], c='gold', s=10)
    drawBEV(ax, lidar_pts, radar_pts, None, None, 'match')
    if draw_match:
        for i in range(mask_match.sum()):
            xyA = l_pts[i, :]
            xyB = r_pts[i, :]
            # draw_cross_line(xyA, xyB, fig, ax, ax)
            x_values = (xyA[0], xyB[0])
            y_values = (xyA[1], xyB[1])
            ax.plot(x_values, y_values, '-', color='orange')
    # if tb_log is not None:
    tb_log.add_figure('one_fig_match', fig)
    plt.close()
    fig = plt.figure(dpi=150)
    ax = fig.add_subplot()
    drawBEV(ax, l_pts, r_pts, None, None, 'matched_pts')
    tb_log.add_figure('matched_pts', fig)
    plt.close()
    
