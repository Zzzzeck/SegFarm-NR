import torch
import torch.nn as nn
import sys

sys.path.append("..")

from prompts.imagenet_template import *

from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData
from mmseg.registry import MODELS

import torch.nn.functional as F

from open_clip import tokenizer, create_model
from BLIP.models.blip_retrieval import blip_retrieval
import gem
from simfeatup_dev.upsamplers import get_upsampler
from simfeatup_dev.enhanced_upsamplers import EnhancedJBUOne


@MODELS.register_module()
class SegFarmSegmentation(BaseSegmentor):
    def __init__(self,
                 clip_type,
                 vit_type,
                 model_type,
                 name_path,
                 device=torch.device('cuda'),
                 ignore_residual=True,
                 prob_thd=0.1,
                 logit_scale=50,
                 slide_stride=112,
                 slide_crop=224,
                 cls_token_lambda=0,
                 mlp_corrector=False,
                 frozen_mlp=False,
                 bg_idx=0,
                 feature_up=True,
                 feature_up_cfg=dict(
                     model_name='jbu_one',
                     model_path='your/model/path'),
                 use_nir=False):
        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            bgr_to_rgb=True)
        super().__init__(data_preprocessor=data_preprocessor)
        
        # Save model name
        self.model_name = 'EnhancedJBUOne'
        
        if clip_type == 'CLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B/16', pretrained='openai', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L-14', pretrained='openai', precision='fp16')
        elif clip_type == 'RemoteCLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B/32', pretrained='checkpoint/RemoteCLIP-ViT-B-32.pt', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L-14', pretrained='checkpoint/RemoteCLIP-ViT-L-14.pt', precision='fp16')
        elif clip_type == 'GeoRSCLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B/32', pretrained='checkpoint/RS5M_ViT-B-32.pt', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L-14', pretrained='checkpoint/RS5M_ViT-L-14.pt', precision='fp16')
            elif 'H' in vit_type:
                self.net = create_model('ViT-H-14', pretrained='checkpoint/RS5M_ViT-H-14.pt', precision='fp16')
        elif clip_type == 'SkyCLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B/32', \
                                        pretrained='checkpoint/SkyCLIP_ViT_B32_top50pct/epoch_20.pt', \
                                        precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L-14', \
                                        pretrained='checkpoint/SkyCLIP_ViT_L14_top30pct_filtered_by_CLIP_laion_RS/epoch_20.pt', \
                                        precision='fp16')
        elif clip_type == 'OpenCLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B/16', pretrained='laion2b_s34b_b88k', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L-14', pretrained='laion2b_s32b_b82k', precision='fp16')
        elif clip_type == 'MetaCLIP':
            if 'B' in vit_type:
                self.net = create_model('ViT-B-16-quickgelu', pretrained='metaclip_fullcc', precision='fp16')
            elif 'L' in vit_type:
                self.net = create_model('ViT-L/14-quickgelu', pretrained='metaclip_fullcc', precision='fp16')
        elif clip_type == 'BLIP':
            if 'B' in vit_type:
                self.net = blip_retrieval(pretrained='checkpoint/model_base_14M.pth', image_size=slide_crop, vit='base')
            elif 'L' in vit_type:
                self.net = blip_retrieval(pretrained='checkpoint/model_large.pth', image_size=slide_crop, vit='large')
            self.net = self.net.half()
        elif clip_type == 'ALIP':
            self.net = create_model('ViT-B/32', pretrained='checkpoint/ALIP_YFCC15M_B32.pt', precision='fp16')

        if model_type == 'GEM':
            if 'B' in vit_type:
                if clip_type == 'CLIP':
                    self.net = gem.create_gem_model('ViT-B/16', 'openai', ignore_residual=ignore_residual, device=device, precision='fp16')
                elif clip_type == 'OpenCLIP':
                    self.net = gem.create_gem_model('ViT-B/16', 'laion2b_s34b_b88k', ignore_residual=ignore_residual, device=device, precision='fp16')
                elif clip_type == 'MetaCLIP':
                    self.net = gem.create_gem_model('ViT-B/16-quickgelu', 'metaclip_fullcc', ignore_residual=ignore_residual, device=device, precision='fp16')
            elif 'L' in vit_type:
                if clip_type == 'CLIP':
                    self.net = gem.create_gem_model('ViT-L-14', 'openai', ignore_residual=ignore_residual, device=device, precision='fp16')
                elif clip_type == 'OpenCLIP':
                    self.net = gem.create_gem_model('ViT-L-14', 'laion2b_s32b_b82k', ignore_residual=ignore_residual, device=device, precision='fp16')
                elif clip_type == 'MetaCLIP':
                    self.net = gem.create_gem_model('ViT-L-14-quickgelu', 'metaclip_fullcc', ignore_residual=ignore_residual, device=device, precision='fp16')
            self.net = self.net.model

        self.net.eval().to(device)
        self.tokenizer = tokenizer.tokenize

        self.clip_type = clip_type
        self.vit_type = vit_type
        self.model_type = model_type
        self.feature_up = feature_up
        self.cls_token_lambda = cls_token_lambda
        self.output_cls_token = cls_token_lambda != 0 or mlp_corrector
        self.mlp_corrector = mlp_corrector
        self.frozen_mlp = frozen_mlp
        self.bg_idx = bg_idx
        self.use_nir = use_nir

        if self.clip_type == 'BLIP':
            self.patch_size = self.net.visual_encoder.patch_size
        else:
            self.patch_size = self.net.visual.patch_size

        query_words, self.query_idx = get_cls_idx(name_path)
        self.num_queries = len(query_words)
        self.num_classes = max(self.query_idx) + 1
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        query_features = []
        with torch.no_grad(): # sub_imagenet_template, openai_imagenet_template
            for qw in query_words:
                if self.clip_type == 'BLIP':
                    query =self.net.tokenizer([temp(qw) for temp in openai_imagenet_template], padding='max_length',
                                           truncation=True, max_length=35,
                                           return_tensors="pt").to(device)
                    text_output = self.net.text_encoder(query.input_ids, attention_mask=query.attention_mask,
                                                        mode='text')
                    feature = F.normalize(self.net.text_proj(text_output.last_hidden_state[:, 0, :]))
                else:
                    query = self.tokenizer([temp(qw) for temp in openai_imagenet_template]).to(device)
                    feature = self.net.encode_text(query)
                    feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
        self.query_features = torch.cat(query_features, dim=0)

        self.dtype = self.query_features.dtype
        self.ignore_residual = ignore_residual
        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.feature_up_cfg = feature_up_cfg

        if feature_up:
            self.feat_dim = self.query_features.shape[-1]
            
            try:
                # Use Enhanced JBUOne model
                print(f"Using Enhanced JBUOne model")
                self.upsampler = EnhancedJBUOne(self.feat_dim).cuda().half()
                    
                # Load weights
                ckpt = torch.load(feature_up_cfg['model_path'])['state_dict']
                
                # Get current model keys
                model_keys = set(self.upsampler.state_dict().keys())
                print(f"Number of keys required by model: {len(model_keys)}")
                
                # Get keys from weight file
                ckpt_keys = set()
                for k in ckpt.keys():
                    # Remove model prefix
                    if k.startswith('model.'):
                        key = k[6:]
                    else:
                        key = k[10:] if k.startswith('upsampler.') else k
                    ckpt_keys.add(key)
                
                print(f"Number of keys in weight file: {len(ckpt_keys)}")
                
                # Check key matching
                missing_in_model = ckpt_keys - model_keys
                missing_in_ckpt = model_keys - ckpt_keys
                
                if missing_in_model:
                    print(f"Warning: {len(missing_in_model)} keys from weight file do not exist in model")
                
                if missing_in_ckpt:
                    print(f"Warning: {len(missing_in_ckpt)} keys from model do not exist in weight file")
                
                # Prepare weights dictionary to load
                weights_dict = {}
                for k, v in ckpt.items():
                    # Remove model prefix
                    if k.startswith('model.'):
                        key = k[6:]
                    else:
                        key = k[10:] if k.startswith('upsampler.') else k
                    
                    # Only keep keys that exist in model
                    if key in model_keys:
                        weights_dict[key] = v
                
                print(f"Number of filtered weight keys: {len(weights_dict)}")
                
                # Load weights in non-strict mode
                self.upsampler.load_state_dict(weights_dict, strict=False)
                print(f"Successfully loaded upsampler weights")
                
            except Exception as e:
                print(f"Error during upsampler initialization or weight loading: {e}")
                print(f"Using default upsampler without loading weights")
                self.upsampler = EnhancedJBUOne(self.feat_dim).cuda().half()
                # Use default initialization without loading weights

        # If using MLP corrector, create MLP network
        if self.mlp_corrector:
            # MLP input dimension is number of query features (num classes), output dimension is also number of query features
            mlp_hidden_dim = 256  # MLP hidden layer dimension
            self.cls_mlp = nn.Sequential(
                nn.Linear(self.num_queries, mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(mlp_hidden_dim, self.num_queries)
            ).to(device).to(self.dtype)
            print("MLP corrector enabled with learnable global-local debiasing weights")
            
            # Freeze MLP weights if needed
            if frozen_mlp:
                self.freeze_mlp_corrector(True)
                print("MLP corrector weights frozen and will not be updated during training")

    def forward_feature(self, img, logit_size=None):
        if type(img) == list:
            img = img[0]
            
        # Check if input contains NIR channel
        has_nir_channel = (img.shape[1] == 4)
        
        # For NIR+RGB input (4 channels)
        if has_nir_channel and self.use_nir:
            # Separate RGB and NIR channels
            rgb_img = img[:, :3, :, :]
            nir_img = img[:, 3:4, :, :]
            
            # Extract CLIP features only from RGB channels
            if self.clip_type == 'BLIP':
                rgb_img = F.interpolate(rgb_img, size=(self.slide_crop, self.slide_crop), mode='bilinear', align_corners=False)
                image_features = self.net.visual_encoder(rgb_img, self.ignore_residual)
                image_features = self.net.vision_proj(image_features[:, 1:, ])
            elif self.model_type == 'GEM':
                image_features = self.net.visual(rgb_img)
            else:
                is_farm_earth_model = self.model_type in ['SegEarth', 'SegFarm']
                image_features = self.net.encode_image(rgb_img, 
                                                      'SegEarth' if is_farm_earth_model else self.model_type, 
                                                      self.ignore_residual, 
                                                      self.output_cls_token)
        else:
            # Original processing approach - use only RGB or RGB merged to 3 channels
            if self.clip_type == 'BLIP':
                img = F.interpolate(img, size=(self.slide_crop, self.slide_crop), mode='bilinear', align_corners=False)
                image_features = self.net.visual_encoder(img, self.ignore_residual)
                image_features = self.net.vision_proj(image_features[:, 1:, ])
            elif self.model_type == 'GEM':
                image_features = self.net.visual(img)
            else:
                is_farm_earth_model = self.model_type in ['SegEarth', 'SegFarm']
                image_features = self.net.encode_image(img, 
                                                      'SegEarth' if is_farm_earth_model else self.model_type,
                                                      self.ignore_residual, 
                                                      self.output_cls_token)
            
        if self.output_cls_token:
            image_cls_token, image_features = image_features
            image_cls_token /= image_cls_token.norm(dim=-1, keepdim=True)
            cls_logits = image_cls_token @ self.query_features.T

        # Feature upsampling - use improved upsampling module according to farmland application requirements
        if self.feature_up:
            feature_w, feature_h = img[0].shape[-2] // self.patch_size[0], img[0].shape[-1] // self.patch_size[1]
            image_w, image_h = img[0].shape[-2], img[0].shape[-1]
            image_features = image_features.permute(0, 2, 1).view(1, self.feat_dim, feature_w, feature_h)
            
            with torch.cuda.amp.autocast():
                # If using multispectral agricultural upsampler with NIR channel
                if has_nir_channel and self.use_nir and 'agricrop' in self.feature_up_cfg['model_name']:
                    # Input full 4-channel image to upsampler
                    image_features = self.upsampler(image_features, img).half()
                else:
                    # Original upsampling approach
                    image_features = self.upsampler(image_features, img[:, :3] if has_nir_channel else img).half()
                    
            image_features = image_features.view(1, self.feat_dim, image_w * image_h).permute(0, 2, 1)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ self.query_features.T

        if self.output_cls_token:
            if self.mlp_corrector:
                # Use MLP corrector: O = Opatch - MLP(O[CLS])
                batch_size = logits.shape[0]
                # cls_logits shape: [batch_size, num_queries]
                # MLP output shape should also be: [batch_size, num_queries]
                mlp_output = self.cls_mlp(cls_logits)
                # Apply MLP output as correction term, each sample has its own correction value for each query feature
                logits = logits - mlp_output.unsqueeze(1).expand_as(logits)
            else:
                # Original method: O = Opatch + λO[CLS] (note addition is used because cls_token_lambda is typically negative)
                logits = logits + cls_logits * self.cls_token_lambda
            
        # Restore original behavior, ensuring correct tensor shape is returned
        if self.feature_up:
            w, h = img[0].shape[-2], img[0].shape[-1]
        else:
            w, h = img[0].shape[-2] // self.patch_size[0], img[0].shape[-1] // self.patch_size[1]
        
        out_dim = logits.shape[-1]
        logits = logits.permute(0, 2, 1).reshape(-1, out_dim, h, w)
        
        if logit_size is None:
            logits = nn.functional.interpolate(logits, size=img.shape[-2:], mode='bilinear')
        else:
            logits = nn.functional.interpolate(logits, size=logit_size, mode='bilinear')
        
        return logits

    def forward_slide(self, img, img_metas, stride=112, crop_size=224):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = self.num_queries
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]

                # pad image when (image_size % patch_size != 0)
                H, W = crop_img.shape[2:]
                pad = self.compute_padsize(H, W, self.patch_size[0])

                if any(pad):
                    crop_img = nn.functional.pad(crop_img, pad)

                crop_seg_logit = self.forward_feature(crop_img)

                # mask cutting for padded image
                if any(pad):
                    l, t = pad[0], pad[2]
                    crop_seg_logit = crop_seg_logit[:, :, t:t + H, l:l + W]

                preds += nn.functional.pad(crop_seg_logit,
                                           (int(x1), int(preds.shape[3] - x2), int(y1),
                                            int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat
        img_size = img_metas[0]['ori_shape'][:2]
        logits = nn.functional.interpolate(preds, size=img_size, mode='bilinear')

        return logits

    @torch.no_grad()
    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                                  dict(
                                      ori_shape=inputs.shape[2:],
                                      img_shape=inputs.shape[2:],
                                      pad_shape=inputs.shape[2:],
                                      padding_size=[0, 0, 0, 0])
                              ] * inputs.shape[0]
        inputs = inputs.half()
        if self.slide_crop > 0:
            seg_logits = self.forward_slide(inputs, batch_img_metas, self.slide_stride, self.slide_crop)
        else:
            seg_logits = self.forward_feature(inputs, batch_img_metas[0]['ori_shape'])

        return self.postprocess_result(seg_logits, data_samples)

    def postprocess_result(self, seg_logits, data_samples):
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            seg_logits = seg_logits[i] * self.logit_scale
            seg_logits = seg_logits.softmax(0)  # n_queries * w * h

            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]

            seg_pred = seg_logits.argmax(0, keepdim=True)
            seg_pred[seg_logits.max(0, keepdim=True)[0] < self.prob_thd] = self.bg_idx

            if data_samples is None:
                return seg_pred
            else:
                data_samples[i].set_data({
                    'seg_logits':
                        PixelData(**{'data': seg_logits}),
                    'pred_sem_seg':
                        PixelData(**{'data': seg_pred})
                })
        return data_samples

    def compute_padsize(self, H: int, W: int, patch_size: int):
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l

        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t

        return l, r, t, b

    def _forward(data_samples):
        """
        """

    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """

    def extract_feat(self, inputs):
        """
        """

    def loss(self, inputs, data_samples):
        """Compute loss function, supports MLP corrector training"""
        # Pass input to forward_feature to get prediction results
        seg_logits = self.forward_feature(inputs)
        
        # Get ground truth
        gt_semantic_segs = [data_sample.gt_sem_seg.data for data_sample in data_samples]
        gt_semantic_segs = torch.stack(gt_semantic_segs, dim=0)
        
        # Ensure prediction results and ground truth have consistent dimensions
        if seg_logits.shape[-2:] != gt_semantic_segs.shape[-2:]:
            seg_logits = F.interpolate(
                seg_logits, size=gt_semantic_segs.shape[-2:], mode='bilinear', align_corners=False)
        
        # Compute cross-entropy loss
        ce_loss = F.cross_entropy(seg_logits, gt_semantic_segs.squeeze(1).long())
        
        # Initialize loss dictionary
        losses = {'loss_ce': ce_loss}
        
        # If using MLP corrector, add regularization loss
        if self.mlp_corrector:
            # L2 regularization to prevent MLP output from becoming too large
            l2_reg = 0.0
            for param in self.cls_mlp.parameters():
                l2_reg += torch.norm(param, p=2)
            
            # Loss weight, can be adjusted as needed
            reg_weight = 0.01
            reg_loss = reg_weight * l2_reg
            
            # Add regularization loss to total loss
            losses['loss_mlp_reg'] = reg_loss
            losses['loss'] = ce_loss + reg_loss
        else:
            losses['loss'] = ce_loss
        
        return losses

    def freeze_mlp_corrector(self, freeze=True):
        """Freeze or unfreeze MLP corrector weights
        
        Args:
            freeze (bool): Whether to freeze weights, True means freeze, False means unfreeze
        """
        if not hasattr(self, 'cls_mlp'):
            print("Warning: Model does not use MLP corrector, cannot perform freeze/unfreeze operation")
            return
            
        for param in self.cls_mlp.parameters():
            param.requires_grad = not freeze
            
        status = "frozen" if freeze else "unfrozen"
        print(f"MLP corrector weights have been {status}")


def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(',')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices 