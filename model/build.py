from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from functools import partial
from torch.nn import functional as F
from .memory import Memory
from .mim import mim_decoder


#添加了mim(文本指导图像重建)
class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature)
        
        # Instantiate the Memory class for semantic slot-based token enhancement
        # num_slots: learnable semantic prototypes (not hard-coded body parts)
        self.memory = Memory(
            self.base_model, 
            feature_dim=self.embed_dim, 
            memory_size=getattr(args, 'memory_size', 32),
            num_slots=getattr(args, 'num_slots', 4),  # Semantic slots
            alpha=getattr(args, 'memory_alpha', 0.1),
            momentum=getattr(args, 'memory_momentum', 0.9),
            warmup_epochs=getattr(args, 'memory_warmup_epochs', 5),
            learnable_alpha=getattr(args, 'learnable_alpha', False)
        )
        
        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)
            
        if 'mim' in args.loss_names:
            self.mask_token = nn.Parameter(torch.zeros([1, 3, 32, 32]))
            self.mim_gen = mim_decoder(base_cfg)
            
    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    @staticmethod
    def compute_batch_similarity(features):
        """
        计算batch内特征的平均余弦相似度
        用于监控Memory是否导致特征坍缩
        
        如果后期训练相似度持续升高 → 特征坍缩!
        """
        feat_norm = F.normalize(features, dim=-1)
        sim_matrix = feat_norm @ feat_norm.t()  # [B, B]
        
        # 排除对角线（自身相似度=1）
        B = sim_matrix.size(0)
        mask = ~torch.eye(B, dtype=torch.bool, device=sim_matrix.device)
        avg_similarity = sim_matrix[mask].mean()
        
        return avg_similarity
    
    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image, all_layer_outputs=False):
        """
        Encode image to features
        Args:
            image: input images
            all_layer_outputs: whether to return all layer outputs
        Returns:
            image features (B, D)
        """
        if not all_layer_outputs:
            x = self.base_model.encode_image(image, all_layer_outputs)
        else:
            x, _ = self.base_model.encode_image(image, all_layer_outputs)
        return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text, use_memory=True):
        """
        Encode text to features with optional memory enhancement
        Args:
            text: input text tokens
            use_memory: whether to use memory module for feature enhancement (default: True)
        Returns:
            text features (B, D) - enhanced with part-aware semantic cache if use_memory=True
        """
        # Get text features and token embeddings from CLIP
        if use_memory:
            # Extract token embeddings for token-level enhancement
            text_feats, text_token_embeddings = self.base_model.encode_text(text, return_token_embeddings=True)
            text_token_embeddings = text_token_embeddings.half()
        else:
            text_feats = self.base_model.encode_text(text)
            text_token_embeddings = None
        
        # Extract pooled text features (e.g., [EOS] token)
        t_feats = text_feats[torch.arange(text_feats.shape[0]), text.argmax(dim=-1)].float()
        
        if use_memory:
            # Apply memory enhancement for better retrieval performance
            # Note: In inference mode, image_token is None (no cache update)
            if text_token_embeddings is not None:
                # Token-level enhancement with memory
                text_token_embeddings_enhanced = self.memory(
                    text_tokens=text_token_embeddings.half(),
                    image_tokens=None  # No image features in inference
                )
                # Pool enhanced tokens: use [EOS] token (at argmax position)
                eos_indices = text.argmax(dim=-1)
                fine_text_features = text_token_embeddings_enhanced[torch.arange(text_token_embeddings_enhanced.shape[0]), eos_indices]
            else:
                # Use pooled features as fallback
                fine_text_features = t_feats.half()
            return fine_text_features.float()
        else:
            # Return original features without memory enhancement
            return t_feats
    
    #mim__loss function needed
    def build_masks_for_one_batch(self, batch_size, mask_ratio=0.75, patch_num=48):
        mask_length = int(patch_num * mask_ratio)
        mask_batch = []
        for i in range(int(batch_size)):
            mask_idx = torch.randperm(patch_num)[:mask_length]
            mask1 = torch.zeros([patch_num])
            mask1[mask_idx] = 1
            mask_batch.append(mask1)
        mask = torch.stack(mask_batch, dim=0)
        return mask
    #mim__loss function needed
    def build_masked_image(self, image, masks):
        assert masks is not None
        image = image.cuda()
        masks = masks.cuda()
        B, C, H, W = image.shape
        mask_tokens = self.mask_token.repeat(B, 1, 12, 4)
        temp_mask = masks.reshape(B, 12, 4).unsqueeze(1).repeat(1, 3, 32, 32)
        x = torch.mul(image, (1.0 - temp_mask)) + torch.mul(temp_mask, mask_tokens)
        return x
    #mim__loss function needed
    def get_unmasked_image(self, image, masks):
        image = image.cuda()
        masks = masks.cuda()
        B = image.shape[0]
        temp_mask = masks.reshape(B, 12, 4).unsqueeze(1).repeat(1, 3, 32, 32)
        reserve_token = (temp_mask == 1)
        image = image[reserve_token]
        return image
    #mim__loss function needed
    def get_mim_loss(self, recon, img):
        l1 = nn.L1Loss()
        return l1(recon,img)
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W) where H = 384, W = 128
        x: (N, L, patch_size**2 * 3)
        """
        p = self.visual_encoder.patch_size  # Assuming patch_size is 16
        assert imgs.shape[2] == 384 and imgs.shape[3] == 128
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0  # Check if the image size is divisible by patch_size

        h = imgs.shape[2] // p  # Number of patches along the height
        w = imgs.shape[3] // p  # Number of patches along the width
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
    
    def get_mae_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
    def forward(self, batch):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']
        
        # Extract token embeddings from CLIP text transformer
        model_output = self.base_model(images, caption_ids, return_token_embeddings=True)
        if len(model_output) == 5:
            image_feats, text_feats, image_fine, top_image_fine, text_token_embeddings = model_output
        else:
            image_feats, text_feats, image_fine, top_image_fine = model_output
            text_token_embeddings = None
            
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        
        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})
        
        # ==================== Semantic Slot Memory for Token Enhancement ====================
        # Enhance text token embeddings with memory slots (detached, residual)
        if text_token_embeddings is not None:
            # Token-level enhancement
            text_token_embeddings_enhanced = self.memory(
                text_tokens=text_token_embeddings.half(),
                image_tokens=image_fine.half()
            )
            # Pool enhanced tokens: use [EOS] token (at argmax position)
            eos_indices = caption_ids.argmax(dim=-1)
            fine_text_features = text_token_embeddings_enhanced[torch.arange(text_token_embeddings_enhanced.shape[0]), eos_indices].float()
        else:
            # No token embeddings available, use pooled features
            fine_text_features = t_feats.float()
        
        # Normalize for stable training
        fine_text_features_norm = F.normalize(fine_text_features, dim=-1)
        
        # ==================== Feature Distribution Monitoring ====================
        # 监控batch内特征相似度，验证Memory是否导致特征坍缩
        if self.training:
            with torch.no_grad():
                # 计算增强text特征的batch内相似度
                fine_text_sim = self.compute_batch_similarity(fine_text_features)
                ret.update({'fine_text_sim': fine_text_sim.item()})
                
                # 计算原始text特征的batch内相似度
                orig_text_sim = self.compute_batch_similarity(t_feats)
                ret.update({'orig_text_sim': orig_text_sim.item()})
                
                # 如果观察到：
                # - fine_text_sim 随epoch增长（如0.3→0.7+） → Memory导致坍缩
                # - orig_text_sim 保持稳定（如0.3→0.35）  → 原始特征正常
        
        # ==================== Loss Functions ====================
        
        if 'itc' in self.current_task:
            # Image-Text Contrastive Loss (后期hard negative mining)
            # ITC作为"hard negative修正器"，只在后期启用以提升判别能力
            # 使用原始text特征，避免Memory导致的特征坍缩
            if self.current_epoch >= self.itc_start_epoch:
                ret.update({'itc_loss': objectives.compute_itc_reid(i_feats, t_feats, logit_scale)*self.args.itc_loss_weight})
            else:
                # 早期不使用ITC，避免hard negative导致训练不稳定
                ret.update({'itc_loss': torch.tensor(0.0, device=i_feats.device)})
        
        if 'sdm' in self.current_task:
            # Self-Distance Mining Loss (使用原始文本特征保持判别能力)
            # 关键：避免Memory增强导致的batch-level特征坍缩
            # 原始text features多样性更强，SDM的KL散度梯度更大
            t_feats_norm = F.normalize(t_feats, dim=-1)
            ret.update({'sdm_loss': objectives.compute_sdm(i_feats, t_feats_norm, batch['pids'], logit_scale)})
        
        if 'cmpm' in self.current_task:
            # Cross-Modal Projection Matching (使用原始特征进行对比)
            ret.update({'cmpm_loss': objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})
        
        if 'id' in self.current_task:
            # Identity Classification Loss (使用增强后的文本特征)
            image_logits = self.classifier(i_feats.half()).float()
            
            # 使用fine_text_features进行分类，融合了part-aware语义
            text_logits = self.classifier(fine_text_features.half()).float()
            
            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})
            
            # id_loss = objectives.compute_id_image_only(image_logits, batch['pids'])
            # ret.update({'id_loss': id_loss * self.args.id_loss_weight})

            # 计算准确率
            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})
            
            # Optional: 添加原始文本特征的ID loss作为辅助监督
            if self.training and hasattr(self.args, 'use_auxiliary_id_loss') and self.args.use_auxiliary_id_loss:
                text_logits_original = self.classifier(t_feats.half()).float()
                auxiliary_id_loss = objectives.compute_id(image_logits, text_logits_original, batch['pids'])
                aux_weight = getattr(self.args, 'auxiliary_id_weight', 0.1)
                ret.update({'auxiliary_id_loss': auxiliary_id_loss * aux_weight})
        
        
        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']

            mlm_feats = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_fine, image_fine)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})
            
        if 'mim' in self.current_task:
            mask_for_one_batch = self.build_masks_for_one_batch(images.shape[0])
            masked_img = self.build_masked_image(images, mask_for_one_batch)
            masked_img_feats= self.encode_image(masked_img).half()
            t_feats = t_feats.half()
            recon_image = self.mim_gen(masked_img_feats, t_feats)
            temp_image = self.get_unmasked_image(images, mask_for_one_batch).reshape([images.shape[0], 3*384*96])
            loss_mim = self.get_mim_loss(recon_image, temp_image)
            ret.update({'mim_loss': loss_mim})
            
            
        return ret

def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    # Keep Memory module in fp32 for numerical stability
    model.memory.float()
    return model
