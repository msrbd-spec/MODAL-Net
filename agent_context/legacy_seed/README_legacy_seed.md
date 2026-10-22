# Legacy seed modules (from the previous RetiNA-Net codebase)

Copy these files into the new MODAL-Net repo keeping the same relative paths, then adapt them as described in the master plan (section "Legacy seed adaptation").

| File | What it does | Adaptation needed for MODAL-Net |
|---|---|---|
| src/preprocessing/lesion_detection.py | Classical lesion detector (microaneurysm / hemorrhage / exudate). detect_lesions_batch() writes one .npz per image (mask 128x128x3, count 3, presence 3, heatmap 128x128). | None. Use as is for `--mode detect_lesions`. |
| src/data/eyepacs_dataset.py | SSL dataset: two augmented views + lesion pseudo-labels, globbing EyePACS images that have a .npz label. | None required. It uses crop + Ben Graham (sigma 10, native resolution) only. Keep for SSL, or pass the MODAL preprocessing function if the plan's preprocessing winner differs. |
| src/models/ssl_model.py | SSLModel (InfoNCE + lesion-type / count / mask heads) around a SwinV2 timm backbone created with features_only=True and NHWC permutes. | MUST be rewritten for MaxViT-L: create the backbone with timm.create_model(name, pretrained=True, num_classes=0) (same class as GradeNet), take stage-3 and stage-4 feature maps through forward hooks on backbone.stages[2] and backbone.stages[3] (NCHW), set stage_channels from backbone.feature_info or a dummy forward, and save backbone.state_dict() with the key names of the num_classes=0 model so GradeNet can load it. Keep the heads and losses. |
| src/training/ssl_trainer.py | SSL training loop with EMA, resume from <ssl_save_dir>/ssl_training_checkpoint.pth, saves ssl_pretrained_backbone.pth. | Keep. Always give every SSL variant its own ssl_save_dir (resume logic skips training when the checkpoint is complete). Replace torch.cuda.amp.GradScaler/autocast by torch.amp equivalents if the installed torch warns. |
| configs/config_ssl.yaml | Old SSL settings (SwinV2). | Use as a template for the new configs/config_ssl.yaml; change ssl_backbone to the MaxViT-L timm id. |

Line endings were converted to LF.
