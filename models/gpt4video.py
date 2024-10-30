import os
import torch
import torch.nn as nn
from typing import List
import lightning.pytorch as pl
import einops
from models.mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
import numpy as np
from deepspeed.ops.adam import DeepSpeedCPUAdam
from torch.optim.lr_scheduler import StepLR
from peft import get_peft_model, LoraConfig, TaskType
from dataset.data_helper import tokenizer


class GPT4Video(pl.LightningModule):
    """
    GPT4Video.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.tokenizer = tokenizer
        print("Loadding model")
        if args.accelerator == 'cpu':
            self.model = MplugOwlForConditionalGeneration.from_pretrained(self.hparams.llm_model)
        else:
            self.model = MplugOwlForConditionalGeneration.from_pretrained(self.hparams.llm_model, torch_dtype=torch.bfloat16)

        if args.llm_use_lora:
            for param in self.model.parameters():
                # freeze base model's layers
                param.requires_grad = False
            peft_config = LoraConfig(
                target_modules=r'.*language_model.*\.(q_proj|v_proj)', 
                inference_mode=args.lora_inference, 
                r=args.llm_r, 
                lora_alpha=args.llm_alpha, 
                lora_dropout=args.lora_dropout
            )

            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        else:
            for name, param in self.model.named_parameters():
                if 'language_model' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        self.input_embeddings = self.model.get_input_embeddings()

        self.val_step_outputs = []
        self.val_score = 100.0

        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location='cpu')['model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')


    def video_encoder(self, video_pixel_values):
        video_embeds = self.model.vision_model(video_pixel_values, return_dict=True).last_hidden_state # bs,4,257,1024

        video_attention_mask = torch.ones(video_embeds.size()[:-1], dtype=torch.long, device=video_embeds.device) # bs,4,257
        video_attention_mask = einops.rearrange(
            video_attention_mask, 'b t n -> b (t n)'
        ) # bs 1028
        query_tokens = self.model.query_tokens.expand(video_embeds.shape[0], -1, -1)
        temporal_query_tokens = self.model.temporal_query_tokens.expand(video_embeds.shape[0], -1, -1)

        video_query_features = self.model.abstractor(
            query_embeds=query_tokens,
            temporal_query_embeds=temporal_query_tokens,
            encoder_hidden_states=video_embeds,
            encoder_attention_mask=video_attention_mask,
        )["last_hidden_state"]
        return video_query_features


    def prompt_wrap(self, video_embeds):
        video_embeds = self.video_encoder(video_embeds)
        batch_size = video_embeds.shape[0]
        prefix_ids = self.tokenizer('Human:', return_tensors="pt", add_special_tokens=False).to(video_embeds.device)
        prefix_embs = self.input_embeddings(prefix_ids.input_ids).expand(batch_size, -1, -1)
        wrapped_video_embeds = torch.cat([prefix_embs, video_embeds], dim=1)
        return wrapped_video_embeds


    def generation(self, video_embeds, labels):
        video_embeds = self.prompt_wrap(video_embeds)
        input_embs = self.input_embeddings(labels) # bs words_length, llm_emb_dim
        empty_labels = (
            torch.ones([video_embeds.shape[0], video_embeds.shape[1]],
                       dtype=torch.long).to(video_embeds[0].device).fill_(-100)
        )
        labels = torch.cat([empty_labels, labels], dim=1)
        labels = labels.masked_fill(labels == 0, -100)

        input_embs = torch.cat([video_embeds, input_embs], dim=1)
        
        output = self.model.language_model(inputs_embeds=input_embs,
                        labels=labels,
                        output_hidden_states=True)
        return output


    def generation_text(self, labels):
        input_embs = self.input_embeddings(labels) # bs words_length, llm_emb_dim
        batch_size, _ = labels.shape 

        labels = labels.masked_fill(labels == 0, -100)
        output = self.model.language_model(inputs_embeds=input_embs,
                        labels=labels,
                        output_hidden_states=True)
        return output


    def training_step(self, batch, batch_idx):
        to_log = {}
        if "video" in batch:
            model_output = self.generation(video_embeds=batch['video'], labels = batch['input_ids'])
        else:
            model_output = self.generation_text(labels = batch['input_ids'])        
        loss = model_output.loss
        to_log['loss'] = loss
        self.log_dict(to_log, prog_bar=True)
        return loss

    def save_checkpoint(self, eval_res):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in param_grad_dic.keys():
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step":global_step
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'checkpoints'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'checkpoints',
            "checkpoint_epoch{}_step{}_val_loss{:3f}.pth".format(current_epoch, global_step, eval_res),
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)
    
    def validation_step(self, batch, batch_idx):
        to_log = {}
        if "video" in batch:
            model_output = self.generation(video_embeds=batch['video'], labels = batch['input_ids'])
        else:
            model_output = self.generation_text(labels = batch['input_ids'])        
        loss = model_output.loss    
        to_log['val_loss'] = loss
        self.log_dict(to_log, sync_dist=True, prog_bar=True)
        self.val_step_outputs.append({"val_loss": loss})
        return to_log
    
    def decode(self, output_token):
        output_text = self.tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.replace('<unk>', '').replace('<s>', '').replace('</s>', '').strip()
        return output_text

    def on_validation_epoch_end(self):
        # last_emb, video_path, val_loss = [], [], []
        val_loss = []
        for i in self.val_step_outputs:
            val_loss.append(i['val_loss'].item())
        val_loss = np.mean(val_loss)
        if self.trainer.local_rank == 0:
            self.save_checkpoint(val_loss)

    def configure_optimizers(self):
        if 'deepspeed' in self.hparams.strategy:
            optimizer = DeepSpeedCPUAdam(self.parameters(), lr=self.hparams.learning_rate)
            scheduler = StepLR(optimizer, step_size=1, gamma=0.85)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
            scheduler = StepLR(optimizer, step_size=1, gamma=0.85)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()