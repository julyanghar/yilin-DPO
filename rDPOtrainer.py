from typing import Dict, List, Union, Tuple, Literal
import torch.distributed
from trl.trainer import DPOTrainer
from trl.trainer.utils import pad_to_length
import torch.nn as nn
import torch.nn.functional as F

class rDPOTrainer(DPOTrainer):
    def concatenated_inputs(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
        concatenated_batch = {}

        if self.is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k or self.is_encoder_decoder else self.padding_value
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k or self.is_encoder_decoder else self.padding_value
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(self.accelerator.device)

        # concatenated_batch["concatenated_images"] = batch["images"] + batch["images"]

        if self.is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1)
            concatenated_batch["concatenated_attention_mask"] = batch["prompt_attention_mask"].repeat(2, 1)

        return concatenated_batch
    
    def concatenated_forward(
        self, model: torch.nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        concatenated_batch = self.concatenated_inputs(batch)
        len_chosen = batch["chosen_labels"].shape[0]
        chosen_batch = concatenated_batch["concatenated_input_ids"][:len_chosen]
        rejected_batch = concatenated_batch["concatenated_input_ids"][len_chosen:]
        chosen_mask = concatenated_batch["concatenated_attention_mask"][:len_chosen]
        rejected_mask = concatenated_batch["concatenated_attention_mask"][len_chosen:]
        chosen_label = concatenated_batch["concatenated_labels"][:len_chosen]
        rejected_label = concatenated_batch["concatenated_labels"][len_chosen:]
        cosine_similarity = -1
        if("cosine_similarity" in batch):
            cosine_similarity = batch["cosine_similarity"]
        # 应该没有用到batch当中的prompt
        chosen_model_kwargs = (
            {
                "labels": chosen_label,
                "decoder_input_ids": concatenated_batch.pop("chosen_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )    
        rejected_model_kwargs = (
            {
                "labels": rejected_label,
                "decoder_input_ids": concatenated_batch.pop("rejected_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )

        # model_kwargs = {
        #     "images": concatenated_batch["concatenated_images"],
        #     "labels": concatenated_batch["concatenated_labels"],
        # }

        # outputs, refined_labels = model(
        #     concatenated_batch["concatenated_input_ids"],
        #     attention_mask=concatenated_batch["concatenated_attention_mask"],
        #     **model_kwargs,
        # )
        # all_logits = outputs.logits.to(torch.float32)

        # all_logps = self._get_batch_logps(
        #     all_logits,
        #     refined_labels,
        #     average_log_prob=False,
        # )

        # chosen_logps = all_logps[:len_chosen]
        # rejected_logps = all_logps[len_chosen:]

        # chosen_logits = all_logits[:len_chosen]
        # rejected_logits = all_logits[len_chosen:]

        # imageless_model_kwargs = {
        #         "labels": batch["chosen_labels"],
        #         "images": batch["image"],
        #         "mask_visual_tokens": True,
        #     }
            
        # imageless_chosen_outputs, imageless_chosen_label = model(
        #     batch["chosen_input_ids"],
        #     attention_mask=batch["chosen_attention_mask"],
        #     **imageless_model_kwargs,
        # )
        # MM-RLHF中，这里同时返回了new_chosen_labels(new_labels)
        chosen_logits = model(
            input_ids = chosen_batch,
            labels = chosen_label,
            images=batch['images'],
            attention_mask=chosen_mask,
            **chosen_model_kwargs,
        ).logits.to(torch.float32)

        _, _, _, _, _, new_chosen_labels = self.model.prepare_inputs_labels_for_multimodal(
                input_ids = chosen_batch,
                position_ids = None,
                attention_mask = chosen_mask,
                past_key_values = None,
                labels = chosen_label,
                images = batch['images']
            )
        
        chosen_logps = self._get_batch_logps(
            chosen_logits,
            new_chosen_labels,
            average_log_prob=False,
        )

        rejected_logits = model(
            input_ids = rejected_batch,
            labels = rejected_label,
            images=batch['images'],
            attention_mask=rejected_mask,
            **rejected_model_kwargs,
        ).logits.to(torch.float32)

        _, _, _, _, _, new_rejected_labels = self.model.prepare_inputs_labels_for_multimodal(
                input_ids = rejected_batch,
                position_ids = None,
                attention_mask = rejected_mask,
                past_key_values = None,
                labels = rejected_label,
                images = batch['images']
            )

        rejected_logps = self._get_noisy_batch_logps(
            rejected_logits,
            rejected_logits,
            new_rejected_labels,
            average_log_prob=False,
        )


        # imageless_model_kwargs = {
        #         "labels": batch["chosen_labels"],
        #         "images": batch["retrieved_images"],
        #     }
        
        # imageless_chosen_outputs, imageless_chosen_label = model(
        #     batch["chosen_input_ids"],
        #     attention_mask=batch["chosen_attention_mask"],
        #     **imageless_model_kwargs,
        # )

        # imageless_chosen_logits = imageless_chosen_outputs.logits.to(torch.float32)

        # imageless_chosen_logps = self._get_batch_logps(
        #     imageless_chosen_logits,
        #     imageless_chosen_label,
        #     average_log_prob=False,
        # )

        imageless_chosen_logits = model(
            input_ids = chosen_batch,
            labels = chosen_label,
            images=batch['retrieved_images'],
            attention_mask=chosen_mask,
            **chosen_model_kwargs,
        ).logits.to(torch.float32)

        _, _, _, _, _, new_imageless_chosen_labels = self.model.prepare_inputs_labels_for_multimodal(
                input_ids = chosen_batch,
                position_ids = None,
                attention_mask = chosen_mask,
                past_key_values = None,
                labels = chosen_label,
                images = batch['retrieved_images']
            )

        imageless_chosen_logps = self._get_noisy_batch_logps(
            imageless_chosen_logits,
            imageless_chosen_logits,
            new_imageless_chosen_labels,
            average_log_prob=False,
        )

        return (chosen_logps, rejected_logps, imageless_chosen_logps, chosen_logits, rejected_logits, imageless_chosen_logits, cosine_similarity)

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_imageless_chosen_logps: torch.FloatTensor, 
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_imageless_chosen_logps: torch.FloatTensor, 
        reference_free: bool = False,
        cosine_similarity: torch.FloatTensor = -1,
    ):
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        chosen_rewards = policy_chosen_logps - reference_chosen_logps
        rejected_rewards = policy_rejected_logps - reference_rejected_logps

        if reference_free:
            ref_logratios = 0

        logits = pi_logratios - ref_logratios  # response preference

        image_conditional_pi_logratios = policy_chosen_logps - policy_imageless_chosen_logps
        image_conditional_ref_logratios = reference_chosen_logps - reference_imageless_chosen_logps

        if reference_free:
            image_conditional_ref_logratios = 0

        image_conditional_logits = image_conditional_pi_logratios - image_conditional_ref_logratios  # image-conditional preference

        
        if self.args.only_anchor:
            anchor_logits = policy_chosen_logps - reference_chosen_logps  # anchored preference
            # anchor_negative_logits = reference_imageless_chosen_logps - policy_imageless_chosen_logps
            losses = -torch.nn.functional.logsigmoid(self.beta * logits) \
            -torch.nn.functional.logsigmoid(self.beta*self.args.beta_v * image_conditional_logits)*self.args.weight_vdpo \
            -torch.nn.functional.logsigmoid(self.beta * anchor_logits) 
            # \
            # -torch.nn.functional.logsigmoid(self.beta * anchor_negative_logits) 
        elif self.args.yilin:
            if isinstance(cosine_similarity, list):
                cosine_similarity = torch.tensor(
                    cosine_similarity,
                    device=logits.device,
                    dtype=logits.dtype
                )
            elif isinstance(cosine_similarity, (float, int)):
                cosine_similarity = torch.full_like(logits, fill_value=cosine_similarity, dtype=logits.dtype, device=logits.device)
            elif isinstance(cosine_similarity, torch.Tensor):
                cosine_similarity = cosine_similarity.to(
                    device=logits.device,
                    dtype=logits.dtype
                )
            beta_used = self.beta * (1 -torch.exp(-self.args.ls_factor_weight * (1 / \
                                        (self.args.similarity_weight * cosine_similarity))))
            beta_used = beta_used.clamp(min=1e-3)
            losses = -torch.nn.functional.logsigmoid(self.beta * logits) \
            -torch.nn.functional.logsigmoid(self.beta * image_conditional_logits) \
        # elif self.args.both:
        #     def all_gather_tensor(tensor):
        #         if torch.distributed.is_available() and torch.distributed.is_initialized():
        #             tensor = tensor.detach()
        #             gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
        #             torch.distributed.all_gather(gathered_tensor, tensor)
        #             tensor = torch.cat(gathered_tensor, dim=0)
        #         # else:
        #         #     print('not distributed')
        #         return tensor
        #     A = all_gather_tensor(logits.detach())
        #     mean = torch.mean(A)
        #     std = torch.std(A)
        #     weight_sample = torch.exp(-0.5 * ((A - mean) / (std + 1e-7)).pow(2))
        #     sample_num = int(weight_sample.numel() * (1 - 0.2) )
        #     sample_index = torch.multinomial(weight_sample, sample_num, replacement=False)
        #     one_hot_like = torch.zeros_like(weight_sample)
        #     one_hot_like[sample_index] = 1
            
        #     A_used = torch.mean(A[sample_index])
        #     beta_used = self.beta * (1 + self.args.ls_factor_weight * (A_used - mean))
        #     cal_loss = F.mse_loss(chosen_rewards,
        #                           torch.tensor(1.0 / (2.0 * beta_used)).to(chosen_rewards)) + F.mse_loss(
        #         rejected_rewards, torch.tensor(-1.0 / (2.0 * beta_used)).to(rejected_rewards))
        #     beta_used = beta_used.clamp(min=1e-3)
        #     losses = -torch.nn.functional.logsigmoid(beta_used * logits) \
        #     -torch.nn.functional.logsigmoid(beta_used * image_conditional_logits)*self.args.weight_vdpo \
        #     + 0.5 * cal_loss
        elif self.args.only_cal_dpo:
            cal_loss = F.mse_loss(chosen_rewards,
                                  torch.tensor(1.0 / (2.0 * self.beta)).to(chosen_rewards)) + F.mse_loss(
                rejected_rewards, torch.tensor(-1.0 / (2.0 * self.beta)).to(rejected_rewards))
            
            losses = -torch.nn.functional.logsigmoid(self.beta * logits) \
            -torch.nn.functional.logsigmoid(self.beta * image_conditional_logits) \
            + 0.5 * cal_loss
        elif self.args.only_beta_dpo:
            def all_gather_tensor(tensor):
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    tensor = tensor.detach()
                    gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
                    torch.distributed.all_gather(gathered_tensor, tensor)
                    tensor = torch.cat(gathered_tensor, dim=0)
                # else:
                #     print('not distributed')
                return tensor
            A = all_gather_tensor(logits.detach())
            mean = torch.mean(A)
            std = torch.std(A)
            weight_sample = torch.exp(-0.5 * ((A - mean) / (std + 1e-7)).pow(2))
            sample_num = int(weight_sample.numel() * (1 - 0.2) )
            sample_index = torch.multinomial(weight_sample, sample_num, replacement=False)
            one_hot_like = torch.zeros_like(weight_sample)
            one_hot_like[sample_index] = 1
            
            A_used = torch.mean(A[sample_index])
            beta_used = self.beta * (1 + self.args.ls_factor_weight * (A_used - mean))
            beta_used = beta_used.clamp(min=1e-3)
            losses = -torch.nn.functional.logsigmoid(beta_used * logits) \
            -torch.nn.functional.logsigmoid(beta_used * image_conditional_logits)
        else:
            losses = -torch.nn.functional.logsigmoid(self.beta * logits) \
                -torch.nn.functional.logsigmoid(self.beta*self.args.beta_v * image_conditional_logits)*self.args.weight_vdpo
            # \
            # -torch.nn.functional.logsigmoid(self.beta * anchor_logits) 
            


        # losses -= policy_chosen_logps / 1024
        
        # KL penalty
        kl =  torch.exp(reference_chosen_logps) * (reference_chosen_logps - policy_chosen_logps)
        # losses += 0.05*kl 

        chosen_rewards = (
            self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        )
        rejected_rewards = (
            self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        )
        imageless_rewards = (
            self.beta * (policy_imageless_chosen_logps - reference_imageless_chosen_logps).detach()
        )

        return losses, chosen_rewards, rejected_rewards, imageless_rewards, kl

    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_imageless_chosen_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_imageless_chosen_logits,
            cosine_similarity,
        ) = self.concatenated_forward(model, batch)
        with torch.no_grad():
            if self.ref_model is None:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        reference_imageless_chosen_logps,
                        _,
                        _,
                        _,
                        _,
                    ) = self.concatenated_forward(self.model, batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    reference_imageless_chosen_logps,
                    _,
                    _,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, batch)

        losses, chosen_rewards, rejected_rewards, imageless_rewards, kl = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            policy_imageless_chosen_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            reference_imageless_chosen_logps,
            cosine_similarity = cosine_similarity,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        imageless_reward_accuracies = (chosen_rewards > imageless_rewards).float()

        loss = losses.mean()

        chosen_labels = batch["chosen_labels"]

        prefix = "eval_" if train_eval == "eval" else ""

        # yilin 临时加入sft_weight
        if not hasattr(self, "sft_weight"):
            self.sft_weight = 0.0

        if self.sft_weight > 0.0:
            if not self.is_encoder_decoder:
                policy_chosen_logits = policy_chosen_logits[..., :-1, :].contiguous()
                chosen_labels = chosen_labels[..., 1:].clone()
            loss_func = nn.CrossEntropyLoss()
            sft_loss = loss_func(policy_chosen_logits.view(-1, policy_chosen_logits.shape[-1]), chosen_labels.view(-1))
            loss = self.sft_weight * sft_loss + loss
            metrics[f"{prefix}sft_loss"] = sft_loss.detach().cpu()

        
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/imageless_chosen"] = imageless_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/imageless_accuracies"] = imageless_reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"{prefix}rewards/imageless_margins"] = (chosen_rewards - imageless_rewards).cpu().mean()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logps/imageless_chosen"] = policy_imageless_chosen_logps.detach().cpu().mean()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().mean()
        metrics[f"{prefix}logits/imageless_chosen"] = policy_imageless_chosen_logits.detach().cpu().mean()
        metrics[f"{prefix}kl div"] = kl.cpu().mean()

        return loss, metrics