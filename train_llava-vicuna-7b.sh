lr=2e-4
beta=0.1
beta_v=1
effective_batch=10
weight_vdpo=1.05
model_name="liuhaotian/llava-v1.6-vicuna-7b"
use_anchor=True
pretrained="/home/yilin/Re-Align/output/llava-vicuna-7b-rdpo-lora-lr-$lr-beta-$beta-new-qkvo-effective_batch-$effective_batch-beta_v-$beta_v-weight_vdpo-$weight_vdpo-use_anchor_$use_anchor"



# python -m debugpy --connect 5679 $(which deepspeed) --include=localhost:0 --master_port 60000 train_rdpo.py \
deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port 60000 train_rdpo.py \
    --model_name_or_path $model_name \
    --data_path "./preference_data/pref_data.json" \
    --deepspeed "./deepspeed/zero2.json" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $effective_batch \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --learning_rate $lr \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --bf16 True \
    --lora_enable True \
    --beta $beta \
    --output_dir $pretrained \
    --image_folder "/data/yilin/train2014/" \
    --mm_projector_lr 2e-5 \
    --mm_projector_type mlp2x_gelu \
    --beta_v $beta_v \
    --weight_vdpo $weight_vdpo \
    # --max_steps 2 \
    # --num_train_epochs 1\
    # --is_resume True \