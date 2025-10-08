lr=1e-6
beta=0.1
beta_v=1
effective_batch=12
weight_vdpo=1
base_model="llava-v1.5-7b"
model_name="liuhaotian/$base_model"


only_anchor=True
yilin=False

only_cal_dpo=False

only_beta_dpo=False
ls_factor_weight=0.6


run_name="yilin-$yilin-$base_model-lr-$lr-acc_batch-$effective_batch-ls_factor_weight-$ls_factor_weight-only_anchor-$only_anchor-only_beta-$only_beta_dpo-only_cal-$only_cal_dpo-h100"
pretrained="/home/yilin/Re-Align/output/$base_model/$run_name"

# data_path="./preference_data/pref_data.json"
data_path="./preference_data/yilin_pref_data.json"

# tasks="hallusion_bench_image"
tasks="pope_random,pope_pop,pope_adv"
# tasks="mmbench_en_test"



# pretrained="/home/yilin/Re-Align/output/llava-v1.5-7b-rdpo-lora-lr-$lr-beta-$beta-new-qkvo-effective_batch-$effective_batch-beta_v-$beta_v-weight_vdpo-$weight_vdpo-use_anchor_$use_anchor"




# deepspeed --include=localhost:0,1 --master_port 60000 train_rdpo.py \
python -m debugpy --connect 5679 $(which deepspeed) --include=localhost:0 --master_port 60000 train_rdpo.py \
    --model_name_or_path $model_name \
    --data_path $data_path \
    --deepspeed "./deepspeed/zero2.json" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $effective_batch \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --learning_rate $lr \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --bf16 True \
    --lora_enable True \
    --beta $beta \
    --output_dir $pretrained \
    --image_folder "/home/yilin/dataset/train2014/" \
    --mm_projector_lr 2e-5 \
    --mm_projector_type mlp2x_gelu \
    --beta_v $beta_v \
    --weight_vdpo $weight_vdpo \
    --only_anchor $only_anchor \
    --run_name $run_name \
    --project_name "yilin-align" \
    --yilin $yilin \
    --ls_factor_weight $ls_factor_weight \
    --only_cal_dpo $only_cal_dpo \
    --only_beta_dpo $only_beta_dpo \
    # --max_steps 2 \
    # --num_train_epochs 1\
    # --is_resume True \