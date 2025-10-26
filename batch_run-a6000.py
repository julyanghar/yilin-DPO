import subprocess
import itertools
import os


# re-align

lr=1e-5
beta=0.1
effective_batch=12
base_model="llava-v1.5-7b"
model_name=f"liuhaotian/{base_model}"
# run_name=f"dpo-llava-v1.5-7b-lr-{lr}-acc_batch-$effective_batch-beta-{beta}"


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

# yilin's loss

use_text_similarity=False
ls_factor_text_weight=0.5
use_img_similarity=False
ls_factor_img_weight=0.5
# run_name=f"direct-last_hidden_state-lr-{lr}-acc_batch-$effective_batch-text_weight-{ls_factor_text_weight}-img_weight-{ls_factor_img_weight}"

# beta_dpo setting

beta_dpo=False
ls_factor_weight=0.1
# run_name="beta_dpo-lr-$lr-acc_batch-$effective_batch-beta-$beta-ls_factor_weight-$ls_factor_weight"

# anchor
use_anchor=False
# run_name="anchor-lr-$lr-acc_batch-$effective_batch-beta-$beta"

master_port=60002

data_path="./preference_data/yilin_pref_data_last_hidden_state.json"
# data_path="./preference_data/yilin_pref_data_pooler_output.json"

image_folder="/data/yilin/train2014/"


def test_yilin_align():
    # for ls_factor_text_weight in ls_factor_text_weights:
    times = 10
    use_text_similarity=True
    ls_factor_text_weight=0.5
    use_img_similarity=True
    ls_factor_img_weight=0.5
    for time in range(times):
    # for ls_factor_text_weight, lr in itertools.product(ls_factor_text_weights, lrs):

        ls_factor_img_weight = ls_factor_text_weight
        run_name=f"reverse-pooler_output-lr-{lr}-acc_batch-{effective_batch}-text_weight-{ls_factor_text_weight}-img_weight-{ls_factor_img_weight}-{time}"
        pretrained=f"/home/yilin/Re-Align/output/{base_model}/{run_name}"
        cmd = [
            f"""deepspeed --include=localhost:0,1,2,3,4,5  --master_port {master_port} train_rdpo.py \
            --model_name_or_path {model_name} \
            --data_path {data_path} \
            --deepspeed "./deepspeed/zero2.json" \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps {effective_batch} \
            --evaluation_strategy "no" \
            --save_strategy "no" \
            --learning_rate {lr} \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --bf16 True \
            --lora_enable True \
            --beta {beta} \
            --output_dir {pretrained} \
            --image_folder {image_folder} \
            --mm_projector_lr 2e-5 \
            --mm_projector_type mlp2x_gelu \
            --run_name {run_name} \
            --project_name "yilin-align" \
            --use_text_similarity {use_text_similarity}  \
            --ls_factor_text_weight {ls_factor_text_weight} \
            --use_img_similarity {use_img_similarity} \
            --ls_factor_img_weight {ls_factor_img_weight} \
            --beta_dpo {beta_dpo} \
            --ls_factor_weight {ls_factor_weight} \
            --use_anchor {use_anchor} """
        ]

        print(f"🚀 Running {base_model} | lr={lr}, bs={effective_batch}")
        ret = subprocess.run(cmd, shell=True)

        if ret.returncode != 0:
            print(f"❌ Failed: {base_model} lr={lr} bs={effective_batch}")
        else:
            print(f"✅ Finished: {base_model} lr={lr} bs={effective_batch}")

def test_re_align():
    # run_name="anchor-lr-$lr-acc_batch-$effective_batch-beta-$beta"
    beta=0.1
    effective_batch=12
    times = 10
    lr = 1e-5
    master_port=60009
    # for ls_factor_text_weight in ls_factor_text_weights:
    for time in range(5,times):
        run_name=f"re-align-add_dis-lr-{lr}-acc_batch-{effective_batch}-beta-{beta}-{time}"
        pretrained=f"/home/yilin/Re-Align/output/{base_model}/{run_name}"
        cmd = [
            f"""deepspeed --include=localhost:0,1,2,3,4,5  --master_port {master_port} train_rdpo.py \
            --model_name_or_path {model_name} \
            --data_path {data_path} \
            --deepspeed "./deepspeed/zero2.json" \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps {effective_batch} \
            --evaluation_strategy "no" \
            --save_strategy "no" \
            --learning_rate {lr} \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --bf16 True \
            --lora_enable True \
            --beta {beta} \
            --output_dir {pretrained} \
            --image_folder {image_folder} \
            --mm_projector_lr 2e-5 \
            --mm_projector_type mlp2x_gelu \
            --run_name {run_name} \
            --project_name "yilin-align" \
            --use_text_similarity {use_text_similarity}  \
            --ls_factor_text_weight {ls_factor_text_weight} \
            --use_img_similarity {use_img_similarity} \
            --ls_factor_img_weight {ls_factor_img_weight} \
            --beta_dpo {beta_dpo} \
            --ls_factor_weight {ls_factor_weight} \
            --use_anchor {use_anchor} """
            
        ]

        print(f"🚀 Running {base_model} | lr={lr}, bs={effective_batch}")
        ret = subprocess.run(cmd, shell=True)

        if ret.returncode != 0:
            print(f"❌ Failed: {base_model} lr={lr} bs={effective_batch}")
        else:
            print(f"✅ Finished: {base_model} lr={lr} bs={effective_batch}")



def test_beta_dpo():
    times = 10
    beta_dpo=True
    ls_factor_weight=0.1
    for time in range(times):
        ls_factor_img_weight = ls_factor_text_weight
        run_name=f"beta_dpo-lr-{lr}-acc_batch-{effective_batch}-ls_factor_weight-{ls_factor_weight}-{time}"
        pretrained=f"/home/yilin/Re-Align/output/{base_model}/{run_name}"
        cmd = [
            f"""deepspeed --include=localhost:0,1,2,3,4,5  --master_port {master_port} train_rdpo.py \
            --model_name_or_path {model_name} \
            --data_path {data_path} \
            --deepspeed "./deepspeed/zero2.json" \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps {effective_batch} \
            --evaluation_strategy "no" \
            --save_strategy "no" \
            --learning_rate {lr} \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --bf16 True \
            --lora_enable True \
            --beta {beta} \
            --output_dir {pretrained} \
            --image_folder {image_folder} \
            --mm_projector_lr 2e-5 \
            --mm_projector_type mlp2x_gelu \
            --run_name {run_name} \
            --project_name "yilin-align" \
            --use_text_similarity {use_text_similarity}  \
            --ls_factor_text_weight {ls_factor_text_weight} \
            --use_img_similarity {use_img_similarity} \
            --ls_factor_img_weight {ls_factor_img_weight} \
            --beta_dpo {beta_dpo} \
            --ls_factor_weight {ls_factor_weight} \
            --use_anchor {use_anchor} """
        ]

        print(f"🚀 Running {base_model} | lr={lr}, bs={effective_batch}")
        ret = subprocess.run(cmd, shell=True)

        if ret.returncode != 0:
            print(f"❌ Failed: {base_model} lr={lr} bs={effective_batch}")
        else:
            print(f"✅ Finished: {base_model} lr={lr} bs={effective_batch}")
    



def test_anchor():
    times = 10
    use_anchor=True
    anchor_beta=0.1
    num_train_epochs=2
    master_port=50123
    for time in range(times):
        run_name=f"anchor-lr-{lr}-acc_batch-{effective_batch}-beta-{beta}-epoch-{num_train_epochs}-{time}"
        pretrained=f"/home/yilin/Re-Align/output/{base_model}/{run_name}"
        cmd = [
            #  f"""python -m debugpy --connect 5679 $(which deepspeed) --include=localhost:0,1 --master_port {master_port} train_rdpo.py \
            f"""deepspeed --include=localhost:2,3  --master_port {master_port} train_rdpo.py \
            --model_name_or_path {model_name} \
            --data_path {data_path} \
            --deepspeed "./deepspeed/zero2.json" \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps {effective_batch} \
            --evaluation_strategy "no" \
            --save_strategy "no" \
            --learning_rate {lr} \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --bf16 True \
            --lora_enable True \
            --beta {beta} \
            --output_dir {pretrained} \
            --image_folder {image_folder} \
            --mm_projector_lr 2e-5 \
            --mm_projector_type mlp2x_gelu \
            --run_name {run_name} \
            --project_name "yilin-align" \
            --use_anchor {use_anchor} \
            --anchor_beta {anchor_beta}"""
        ]

        print(f"🚀 Running {base_model} | lr={lr}, bs={effective_batch}")
        ret = subprocess.run(cmd, shell=True)

        if ret.returncode != 0:
            print(f"❌ Failed: {base_model} lr={lr} bs={effective_batch}")
        else:
            print(f"✅ Finished: {base_model} lr={lr} bs={effective_batch}")
    

def test_yilin_anchor():
    times = 5
    yilin_anchor=True
    master_port=12354
    ls_factor_weights = [0.5,0.7,0.3,0.9,0.1]
    for time, ls_factor_weight in itertools.product(range(times), ls_factor_weights):
    # for time in range(times):
        run_name=f"direct-yilin_anchor-lr-{lr}-acc_batch-{effective_batch}-beta-{beta}-ls_factor_weight-{ls_factor_weight}-{time}"
        pretrained=f"/home/yilin/Re-Align/output/{base_model}/{run_name}"
        cmd = [
            # f"""python -m debugpy --connect 5679 $(which deepspeed) --include=localhost:0 --master_port 60000 train_rdpo.py \
            f"""deepspeed --include=localhost:0,1,2,3,4,5  --master_port {master_port} train_rdpo.py \
            --model_name_or_path {model_name} \
            --data_path {data_path} \
            --deepspeed "./deepspeed/zero2.json" \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps {effective_batch} \
            --evaluation_strategy "no" \
            --save_strategy "no" \
            --learning_rate {lr} \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --bf16 True \
            --lora_enable True \
            --beta {beta} \
            --output_dir {pretrained} \
            --image_folder {image_folder} \
            --mm_projector_lr 2e-5 \
            --mm_projector_type mlp2x_gelu \
            --run_name {run_name} \
            --project_name "yilin-align" \
            --use_text_similarity {use_text_similarity}  \
            --ls_factor_text_weight {ls_factor_text_weight} \
            --use_img_similarity {use_img_similarity} \
            --ls_factor_img_weight {ls_factor_img_weight} \
            --beta_dpo {beta_dpo} \
            --ls_factor_weight {ls_factor_weight} \
            --use_anchor {use_anchor} \
            --yilin_anchor {yilin_anchor} """
        ]

        print(f"🚀 Running {base_model} | lr={lr}, bs={effective_batch}")
        ret = subprocess.run(cmd, shell=True)

        if ret.returncode != 0:
            print(f"❌ Failed: {base_model} lr={lr} bs={effective_batch}")
        else:
            print(f"✅ Finished: {base_model} lr={lr} bs={effective_batch}")
    
def test_sample_weight():
    times = 5
    use_sample_weight=True
    master_port=12423
    ls_factor_weights = [1,2]
    lrs=[1e-5]
    for time, lr, ls_factor_weight in itertools.product(range(times), lrs, ls_factor_weights):
    # for time in range(times):
        run_name=f"use_sample_weight-exp-lr-{lr}-acc_batch-{effective_batch}-beta-{beta}-ls_factor_weight-{ls_factor_weight}-{time}"
        pretrained=f"/home/yilin/Re-Align/output/{base_model}/{run_name}"
        cmd = [
            # f"""python -m debugpy --connect 5679 $(which deepspeed) --include=localhost:0 --master_port {master_port} train_rdpo.py \
            f"""deepspeed --include=localhost:0,1,2,3,4,5  --master_port {master_port} train_rdpo.py \
            --model_name_or_path {model_name} \
            --data_path {data_path} \
            --deepspeed "./deepspeed/zero2.json" \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps {effective_batch} \
            --evaluation_strategy "no" \
            --save_strategy "no" \
            --learning_rate {lr} \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --bf16 True \
            --lora_enable True \
            --beta {beta} \
            --output_dir {pretrained} \
            --image_folder {image_folder} \
            --mm_projector_lr 2e-5 \
            --mm_projector_type mlp2x_gelu \
            --run_name {run_name} \
            --project_name "yilin-align" \
            --use_text_similarity {use_text_similarity}  \
            --ls_factor_text_weight {ls_factor_text_weight} \
            --use_img_similarity {use_img_similarity} \
            --ls_factor_img_weight {ls_factor_img_weight} \
            --beta_dpo {beta_dpo} \
            --ls_factor_weight {ls_factor_weight} \
            --use_anchor {use_anchor} \
            --use_sample_weight {use_sample_weight} """
        ]

        print(f"🚀 Running {base_model} | lr={lr}, bs={effective_batch}")
        ret = subprocess.run(cmd, shell=True)

        if ret.returncode != 0:
            print(f"❌ Failed: {base_model} lr={lr} bs={effective_batch}")
        else:
            print(f"✅ Finished: {base_model} lr={lr} bs={effective_batch}")
    

def test_sample_weight_sliding_avg():
    times = 5
    use_sample_weight=True
    master_port=12423
    ls_factor_weights = [1,2]
    lrs=[1e-5]
    for time, lr, ls_factor_weight in itertools.product(range(times), lrs, ls_factor_weights):
    # for time in range(times):
        run_name=f"sliding_avg_norm_-lr-{lr}-acc_batch-{effective_batch}-beta-{beta}-ls_factor_weight-{ls_factor_weight}-{time}"
        pretrained=f"/home/yilin/Re-Align/output/{base_model}/{run_name}"
        cmd = [
            # f"""python -m debugpy --connect 5679 $(which deepspeed) --include=localhost:0,1 --master_port {master_port} train_rdpo.py \
            f"""deepspeed --include=localhost:0,1,2,3,4,5  --master_port {master_port} train_rdpo.py \
            --model_name_or_path {model_name} \
            --data_path {data_path} \
            --deepspeed "./deepspeed/zero2.json" \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps {effective_batch} \
            --evaluation_strategy "no" \
            --save_strategy "no" \
            --learning_rate {lr} \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --bf16 True \
            --lora_enable True \
            --beta {beta} \
            --output_dir {pretrained} \
            --image_folder {image_folder} \
            --mm_projector_lr 2e-5 \
            --mm_projector_type mlp2x_gelu \
            --run_name {run_name} \
            --project_name "yilin-align" \
            --use_text_similarity {use_text_similarity}  \
            --ls_factor_text_weight {ls_factor_text_weight} \
            --use_img_similarity {use_img_similarity} \
            --ls_factor_img_weight {ls_factor_img_weight} \
            --beta_dpo {beta_dpo} \
            --ls_factor_weight {ls_factor_weight} \
            --use_anchor {use_anchor} \
            --use_sample_weight {use_sample_weight} """
        ]

        print(f"🚀 Running {base_model} | lr={lr}, bs={effective_batch}")
        ret = subprocess.run(cmd, shell=True)

        if ret.returncode != 0:
            print(f"❌ Failed: {base_model} lr={lr} bs={effective_batch}")
        else:
            print(f"✅ Finished: {base_model} lr={lr} bs={effective_batch}")

def test_filter_text():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"
    times = 6
    master_port=60040
    lrs=[1e-5]
    num_train_epochs=2
    filter_factor_text_upper = 1
    filter_factor_text_lower = 0
    for time, lr, filter_factor_text_lower, use_anchor, num_train_epochs in itertools.product(
                        range(times), lrs, [0.999], [True], [2]):
    # for time in range(times):
        run_name=f"filter-text-lr-{lr}-batch-{effective_batch}-beta-{beta}-upper-{filter_factor_text_upper}-lower-{filter_factor_text_lower}-use_anchor-{use_anchor}-epoch-{num_train_epochs}-{time}"
        pretrained=f"/home/yilin/Re-Align/output/{base_model}/{run_name}"
        cmd = [
            # f"""python -m debugpy --connect 5679 $(which deepspeed) --include=localhost:0,1 --master_port {master_port} train_rdpo.py \
            f"""deepspeed --include=localhost:0,1 --master_port {master_port} train_rdpo.py \
            --model_name_or_path {model_name} \
            --data_path {data_path} \
            --deepspeed "./deepspeed/zero2.json" \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps {effective_batch} \
            --evaluation_strategy "no" \
            --save_strategy "no" \
            --learning_rate {lr} \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --bf16 True \
            --lora_enable True \
            --beta {beta} \
            --output_dir {pretrained} \
            --image_folder {image_folder} \
            --mm_projector_lr 2e-5 \
            --mm_projector_type mlp2x_gelu \
            --run_name {run_name} \
            --project_name "yilin-align" \
            --num_train_epochs {num_train_epochs} \
            --use_anchor {use_anchor} \
            --filter_factor_text_upper {filter_factor_text_upper} \
            --filter_factor_text_lower {filter_factor_text_lower} """
        ]

        print(f"🚀 Running {base_model} | lr={lr}, bs={effective_batch}")
        ret = subprocess.run(cmd, shell=True)

        if ret.returncode != 0:
            print(f"❌ Failed: {base_model} lr={lr} bs={effective_batch}")
        else:
            print(f"✅ Finished: {base_model} lr={lr} bs={effective_batch}")

def test_filter_img():
    times = 5
    master_port=60111
    lrs=[1e-5]
    filter_factor_img_upper = 1
    filter_factor_img_lower = 0
    data_path="./preference_data/yilin_pref_data_pooler_output.json"
    for time, lr, use_anchor, filter_factor_img_lower in itertools.product(
        range(times), lrs,[True], [0.7]):
    # for time in range(times):
        run_name=f"filter-pooler_output-lr-{lr}-acc_batch-{effective_batch}-beta-{beta}-upper-{filter_factor_img_upper}-lower-{filter_factor_img_lower}-use_anchor-{use_anchor}-{time}"
        pretrained=f"/home/yilin/Re-Align/output/{base_model}/{run_name}"
        cmd = [
            # f"""python -m debugpy --connect 5679 $(which deepspeed) --include=localhost:2 --master_port {master_port} train_rdpo.py \
            f"""deepspeed --include=localhost:0,1 --master_port {master_port} train_rdpo.py \
            --model_name_or_path {model_name} \
            --data_path {data_path} \
            --deepspeed "./deepspeed/zero2.json" \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps {effective_batch} \
            --evaluation_strategy "no" \
            --save_strategy "no" \
            --learning_rate {lr} \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --bf16 True \
            --lora_enable True \
            --beta {beta} \
            --output_dir {pretrained} \
            --image_folder {image_folder} \
            --mm_projector_lr 2e-5 \
            --mm_projector_type mlp2x_gelu \
            --run_name {run_name} \
            --project_name "yilin-align" \
            --filter_factor_img_lower {filter_factor_img_lower} \
            --filter_factor_img_upper {filter_factor_img_upper} \
            --use_anchor {use_anchor}"""
        ]

        print(f"🚀 Running {base_model} | lr={lr}, bs={effective_batch}")
        ret = subprocess.run(cmd, shell=True)

        if ret.returncode != 0:
            print(f"❌ Failed: {base_model} lr={lr} bs={effective_batch}")
        else:
            print(f"✅ Finished: {base_model} lr={lr} bs={effective_batch}")


def only_anchor():
    times = 10
    use_anchor=True
    anchor_beta=0.1
    for time in range(times):
        ls_factor_img_weight = ls_factor_text_weight
        run_name=f"only_use_anchor-lr-{lr}-acc_batch-{effective_batch}-beta-{beta}-{time}"
        pretrained=f"/home/yilin/Re-Align/output/{base_model}/{run_name}"
        cmd = [
            #  f"""python -m debugpy --connect 5679 $(which deepspeed) --include=localhost:0,1 --master_port {master_port} train_rdpo.py \
            f"""deepspeed --include=localhost:0,1  --master_port {master_port} train_rdpo.py \
            --model_name_or_path {model_name} \
            --data_path {data_path} \
            --deepspeed "./deepspeed/zero2.json" \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps {effective_batch} \
            --evaluation_strategy "no" \
            --save_strategy "no" \
            --learning_rate {lr} \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --bf16 True \
            --lora_enable True \
            --beta {beta} \
            --output_dir {pretrained} \
            --image_folder {image_folder} \
            --mm_projector_lr 2e-5 \
            --mm_projector_type mlp2x_gelu \
            --run_name {run_name} \
            --project_name "yilin-align" \
            --use_text_similarity {use_text_similarity}  \
            --ls_factor_text_weight {ls_factor_text_weight} \
            --use_img_similarity {use_img_similarity} \
            --ls_factor_img_weight {ls_factor_img_weight} \
            --beta_dpo {beta_dpo} \
            --ls_factor_weight {ls_factor_weight} \
            --use_anchor {use_anchor} \
            --anchor_beta {anchor_beta}"""
        ]

        print(f"🚀 Running {base_model} | lr={lr}, bs={effective_batch}")
        ret = subprocess.run(cmd, shell=True)

        if ret.returncode != 0:
            print(f"❌ Failed: {base_model} lr={lr} bs={effective_batch}")
        else:
            print(f"✅ Finished: {base_model} lr={lr} bs={effective_batch}")




def test_only_anchor():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"
    times = 5
    master_port=60040
    lrs=[1e-5]
    num_train_epochs=1
    filter_factor_img_upper=1
    filter_factor_img_lower=0.999
    filter_factor_text_upper=1
    filter_factor_text_lower=0.999
    for time, lr, filter_factor_text_lower, use_anchor, num_train_epochs in itertools.product(
                        range(times), lrs, [0.999], [True], [1]):
    # for time in range(times):
        run_name=f"only_use_anchor-lr-{lr}-batch-{effective_batch}-beta-{beta}-epoch-{num_train_epochs}-{time}"
        pretrained=f"/home/yilin/Re-Align/output/{base_model}/{run_name}"
        cmd = [
            # f"""deepspeed --include=localhost:0,1 --master_port {master_port} train_rdpo.py \
            f"""python -m debugpy --connect 5679 $(which deepspeed) --include=localhost:0 --master_port {master_port} train_rdpo.py \
            --model_name_or_path {model_name} \
            --data_path {data_path} \
            --deepspeed "./deepspeed/zero2.json" \
            --per_device_train_batch_size 1 \
            --per_device_eval_batch_size 1 \
            --gradient_accumulation_steps {effective_batch} \
            --evaluation_strategy "no" \
            --save_strategy "no" \
            --learning_rate {lr} \
            --weight_decay 0. \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine" \
            --bf16 True \
            --lora_enable True \
            --beta {beta} \
            --output_dir {pretrained} \
            --image_folder {image_folder} \
            --mm_projector_lr 2e-5 \
            --mm_projector_type mlp2x_gelu \
            --run_name {run_name} \
            --project_name "yilin-align" \
            --num_train_epochs {num_train_epochs} \
            --use_anchor {use_anchor} \
            --filter_factor_img_upper {filter_factor_img_upper} \
            --filter_factor_img_lower {filter_factor_img_lower} \
            --filter_factor_text_upper {filter_factor_text_upper} \
            --filter_factor_text_lower {filter_factor_text_lower} """
        ]

        print(f"🚀 Running {base_model} | lr={lr}, bs={effective_batch}")
        ret = subprocess.run(cmd, shell=True)

        if ret.returncode != 0:
            print(f"❌ Failed: {base_model} lr={lr} bs={effective_batch}")
        else:
            print(f"✅ Finished: {base_model} lr={lr} bs={effective_batch}")



if __name__ == "__main__":
    # test_beta_dpo()
    # test_anchor()
    # test_yilin_anchor()
    # test_sample_weight()
    # test_sample_weight_sliding_avg()
    # test_re_align()
    test_filter_img()
    # test_filter_text()
    # test_only_anchor()