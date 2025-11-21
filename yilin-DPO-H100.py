import subprocess
import itertools
import os


# re-align

lr=1e-5
beta=0.1
effective_batch=12
base_model="llava-v1.5-7b"
# base_model="llava-v1.6-vicuna-7b"
model_name=f"liuhaotian/{base_model}"
# run_name=f"dpo-llava-v1.5-7b-lr-$lr-batch-$effective_batch-beta-{beta}"


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# yilin's loss

use_text_similarity=False
ls_factor_text_weight=0.5
use_img_similarity=False
ls_factor_img_weight=0.5
# run_name=f"reverse-pooler_output-lr-{lr}-batch-$effective_batch-text_weight-{ls_factor_text_weight}-img_weight-{ls_factor_img_weight}"

# beta_dpo setting

beta_dpo=False
ls_factor_weight=0.1
# run_name="beta_dpo-lr-$lr-batch-$effective_batch-beta-$beta-ls_factor_weight-$ls_factor_weight"

# anchor
use_anchor=False
# run_name="anchor-lr-$lr-batch-$effective_batch-beta-$beta"

master_ports=[60010,60011,60012]

# data_path="./preference_data/yilin_pref_data_last_hidden_state.json"
# data_path="./dataset/mmrlhf/converted-dpo_pairs.json"
data_path="./preference_data/proc_pooler_output.json"

# image_folder="/data/yilin/train2014/"
image_folder="/home/yilin/dataset/train2014/"
# image_folder="/home/yilin/yilin-DPO/dataset/mmrlhf/"

pretrained="/home/yilin/yilin-DPO/output/$base_model/$run_name"



def test_re_align():
    i = 0
    times = 1
    lrs = [1e-5]
    betas = [0]
    processes = []
    num_train_epochs=2
    data_path="./preference_data/proc_pooler_output.json"

    for time, lr, beta in itertools.product(range(times), lrs, betas):
    
        if i >= len(master_ports):
            raise ValueError(f"i={i} 超出 master_ports 长度 {len(master_ports)}")
        master_port = master_ports[i]
        i += 1
        run_name=f"re_align-lr-{lr}-batch-{effective_batch}-beta-{beta}-epoch-{num_train_epochs}-{time}"
        pretrained=f"/home/yilin/yilin-DPO/output/{base_model}/{run_name}"
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
            --use_anchor False \
            --filter_factor_text_lower 20 \
            --filter_factor_img_lower 20 \
            --num_train_epochs {num_train_epochs}"""
        ]
        print(f"🚀 Running {run_name}")
        p = subprocess.Popen(cmd, shell=True)
        p.cmd = cmd
        
        processes.append(p)
        print(f"当前进程数: {len(processes)}")
            # 如果已经启动了 1 个，就等待它们跑完
        if len(processes) == 1:
            for p in processes:
                p.wait()     # 等待当前这批都结束
                if p.returncode != 0:
                    print(f"❌ Failed: {base_model} cmd={p.cmd}")
                else:
                    print(f"✅ Finished: {base_model} cmd={p.cmd}")
            i = 0
            processes = []     # 清空进程列表，开始下一批


    for p in processes:
        p.wait()     # 等待当前这批都结束
        if p.returncode != 0:
            print(f"❌ Failed: {base_model} cmd={p.cmd}")
        else:
            print(f"✅ Finished: {base_model} cmd={p.cmd}")
    processes = [] 

def test_filter():
    i = 0

    # data_path="./preference_data/filterred.json"
    data_path="./preference_data/top2_masked_pooler.json"
    num_train_epochs = 2
    image_folder="/home/yilin/dataset/train2014/"

    times = 6
    lr = 1e-5
    filter_factor_text_upper = 1
    filter_factor_text_lower = 0
    filter_factor_img_upper = 1
    filter_factor_img_lower = 0
    master_ports=[60001,60011,60012]
    base_model = "llava-v1.5-7b"
    # base_model="llava-v1.6-vicuna-7b"
    model_name = f"liuhaotian/{base_model}"
    processes = []
    for lr, time, use_anchor in itertools.product(
                                    [1e-5], range(1), [False],    
                                ):

            if i >= len(master_ports):
                raise ValueError(f"i={i} 超出 master_ports 长度 {len(master_ports)}")
            master_port = master_ports[i]
            i += 1
            run_name=f"H100-no-noisy-full-lr-{lr}-batch-{effective_batch}-beta-{beta}-anchor-{use_anchor}-{base_model}-{time}"
            pretrained=f"/home/yilin/yilin-DPO/output/{base_model}/{run_name}"

            cmd = [
                # f"""deepspeed --include=localhost:0,1  --master_port {master_port} train_rdpo.py \
                f"""python -m debugpy --connect 5679 $(which deepspeed) --include=localhost:1 --master_port {master_port} train_rdpo.py \
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
                --num_train_epochs {num_train_epochs} \
                --filter_factor_img_upper {filter_factor_img_upper} \
                --filter_factor_img_lower {filter_factor_img_lower} \
                --filter_factor_text_lower {filter_factor_text_lower} \
                --filter_factor_text_upper {filter_factor_text_upper} """
            ]
            print(f"🚀 Running {run_name}")
            p = subprocess.Popen(cmd, shell=True)
            p.cmd = cmd
            
            processes.append(p)
            print(f"当前进程数: {len(processes)}")
                # 如果已经启动了 3 个，就等待它们跑完
            if len(processes) == 1:
                for p in processes:
                    p.wait()     # 等待当前这批都结束
                    if p.returncode != 0:
                        print(f"❌ Failed: {base_model} cmd={p.cmd}")
                    else:
                        print(f"✅ Finished: {base_model} cmd={p.cmd}")
                i = 0
                processes = []     # 清空进程列表，开始下一批


    for p in processes:
        p.wait()     # 等待当前这批都结束
        if p.returncode != 0:
            print(f"❌ Failed: {base_model} cmd={p.cmd}")
        else:
            print(f"✅ Finished: {base_model} cmd={p.cmd}")
    processes = [] 


def test_only_anchor_logits():
    i = 0
    # run_name="anchor-lr-$lr-batch-$effective_batch-beta-$beta"
    times = 6
    lr = 1e-5
    filter_factor_img_upper=1
    filter_factor_img_lower=0.999
    filter_factor_text_upper=1
    filter_factor_text_lower=0.999
    processes = []

    for time, use_anchor, num_train_epochs in itertools.product(range(times), [True],[1]):
    # for time in range(times):
            if i >= len(master_ports):
                raise ValueError(f"i={i} 超出 master_ports 长度 {len(master_ports)}")
            master_port = master_ports[i]
            i += 1
            # run_name=f"img_filter-upper-{filter_factor_img_upper}-lower-{filter_factor_img_lower}-lr-{lr}-batch-{effective_batch}-beta-{beta}-use_anchor-{use_anchor}-{time}"
            run_name=f"no_img_text-lr-{lr}-batch-{effective_batch}-beta-{beta}-epoch-{num_train_epochs}-{time}"
            pretrained=f"/home/yilin/yilin-DPO/output/{base_model}/{run_name}"

            cmd = [
                # f"""python -m debugpy --connect 5679 $(which deepspeed) --include=localhost:0 --master_port {master_port} train_rdpo.py \
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
                --use_anchor {use_anchor} \
                --num_train_epochs {num_train_epochs} \
                --filter_factor_img_upper {filter_factor_img_upper} \
                --filter_factor_img_lower {filter_factor_img_lower} \
                --filter_factor_text_upper {filter_factor_text_upper} \
                --filter_factor_text_lower {filter_factor_text_lower} """
            ]
            print(f"🚀 Running {run_name}")
            p = subprocess.Popen(cmd, shell=True)
            p.cmd = cmd
            
            processes.append(p)
            print(f"当前进程数: {len(processes)}")
                # 如果已经启动了 3 个，就等待它们跑完
            if len(processes) == 3:
                for p in processes:
                    p.wait()     # 等待当前这批都结束
                    if p.returncode != 0:
                        print(f"❌ Failed: {base_model} cmd={p.cmd}")
                    else:
                        print(f"✅ Finished: {base_model} cmd={p.cmd}")
                i = 0
                processes = []     # 清空进程列表，开始下一批


    for p in processes:
        p.wait()     # 等待当前这批都结束
        if p.returncode != 0:
            print(f"❌ Failed: {base_model} cmd={p.cmd}")
        else:
            print(f"✅ Finished: {base_model} cmd={p.cmd}")
    processes = [] 


def test_mm_dpo():
    i = 0
    use_anchor=False
    data_path="./dataset/mmrlhf/no-mcq-CN-converted-dpo_pairs.json"
    # data_path="./dataset/mmrlhf/no-CN-converted-dpo_pairs.json"
    times = 1
    effective_batch = 12
    lrs = [5e-5, 1e-6, 5e-7]
    processes = []
    master_ports=[60010,60011,60012]
    for time, lr, use_anchor in itertools.product(range(times), lrs, [False]):
    # for time in range(times):
            if i >= len(master_ports):
                raise ValueError(f"i={i} 超出 master_ports 长度 {len(master_ports)}")
            master_port = master_ports[i]
            i += 1
            run_name=f"no-CN-mcq-lr-{lr}-batch-{effective_batch}-beta-{beta}-use_anchor-{use_anchor}-{time}"
            pretrained=f"/home/yilin/yilin-DPO/output/{base_model}/{run_name}"

            cmd = [
                # f"""python -m debugpy --connect 5679 $(which deepspeed) --include=localhost:0 --master_port {master_port} train_rdpo.py \
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
                --use_anchor {use_anchor} """
            ]
                # --max_steps 3 \
            print(f"🚀 Running {run_name}")
            p = subprocess.Popen(cmd, shell=True)
            p.cmd = cmd
            
            processes.append(p)
            print(f"当前进程数: {len(processes)}")
                # 如果已经启动了 3 个，就等待它们跑完
            if len(processes) == 2:
                for p in processes:
                    p.wait()     # 等待当前这批都结束
                    if p.returncode != 0:
                        print(f"❌ Failed: {base_model} cmd={p.cmd}")
                    else:
                        print(f"✅ Finished: {base_model} cmd={p.cmd}")
                i = 0
                processes = []     # 清空进程列表，开始下一批



def test_mdpo_silkie_dpo():
    i = 0
    use_anchor=False
    data_path="./dataset/silkie/converted_vlfeedback_llava_10k.json"
    image_folder="/home/yilin/yilin-DPO/dataset/silkie/merged_images/"
    times = 1
    num_train_epochs=3
    effective_batch = 12
    lrs = [1e-6]
    processes = []
    master_ports=[60110,60011,60012]
    for time, lr, use_anchor in itertools.product(range(times), lrs, [False]):
    # for time in range(times):
            if i >= len(master_ports):
                raise ValueError(f"i={i} 超出 master_ports 长度 {len(master_ports)}")
            master_port = master_ports[i]
            i += 1
            run_name=f"mdpo-silkie-lr-{lr}-batch-{effective_batch}-beta-{beta}-use_anchor-{use_anchor}-epoch-{num_train_epochs}-{time}"
            pretrained=f"/home/yilin/yilin-DPO/output/{base_model}/{run_name}"

            cmd = [
                # f"""python -m debugpy --connect 5679 $(which deepspeed) --include=localhost:0 --master_port {master_port} train_rdpo.py \
                f"""deepspeed --include=localhost:0  --master_port {master_port} train_rdpo.py \
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
                --num_train_epochs {num_train_epochs} \
                --project_name "yilin-align" \
                --use_anchor {use_anchor} """
            ]
            # --num_train_epochs 2
                # --max_steps 3 \
            print(f"🚀 Running {run_name}")
            p = subprocess.Popen(cmd, shell=True)
            p.cmd = cmd
            
            processes.append(p)
            print(f"当前进程数: {len(processes)}")
                # 如果已经启动了 3 个，就等待它们跑完
            if len(processes) == 1:
                for p in processes:
                    p.wait()     # 等待当前这批都结束
                    if p.returncode != 0:
                        print(f"❌ Failed: {base_model} cmd={p.cmd}")
                    else:
                        print(f"✅ Finished: {base_model} cmd={p.cmd}")
                i = 0
                processes = []     # 清空进程列表，开始下一批

def test_masked_re_align():
    i = 0
    use_anchor=False
    data_path="./preference_data/top2_masked_pooler.json"
    image_folder="/home/yilin/dataset/train2014/"
    times = 4
    num_train_epochs=1
    effective_batch = 12
    lrs = [1e-5]
    processes = []
    master_ports=[60110,60011,60012]
    for time, lr, use_anchor in itertools.product(range(times), lrs, [True]):
    # for time in range(times):
            if i >= len(master_ports):
                raise ValueError(f"i={i} 超出 master_ports 长度 {len(master_ports)}")
            master_port = master_ports[i]
            i += 1
            run_name=f"masked_re-lr-{lr}-batch-{effective_batch}-beta-{beta}-use_anchor-{use_anchor}-{time}"
            pretrained=f"/home/yilin/yilin-DPO/output/{base_model}/{run_name}"

            cmd = [
                # f"""python -m debugpy --connect 5679 $(which deepspeed) --include=localhost:0 --master_port {master_port} train_rdpo.py \
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
                --num_train_epochs {num_train_epochs} \
                --project_name "yilin-align" \
                --use_mask_loss True \
                --use_anchor {use_anchor} """
            ]
            # --num_train_epochs 2
                # --max_steps 3 \
            print(f"🚀 Running {run_name}")
            p = subprocess.Popen(cmd, shell=True)
            p.cmd = cmd
            
            processes.append(p)
            print(f"当前进程数: {len(processes)}")
                # 如果已经启动了 3 个，就等待它们跑完
            if len(processes) == 3:
                for p in processes:
                    p.wait()     # 等待当前这批都结束
                    if p.returncode != 0:
                        print(f"❌ Failed: {base_model} cmd={p.cmd}")
                    else:
                        print(f"✅ Finished: {base_model} cmd={p.cmd}")
                i = 0
                processes = []     # 清空进程列表，开始下一批



if __name__ == "__main__":
    # test_filter_img()
    # test_only_anchor_logits()
    # test_mm_dpo()
    # test_masked_re_align()
    test_filter()
    # test_re_align()