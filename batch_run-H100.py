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
# run_name=f"dpo-llava-v1.5-7b-lr-$lr-acc_batch-$effective_batch-beta-{beta}"


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# yilin's loss

use_text_similarity=False
ls_factor_text_weight=0.5
use_img_similarity=False
ls_factor_img_weight=0.5
# run_name=f"reverse-pooler_output-lr-{lr}-acc_batch-$effective_batch-text_weight-{ls_factor_text_weight}-img_weight-{ls_factor_img_weight}"

# beta_dpo setting

beta_dpo=False
ls_factor_weight=0.1
# run_name="beta_dpo-lr-$lr-acc_batch-$effective_batch-beta-$beta-ls_factor_weight-$ls_factor_weight"

# anchor
use_anchor=False
# run_name="anchor-lr-$lr-acc_batch-$effective_batch-beta-$beta"

master_ports=[60010,60011,60012]

# data_path="./preference_data/yilin_pref_data_last_hidden_state.json"
data_path="./preference_data/yilin_pref_data_pooler_output.json"

image_folder="/home/yilin/dataset/train2014/"


max_parallel=3


def test_anchor():
    i = 0
    use_anchor=True
    # run_name="anchor-lr-$lr-acc_batch-$effective_batch-beta-$beta"
    times = 6
    lr = 1e-5
    processes = []
    # for ls_factor_text_weight in ls_factor_text_weights:
    for time in range(times):
            if i >= len(master_ports):
                raise ValueError(f"i={i} 超出 master_ports 长度 {len(master_ports)}")
            master_port = master_ports[i]
            i += 1
            run_name=f"anchor-lr-{lr}-acc_batch-{effective_batch}-beta-{beta}-{time}-{base_model}"
            pretrained=f"/home/yilin/Re-Align/output/{base_model}/{run_name}"

            cmd = [
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
                --ls_factor_img_weight {ls_factor_text_weight} \
                --beta_dpo {beta_dpo} \
                --ls_factor_weight {ls_factor_weight} \
                --use_anchor {use_anchor}"""
            ]
            print(f"🚀 Running {base_model} | lr={lr}, bs={effective_batch}")
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




def test_yilin_align():
    i = 0
    times = 4
    lr=1e-5
    use_text_similarity=False
    ls_factor_text_weight=0.3
    use_img_similarity=True
    ls_factor_img_weight=0.3
    processes = []
    # for ls_factor_text_weight in ls_factor_text_weights:
    for time in range(times):
            master_port = master_ports[i]
            i += 1
            ls_factor_img_weight = ls_factor_text_weight
            run_name=f"reverse-pooler_output-lr-{lr}-acc_batch-{effective_batch}-img_weight-{ls_factor_img_weight}-{time}"
            pretrained=f"/home/yilin/Re-Align/output/{base_model}/{run_name}"
            cmd = [
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
                --ls_factor_img_weight {ls_factor_text_weight} \
                --beta_dpo {beta_dpo} \
                --ls_factor_weight {ls_factor_weight} \
                --use_anchor {use_anchor}"""
            ]

            print(f"🚀 Running {base_model} | lr={lr}, bs={effective_batch}")
            p = subprocess.Popen(cmd, shell=True)
            p.cmd = cmd
            
            processes.append(p)

                # 如果已经启动了 max_parallel 个，就等待它们跑完
            if len(processes) == max_parallel:
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



def test_re_align():
    i = 0
    times = 10
    lrs = [3.333e-6]
    betas = [0.1]
    processes = []
    seeds = [3522768589, 4154981737, 953889115 ,1898759133,203703611,3430150436,3301789110,1967766348,4184740135,1965031076]
    # for ls_factor_text_weight in ls_factor_text_weights:
    for time, lr, beta in itertools.product(range(times), lrs, betas):
    # for time in range(times):
        
        seed = seeds[i]
        if i >= len(master_ports):
            raise ValueError(f"i={i} 超出 master_ports 长度 {len(master_ports)}")
        master_port = master_ports[i]
        i += 1
        run_name=f"re_align-lr-{lr}-acc_batch-{effective_batch}-beta-{beta}-{time}"
        pretrained=f"/home/yilin/Re-Align/output/{base_model}/{run_name}"

        cmd = [
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
            --ls_factor_img_weight {ls_factor_text_weight} \
            --beta_dpo {beta_dpo} \
            --ls_factor_weight {ls_factor_weight} \
            --use_anchor {use_anchor} \
            --seed {seed} \
            --random_seed False """
        ]
        print(f"🚀 Running {base_model} | lr={lr}, bs={effective_batch}")
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


    for p in processes:
        p.wait()     # 等待当前这批都结束
        if p.returncode != 0:
            print(f"❌ Failed: {base_model} cmd={p.cmd}")
        else:
            print(f"✅ Finished: {base_model} cmd={p.cmd}")
    processes = [] 

def test_anchor_beta():
    i = 0
    use_anchor=True
    # run_name="anchor-lr-$lr-acc_batch-$effective_batch-beta-$beta"
    times = 8
    lrs = [1e-5]
    anchor_betas=[0.1]
    processes = []
    for time, anchor_beta, lr in itertools.product(range(times), anchor_betas, lrs):
    # for time in range(times):
            if i >= len(master_ports):
                raise ValueError(f"i={i} 超出 master_ports 长度 {len(master_ports)}")
            master_port = master_ports[i]
            i += 1
            run_name=f"rejected_anchor_beta-lr-{lr}-acc_batch-{effective_batch}-beta-{beta}-anchor_betas-{anchor_beta}-{time}"
            pretrained=f"/home/yilin/Re-Align/output/{base_model}/{run_name}"

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
                --ls_factor_weight {ls_factor_weight} \
                --use_anchor {use_anchor} \
                --anchor_beta {anchor_beta} """
            ]
            print(f"🚀 Running {base_model} | lr={lr}, bs={effective_batch}")
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


    for p in processes:
        p.wait()     # 等待当前这批都结束
        if p.returncode != 0:
            print(f"❌ Failed: {base_model} cmd={p.cmd}")
        else:
            print(f"✅ Finished: {base_model} cmd={p.cmd}")
    processes = [] 




def test_yilin_anchor():
    i = 0
    yilin_anchor=True
    ls_factor_weight=1
    # run_name="anchor-lr-$lr-acc_batch-$effective_batch-beta-$beta"
    times = 6
    lr = 1e-5
    processes = []
    # for ls_factor_text_weight in ls_factor_text_weights:
    for time in range(times):
            if i >= len(master_ports):
                raise ValueError(f"i={i} 超出 master_ports 长度 {len(master_ports)}")
            master_port = master_ports[i]
            i += 1
            run_name=f"yilin_anchor_exp-lr-{lr}-acc_batch-{effective_batch}-beta-{beta}-{time}"
            pretrained=f"/home/yilin/Re-Align/output/{base_model}/{run_name}"

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
                --use_text_similarity {use_text_similarity}  \
                --ls_factor_text_weight {ls_factor_text_weight} \
                --use_img_similarity {use_img_similarity} \
                --ls_factor_img_weight {ls_factor_text_weight} \
                --beta_dpo {beta_dpo} \
                --ls_factor_weight {ls_factor_weight} \
                --yilin_anchor {yilin_anchor}"""
            ]
            print(f"🚀 Running {base_model} | lr={lr}, bs={effective_batch}")
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



def test_filter_text():
    i = 0
    # run_name="anchor-lr-$lr-acc_batch-$effective_batch-beta-$beta"
    times = 6
    lr = 1e-5
    filter_factor_text_upper=1
    filter_factor_text_lower=0
    processes = []

    for time, filter_factor_text_lower, use_anchor in itertools.product(range(0, times),
                                                [0.8,0.9], [True]):
    # for time in range(times):
            if i >= len(master_ports):
                raise ValueError(f"i={i} 超出 master_ports 长度 {len(master_ports)}")
            master_port = master_ports[i]
            i += 1
            run_name=f"text_filter-upper-{filter_factor_text_upper}-lower-{filter_factor_text_lower}-lr-{lr}-acc_batch-{effective_batch}-beta-{beta}-use_anchor-{use_anchor}-{time}"
            pretrained=f"/home/yilin/Re-Align/output/{base_model}/{run_name}"

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
                --filter_factor_text_lower {filter_factor_text_lower} \
                --filter_factor_text_upper {filter_factor_text_upper} """
            ]
            print(f"🚀 Running {base_model} | lr={lr}, bs={effective_batch}")
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



def test_filter_img():
    i = 0
    # run_name="anchor-lr-$lr-acc_batch-$effective_batch-beta-$beta"
    times = 5
    lr = 1e-5
    filter_factor_img_upper=1
    filter_factor_img_lower=0
    processes = []

    for time, filter_factor_img_lower, use_anchor in itertools.product(range(times),
                                                [0.9999], [True, False]):
    # for time in range(times):
            if i >= len(master_ports):
                raise ValueError(f"i={i} 超出 master_ports 长度 {len(master_ports)}")
            master_port = master_ports[i]
            i += 1
            # run_name=f"img_filter-upper-{filter_factor_img_upper}-lower-{filter_factor_img_lower}-lr-{lr}-acc_batch-{effective_batch}-beta-{beta}-use_anchor-{use_anchor}-{time}"
            run_name=f"no_img-{filter_factor_img_upper}-lower-{filter_factor_img_lower}-lr-{lr}-acc_batch-{effective_batch}-beta-{beta}-use_anchor-{use_anchor}-{time}"
            pretrained=f"/home/yilin/Re-Align/output/{base_model}/{run_name}"

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
                --filter_factor_img_upper {filter_factor_img_upper} \
                --filter_factor_img_lower {filter_factor_img_lower} """
            ]
            print(f"🚀 Running {base_model} | lr={lr}, bs={effective_batch}")
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



def test_dpop():
    i = 0
    # run_name="anchor-lr-$lr-acc_batch-$effective_batch-beta-$beta"
    times = 5
    lr = 1e-5
    processes = []

    for time, use_dpop_text, use_anchor, dpop_text_lambda in itertools.product(range(times),
                                                [True], [True, False], [5,50,500]):
    # for time in range(times):
            if i >= len(master_ports):
                raise ValueError(f"i={i} 超出 master_ports 长度 {len(master_ports)}")
            master_port = master_ports[i]
            i += 1
            # run_name=f"img_filter-upper-{filter_factor_img_upper}-lower-{filter_factor_img_lower}-lr-{lr}-acc_batch-{effective_batch}-beta-{beta}-use_anchor-{use_anchor}-{time}"
            run_name=f"use_dpop_text-{use_dpop_text}-lambda-{dpop_text_lambda}-use_anchor-{use_anchor}-lr-{lr}-acc_batch-{effective_batch}-beta-{beta}-{time}"
            pretrained=f"/home/yilin/Re-Align/output/{base_model}/{run_name}"

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
                --use_dpop_text {use_dpop_text} \
                --dpop_text_lambda {dpop_text_lambda} """
            ]
            print(f"🚀 Running {base_model} | lr={lr}, bs={effective_batch}")
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



def test_only_anchor_logits():
    i = 0
    # run_name="anchor-lr-$lr-acc_batch-$effective_batch-beta-$beta"
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
            # run_name=f"img_filter-upper-{filter_factor_img_upper}-lower-{filter_factor_img_lower}-lr-{lr}-acc_batch-{effective_batch}-beta-{beta}-use_anchor-{use_anchor}-{time}"
            run_name=f"no_img_text-lr-{lr}-batch-{effective_batch}-beta-{beta}-epoch-{num_train_epochs}-{time}"
            pretrained=f"/home/yilin/Re-Align/output/{base_model}/{run_name}"

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
            print(f"🚀 Running {base_model} | lr={lr}, bs={effective_batch}")
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


def test_yilin_data():
    i = 0
    use_anchor=False
    data_path="./preference_data/new_yilin_pooler_pref_data.json"
    times = 6
    lrs = [1e-5]
    processes = []
    master_ports=[60010,60011,60012]
    for time, lr in itertools.product(range(times), lrs):
    # for time in range(times):
            if i >= len(master_ports):
                raise ValueError(f"i={i} 超出 master_ports 长度 {len(master_ports)}")
            master_port = master_ports[i]
            i += 1
            run_name=f"full_yilin_gpt4o-lr-{lr}-acc_batch-{effective_batch}-beta-{beta}-use_anchor-{use_anchor}-{time}"
            pretrained=f"/home/yilin/Re-Align/output/{base_model}/{run_name}"

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
            print(f"🚀 Running {base_model} | lr={lr}, bs={effective_batch}")
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
    # test_yilin_anchor()
    # test_yilin_align()
    # test_re_align()
    # test_anchor_beta()
    # test_filter_text()
    # test_filter_img()
    # test_dpop()
    # test_only_anchor_logits()
    test_yilin_data()