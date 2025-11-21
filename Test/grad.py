import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

device = "cuda"
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

# ✅ GPU 支持 BF16 → 用 BF16（不需要 GradScaler）
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,   # ✅ 参数直接存 BF16
).to(device)

# ✅ LoRA 注入
lora = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora)
model.train()

print("\n✅ Trainable parameters after LoRA injection:")
model.print_trainable_parameters()

# ✅ Dummy 文本数据
text = "TinyLLaMA is small but powerful."
inputs = tokenizer(text, return_tensors="pt").to(device)
labels = inputs["input_ids"].clone()

# ✅ 优化器 + 小学习率（避免 early blowing up）
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)

# ✅ 训练循环（无 GradScaler，无 FP16）
for step in range(5):
    optimizer.zero_grad(set_to_none=True)

    # ✅ 用 BF16 计算（比 FP16 更稳定）
    with torch.autocast(device_type="cuda", dtype=dtype):
        out = model(**inputs, labels=labels)
        loss = out.loss

    loss.backward()

    # ✅ 梯度裁剪（防止偶发爆炸）
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

    # ✅ 打印 LoRA 梯度（转 float32 后统计，避免 BF16 舍入）
    print(f"\nStep {step} | Loss = {loss.item():.6f}")
    for name, p in model.named_parameters():
        if "lora" in name.lower():
            g = "None" if p.grad is None else f"{p.grad.detach().float().abs().mean().item():.6g}"
            print(f"{name:60s} grad={g}")
