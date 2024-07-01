# llama & its sibling architectures only
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig
torch.set_default_dtype(torch.float16)

#model_name = "NousResearch/Meta-Llama-3-8B"
model_name = "NousResearch/Meta-Llama-3-70B"

cfg = AutoConfig.from_pretrained(model_name)

architectures            = cfg.architectures
hidden_size              = cfg.hidden_size
intermediate_size        = cfg.intermediate_size
num_attention_heads      = cfg.num_attention_heads
num_hidden_layers        = cfg.num_hidden_layers
num_key_value_heads      = cfg.num_key_value_heads
vocab_size               = cfg.vocab_size

if "LlamaForCausalLM" in architectures:
    pass
else:
    raise ValueError("Unsuported model")

mode = "PREFILL"
mode = "DECODING"

sl = 2048 if mode == "PREFILL" else 1
tp = 8
iters = 10
warmup = 5
data_in_byte = 2 #fp16, bf16

bs_list = [
        1,
        8,
        16,
        32,
        64,
        ]

module_list = [
            "emb",
            "qkv_proj",
            "o_proj",
            "gate_up_proj",
            "down_proj",
            ]

print("[INFO] torch:        ",torch.__version__)
print("[INFO] cuda/rocm:    ", torch._C._cuda_getCompiledVersion())
print("[INFO] device:       ", torch.cuda.get_device_name(0))

def row_parallel_K(x, w, tp):
    return x/tp, w

def col_parallel_K(x, w, tp):
    return x, w/tp

for bs in bs_list:
    print("batch size: ", bs)
    print("module:x.shape:wT.shape:tflops:arith_intensity")
    for module in module_list:
        latency_set = []
        x0 = bs * sl
        if module == "emb": # Col
            x1 = hidden_size
            w1 = vocab_size 
            x1, w1 = col_parallel_K (x1, w1, tp)
        elif module == "qkv_proj": # Col
            x1 = hidden_size
            w1 = (hidden_size/num_attention_heads)*(num_attention_heads + 2 * num_key_value_heads) 
            x1, w1 = col_parallel_K (x1, w1, tp)
        elif module == "o_proj": # Row
            x1 = hidden_size 
            w1 = hidden_size 
            x1, w1 = row_parallel_K (x1, w1, tp)
        elif module == "gate_up_proj": #Col
            x1 = hidden_size
            w1 = 2 * intermediate_size 
            x1, w1 = col_parallel_K (x1, w1, tp)
        elif module == "down_proj": # Row
            x1 = intermediate_size 
            w1 = hidden_size
            x1, w1 = row_parallel_K (x1, w1, tp)
        x1, w1 = int(x1), int(w1)
        w0 = x1
        x  = torch.randn(x0, x1, device="cuda:0", dtype=torch.float16)
        wT = torch.randn(w1, w0, device="cuda:0", dtype=torch.float16)

        with torch.no_grad():
            for itr in range(iters):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start_event.record()

                out = F.linear(x, wT)

                end_event.record()
                torch.cuda.synchronize()
                time = start_event.elapsed_time(end_event)
                latency_set.append(time)

        latency_set = latency_set[warmup:]
        latency_avg = sum(latency_set) / len(latency_set)
        tflops = x0 * x1 * w1 * 2 /1e9/(latency_avg) 
        ddr_access = data_in_byte * (x0 * x1 + w0 * w1 + x0 * w1)

        arith_intensity = (x0 * x1 * w1 * 2) / ddr_access 
        print("{}:{}:{}:{}:{}".format(module, x.shape, wT.shape, tflops, arith_intensity))
