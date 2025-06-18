#!/bin/bash

NGPUS=8
TASK=t2v-14B
SIZE=1280*720
MODEL_PATH="/data00/models/Wan2.1-T2V-14B"
IMAGE_PATH="./i2v_input.jpg"
PROMPT="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."

# sequence parallel options
ULYSSES_SIZE=8
RING_SIZE=1

# avaiable options
opt_fsdp=0
opt_fa3=0
opt_sage_attn=1

opt_teacache=1
teacache_thresh=0.2

cmd=(
	torchrun
	--nproc_per_node="$NGPUS"
	generate.py
	--task "$TASK"
	--size "$SIZE"
	--ckpt_dir "$MODEL_PATH"
	--ulysses_size "$ULYSSES_SIZE"
	--ring_size "$RING_SIZE"
	--prompt "${PROMPT}"
)

if [[ "${TASK}" == *"i2v"* ]]; then
	cmd+=(--image $IMAGE_PATH)
fi

if (( opt_fsdp == 1 )); then
	cmd+=(--t5_fsdp)
	cmd+=(--dit_fsdp)
fi
 
if (( opt_fa3 == 1 )); then
	cmd+=(--enable-fa3)
fi

if (( opt_sage_attn == 1 )); then
	cmd+=(--enable-sage-attn)
fi

if (( opt_teacache == 1 )); then
	cmd+=(--enable_teacache)
	cmd+=(--teacache_thresh $teacache_thresh)
fi

print_quoted_cmd() {
    local arg quoted=()
    for arg in "$@"; do
        # 如果参数包含空格、制表符、引号或换行，则添加引号
        if [[ "$arg" =~ [[:space:]\'\"] ]]; then
            # 转义内部引号并添加外部单引号
            quoted+=("'${arg//\'/\'\"\'\"\'}'")
        else
            quoted+=("$arg")
        fi
    done
    printf "%s\n" "${quoted[*]}"
}

print_quoted_cmd "${cmd[@]}"
exec "${cmd[@]}"
