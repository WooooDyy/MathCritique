dataset_name="lighteval/MATH openai/gsm8k"
dataset_type="math gsm8k"
sample_num=1
temperature=0.7
reserved_new_data=1
model_name="meta-llama/Meta-Llama-3-8B"
model_type="Base"
template="default" # llama3 default
ITER_NUM=1
EXP_NUM="test"
actor_model_name=${model_name}
USE_CRITIC=1
USE_ORIGINAL=1
USE_PREVIOUS=0
USE_SELFIMPROVE_FEEDBACK=0
USE_GPT4_FEEDBACK=1
actor_model_name="meta-llama/Meta-Llama-3-8B"
critic_model_name="meta-llama/Meta-Llama-3-8B-Instruct"
workdir=$(pwd)

for ((ITER = 0; ITER <= ITER_NUM; ITER++))
do
    cd ${workdir}/selfimprove
    if [ "${ITER}" -ge 1 ]; then

        echo "Sampling ${ITER} process."

        python inference.py \
                --actor_name ${actor_model_name} \
                --model_name ${model_name} \
                --dataset_name ${dataset_name} \
                --dataset_type ${dataset_type} \
                --temperature ${temperature} \
                --sample_num ${sample_num} \
                --results_file ${workdir}/selfimprove/${model_name}-all-selfimprove${EXP_NUM}_${ITER}_origin.json \
                --mode 'inference' 
        sleep 5

        if [ "${USE_CRITIC}" -eq 1 ]; then
            python critic.py \
                    --critic_name ${critic_model_name} \
                    --temperature 0 \
                    --sample_num 1   
            sleep 5

            python inference.py \
                    --actor_name ${actor_model_name} \
                    --model_name ${model_name} \
                    --dataset_name ${dataset_name} \
                    --dataset_type ${dataset_type} \
                    --temperature ${temperature} \
                    --sample_num ${sample_num} \
                    --results_file ${workdir}/selfimprove/${model_name}-all-selfimprove${EXP_NUM}_${ITER}_new.json \
                    --reserved_new_data ${reserved_new_data} \
                    --mode 'new'
        fi

        echo "Sampling ${ITER} finished."
    fi

    if [ "${USE_CRITIC}" -eq 1 ] && [ "${ITER}" -ne 0 ]; then
        sleep 1
        python data_filter.py \
            --model_type ${model_type} \
            --model_name ${model_name} \
            --inference_file ${workdir}/selfimprove/${model_name}-all-selfimprove${EXP_NUM}_${ITER}_new.json \
            --results_file ${workdir}/selfimprove/${model_name}-all-sftdata_${ITER}_new.json \
            --use_selfimprove_feedback ${USE_SELFIMPROVE_FEEDBACK} \
            --use_gpt4_feedback ${USE_GPT4_FEEDBACK} \
            --iter ${ITER}
    fi

    sleep 1
    python data_filter.py \
        --model_type ${model_type} \
        --model_name ${model_name} \
        --inference_file ${workdir}/selfimprove/${model_name}-all-selfimprove${EXP_NUM}_${ITER}_origin.json \
        --results_file ${workdir}/selfimprove/${model_name}-all-sftdata_${ITER}_origin.json \
        --use_critic_data ${USE_CRITIC} \
        --use_original_data ${USE_ORIGINAL} \
        --use_previous_data ${USE_PREVIOUS} \
        --sft_file ${workdir}/selfimprove/${model_name}-all-sftdata.json \
        --use_selfimprove_feedback ${USE_SELFIMPROVE_FEEDBACK} \
        --use_gpt4_feedback ${USE_GPT4_FEEDBACK} \
        --iter ${ITER}

    sleep 3
    cp "${workdir}/selfimprove/${model_name}-all-sftdata.json" "${workdir}/LLaMA-Factory/data/train_sft_${model_type}.json"
    sleep 3

    if [ "${ITER}" -ge 1 ]; then
        base_model_name="meta-llama/Meta-Llama-3-8B"
    else
        base_model_name="meta-llama/Meta-Llama-3-8B"
    fi

    cd ${workdir}/LLaMA-Factory
    echo "Actor SFT ${ITER} started."
    llamafactory-cli train \
        --stage sft \
        --do_train True \
        --model_name_or_path ${base_model_name} \
        --preprocessing_num_workers 16 \
        --finetuning_type full \
        --template ${template} \
        --flash_attn fa2 \
        --dataset_dir data \
        --dataset train_sft_${model_type} \
        --cutoff_len 4096 \
        --learning_rate 5e-06 \
        --num_train_epochs 1.0 \
        --max_samples 400000 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --lr_scheduler_type cosine \
        --max_grad_norm 1.0 \
        --logging_steps 5 \
        --save_steps 5000 \
        --warmup_steps 0 \
        --optim adamw_torch \
        --packing False \
        --report_to none \
        --output_dir ${workdir}/models/${model_name}-all-selfimprove${EXP_NUM}_${USE_CRITIC}_${ITER} \
        --bf16 True \
        --plot_loss True \
        --ddp_timeout 180000000 \
        --include_num_input_tokens_seen True \
        --deepspeed examples/deepspeed/ds_z2_config.json  \
        --save_only_model
    
    echo "Actor SFT ${ITER} finished."
    sleep 5
    actor_model_name="${workdir}/models/${model_name}-all-selfimprove${EXP_NUM}_${USE_CRITIC}_${ITER}"
done
cd ${workdir}/selfimprove
source evaluate-all.sh
