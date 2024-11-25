# 设置变量
dataset_name="lighteval/MATH openai/gsm8k"
dataset_type="math gsm8k"
sample_num=1
reserved_new_data=1
temperature=0
model_name="meta-llama/Meta-Llama-3-8B"
model_type="Base"
ITER_NUM=1
MV_NUM=1
ONLY_FINAL_SEQUENTIAL=0
EXP_NUM="test"
actor_model_name=${model_name}
USE_CRITIC=1
TEST_USE_CRITIC=1
TEST_KNOW_ANSWER=0
critic_model_name="meta-llama/Meta-Llama-3-8B-Instruct"

for ((ITER = 0; ITER <= ITER_NUM; ITER++))
do
    results_file="${workdir}/selfimprove/${model_name}-all-selfimprove${EXP_NUM}_${ITER}_test_${sample_num}-${MV_NUM}_${TEST_USE_CRITIC}${TEST_KNOW_ANSWER}.json"
    cd ${workdir}/selfimprove
    actor_model_name="${workdir}/models/${model_name}-all-selfimprove${EXP_NUM}_${USE_CRITIC}_${ITER}"
    
    echo "Evaluating start."

    python inference.py \
        --actor_name ${actor_model_name} \
        --model_name ${model_name} \
        --dataset_name ${dataset_name} \
        --dataset_type ${dataset_type} \
        --temperature ${temperature} \
        --sample_num ${sample_num} \
        --results_file ${results_file} \
        --mode 'test' \
        --test_know_answer ${TEST_KNOW_ANSWER} \
        --need_false_data 0

    for ((MV = 1; MV <= MV_NUM; MV++))
    do
        echo "MV ${MV} started"
        if [ "${TEST_USE_CRITIC}" -eq 1 ]; then
            sleep 5

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
                    --temperature 0 \
                    --sample_num 1 \
                    --results_file ${results_file} \
                    --mode 'new' \
                    --reserved_new_data ${reserved_new_data} \
                    --need_false_data 1
            
            sleep 5
            python test_filter.py \
                --mode 'sequential'
        fi
    done
    
    if [ "${ONLY_FINAL_SEQUENTIAL}" -eq 1 ]; then
        python test_filter.py \
            --results_file ${results_file} \
            --mode 'only_final_sequential'
    fi

    echo "Evaluating finished."
    sleep 5
    echo ${results_file}

    python test_filter.py \
        --dataset_name ${dataset_name} \
        --dataset_type ${dataset_type} \
        --results_file ${results_file} \
        --mode "majority"
done

