#!/bin/bash
# 该方法能够一次性训练文件夹下的所有link，每个link一个模型
# --- 基础配置 ---
# 要使用的 GPU 编号
export CUDA_VISIBLE_DEVICES=0

# 模型名称
model_name=DLinear

DATA_ROOT_PATH="traffic/edge_test" 

# 2. 其他固定的训练参数
TASK_NAME="long_term_forecast"
IS_TRAINING=1
DATASET_TYPE="Traffic_Singlevariate"  # 这个值需要与你 Python 代码中定义的数据集类型匹配
FEATURES="M"
SEQ_LEN=10080
LABEL_LEN=0
PRED_LEN=1440
E_LAYERS=2
D_LAYERS=2
FACTOR=1
ENC_IN=17  # 根据你的数据特征数调整
DEC_IN=17  # 根据你的数据特征数调整
C_OUT=1   # 根据你的预测目标数调整
DES="test"
D_MODEL=512
D_FF=512
ITR=1
model_id=Traffic

# --- 遍历并训练 ---
echo "开始遍历目录: ${DATA_ROOT_PATH} 下的所有 CSV 文件..."

# 检查数据根目录是否存在
if [ ! -d "${DATA_ROOT_PATH}" ]; then
    echo "错误：数据根目录 ${DATA_ROOT_PATH} 不存在！"
    exit 1
fi

# for 循环遍历目录下的所有 .csv 文件
# 使用 find 命令确保能正确处理带有空格的文件名
find "${DATA_ROOT_PATH}" -maxdepth 1 -type f -name "*.csv" | while read -r csv_file_path; do
    
    # 获取文件名（不含路径）
    csv_file_name=$(basename "${csv_file_path}")
    
    # 获取文件名（不含扩展名），用于生成唯一的 model_id
    model_id_base=$(basename "${csv_file_name}" .csv)
    
    # 3. 【重要】动态生成唯一的 model_id
    #    格式例如：Traffic_id_1-2_iTransformer_96_0_8
    unique_model_id="${DATASET_TYPE}_${model_id_base}_${model_name}_${SEQ_LEN}_${LABEL_LEN}_${PRED_LEN}"
    
    echo "======================================================================"
    echo "正在处理文件: ${csv_file_name}"
    echo "生成的 model_id: ${unique_model_id}"
    echo "开始训练..."
    echo "======================================================================"

    # 4. 执行训练命令
    python -u traffic/Code/run.py \
      --task_name "${TASK_NAME}" \
      --is_training "${IS_TRAINING}" \
      --root_path "${DATA_ROOT_PATH}" \
      --data_path "${csv_file_name}" \
      --model_id "${model_id}" \
      --model "${model_name}" \
      --data "${DATASET_TYPE}" \
      --features "${FEATURES}" \
      --seq_len "${SEQ_LEN}" \
      --label_len "${LABEL_LEN}" \
      --pred_len "${PRED_LEN}" \
      --e_layers "${E_LAYERS}" \
      --d_layers "${D_LAYERS}" \
      --factor "${FACTOR}" \
      --enc_in "${ENC_IN}" \
      --dec_in "${DEC_IN}" \
      --c_out "${C_OUT}" \
      --des "${DES}" \
      --d_model "${D_MODEL}" \
      --d_ff "${D_FF}" \
      --itr "${ITR}"

    # 检查上一个命令是否执行成功
    if [ $? -eq 0 ]; then
        echo "文件 ${csv_file_name} 训练完成。"
    else
        echo "文件 ${csv_file_name} 训练失败！"
        # 如果希望某个文件训练失败后继续执行下一个，就注释掉下面的 exit 1
        # exit 1
    fi

done

echo "======================================================================"
echo "所有 CSV 文件的训练任务已全部提交。"
echo "======================================================================"