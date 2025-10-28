#!/bin/bash
# 
# 条件付き分子生成スクリプト
# 
# 使用方法:
#   bash generate_all_conditions.sh [model_path]
# 
# 例:
#   bash generate_all_conditions.sh outputs/molecular_descriptor_model
#

# デフォルトのモデルパス
MODEL_PATH="${1:-outputs/molecular_descriptor_model}"

echo "========================================"
echo "条件付き分子生成スクリプト"
echo "========================================"
echo "モデルパス: ${MODEL_PATH}"
echo ""

# モデルディレクトリの存在確認
if [ ! -d "${MODEL_PATH}" ]; then
    echo "エラー: モデルディレクトリが見つかりません: ${MODEL_PATH}"
    echo "正しいパスを指定してください。"
    exit 1
fi

# 出力ディレクトリの作成
OUTPUT_DIR="${MODEL_PATH}/generated_samples_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUTPUT_DIR}"

echo "出力ディレクトリ: ${OUTPUT_DIR}"
echo ""

# 分子量と共役性の組み合わせ
MOLECULAR_WEIGHTS=(100 200)
PI_CONJUGATIONS=(0.8 0.9 1.0)

# 共通パラメータ
ATOM_TYPES="[C,H,O,N]"
FUNCTIONAL_GROUPS="[OH,COOH]"
N_SWEEPS=5

echo "生成条件:"
echo "  原子種: ${ATOM_TYPES}"
echo "  官能基: ${FUNCTIONAL_GROUPS}"
echo "  分子量: ${MOLECULAR_WEIGHTS[@]}"
echo "  π共役性: ${PI_CONJUGATIONS[@]}"
echo "  サンプル数/条件: ${N_SWEEPS}"
echo ""

# ログファイル
LOG_FILE="${OUTPUT_DIR}/generation_log.txt"
echo "ログファイル: ${LOG_FILE}"
echo "========================================"
echo ""

# 生成開始時刻
START_TIME=$(date +%s)

# カウンター
TOTAL_COMBINATIONS=$((${#MOLECULAR_WEIGHTS[@]} * ${#PI_CONJUGATIONS[@]}))
CURRENT=0

# すべての組み合わせで分子を生成
for MW in "${MOLECULAR_WEIGHTS[@]}"; do
    for PC in "${PI_CONJUGATIONS[@]}"; do
        CURRENT=$((CURRENT + 1))
        
        echo "[$CURRENT/$TOTAL_COMBINATIONS] 分子生成中..."
        echo "  分子量: ${MW}"
        echo "  π共役性: ${PC}"
        
        # プロパティ値の構築
        PROPERTY_VALUES="molecular_weight=${MW},pi_conjugation_ratio=${PC},atom_types_encoding=${ATOM_TYPES},functional_groups_encoding=${FUNCTIONAL_GROUPS}"
        
        # 出力サブディレクトリ
        SUBDIR="${OUTPUT_DIR}/mw${MW}_pc${PC}"
        mkdir -p "${SUBDIR}"
        
        # 生成コマンドの実行
        echo "コマンド: python eval_conditional_qm9.py --generators_path ${MODEL_PATH} --task qualitative --use_exact_conditions --property_values '${PROPERTY_VALUES}' --n_sweeps ${N_SWEEPS}" >> "${LOG_FILE}"
        echo "開始時刻: $(date)" >> "${LOG_FILE}"
        
        python eval_conditional_qm9.py \
            --generators_path "${MODEL_PATH}" \
            --task qualitative \
            --use_exact_conditions \
            --property_values "${PROPERTY_VALUES}" \
            --n_sweeps ${N_SWEEPS} \
            2>&1 | tee -a "${LOG_FILE}"
        
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -eq 0 ]; then
            echo "  ✓ 成功" | tee -a "${LOG_FILE}"
        else
            echo "  ✗ 失敗 (終了コード: ${EXIT_CODE})" | tee -a "${LOG_FILE}"
        fi
        
        echo "終了時刻: $(date)" >> "${LOG_FILE}"
        echo "----------------------------------------" >> "${LOG_FILE}"
        echo ""
    done
done

# 生成終了時刻
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo "========================================"
echo "全ての生成が完了しました"
echo "========================================"
echo "実行時間: ${MINUTES}分${SECONDS}秒"
echo "出力ディレクトリ: ${OUTPUT_DIR}"
echo "ログファイル: ${LOG_FILE}"
echo ""
echo "生成された分子の確認:"
echo "  ls ${OUTPUT_DIR}"
echo ""

# サマリーの作成
SUMMARY_FILE="${OUTPUT_DIR}/summary.txt"
cat > "${SUMMARY_FILE}" << EOF
===========================================
条件付き分子生成サマリー
===========================================

実行日時: $(date)
モデルパス: ${MODEL_PATH}
出力ディレクトリ: ${OUTPUT_DIR}

生成条件:
  原子種: ${ATOM_TYPES}
  官能基: ${FUNCTIONAL_GROUPS}
  分子量: ${MOLECULAR_WEIGHTS[@]}
  π共役性: ${PI_CONJUGATIONS[@]}
  サンプル数/条件: ${N_SWEEPS}

総組み合わせ数: ${TOTAL_COMBINATIONS}
実行時間: ${MINUTES}分${SECONDS}秒

詳細ログ: ${LOG_FILE}
===========================================
EOF

echo "サマリーファイル: ${SUMMARY_FILE}"
cat "${SUMMARY_FILE}"

echo ""
echo "次のステップ:"
echo "  1. 生成された分子の可視化:"
echo "     python eval_sample.py --model_path ${MODEL_PATH}"
echo ""
echo "  2. 分子の評価と解析:"
echo "     python eval_analyze.py --model_path ${MODEL_PATH} --n_samples 1000"
echo ""
