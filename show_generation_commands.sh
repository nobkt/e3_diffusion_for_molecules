#!/bin/bash
#
# 問題ステートメントに対する直接的な回答
# 
# 条件付き学習で、--conditioning molecular_weight pi_conjugation_ratio 
# atom_types_encoding functional_groups_encodingで学習させたのち、
# [C,H,O,N]を含み、分子量が100、200で、官能基にOH、COOHを含み、
# π共役性が0.8、0.9、1.0の条件で分子生成させる場合の生成コマンド
#

# 注意: 以下のコマンドを実行する前に、モデルパスを実際のパスに変更してください
MODEL_PATH="outputs/molecular_descriptor_model"

echo "=========================================="
echo "問題ステートメントに対する回答"
echo "=========================================="
echo ""
echo "以下のコマンドで、指定された条件の分子を生成できます："
echo ""
echo "条件："
echo "  - 原子種: C, H, O, N"
echo "  - 分子量: 100, 200"
echo "  - 官能基: OH, COOH"
echo "  - π共役性: 0.8, 0.9, 1.0"
echo ""
echo "=========================================="
echo ""

# すべての組み合わせのコマンドを表示
echo "# 分子量 100, π共役性 0.8"
echo "python eval_conditional_qm9.py \\"
echo "    --generators_path ${MODEL_PATH} \\"
echo "    --task qualitative \\"
echo "    --use_exact_conditions \\"
echo "    --property_values 'molecular_weight=100,pi_conjugation_ratio=0.8,atom_types_encoding=[C,H,O,N],functional_groups_encoding=[OH,COOH]' \\"
echo "    --n_sweeps 5"
echo ""

echo "# 分子量 100, π共役性 0.9"
echo "python eval_conditional_qm9.py \\"
echo "    --generators_path ${MODEL_PATH} \\"
echo "    --task qualitative \\"
echo "    --use_exact_conditions \\"
echo "    --property_values 'molecular_weight=100,pi_conjugation_ratio=0.9,atom_types_encoding=[C,H,O,N],functional_groups_encoding=[OH,COOH]' \\"
echo "    --n_sweeps 5"
echo ""

echo "# 分子量 100, π共役性 1.0"
echo "python eval_conditional_qm9.py \\"
echo "    --generators_path ${MODEL_PATH} \\"
echo "    --task qualitative \\"
echo "    --use_exact_conditions \\"
echo "    --property_values 'molecular_weight=100,pi_conjugation_ratio=1.0,atom_types_encoding=[C,H,O,N],functional_groups_encoding=[OH,COOH]' \\"
echo "    --n_sweeps 5"
echo ""

echo "# 分子量 200, π共役性 0.8"
echo "python eval_conditional_qm9.py \\"
echo "    --generators_path ${MODEL_PATH} \\"
echo "    --task qualitative \\"
echo "    --use_exact_conditions \\"
echo "    --property_values 'molecular_weight=200,pi_conjugation_ratio=0.8,atom_types_encoding=[C,H,O,N],functional_groups_encoding=[OH,COOH]' \\"
echo "    --n_sweeps 5"
echo ""

echo "# 分子量 200, π共役性 0.9"
echo "python eval_conditional_qm9.py \\"
echo "    --generators_path ${MODEL_PATH} \\"
echo "    --task qualitative \\"
echo "    --use_exact_conditions \\"
echo "    --property_values 'molecular_weight=200,pi_conjugation_ratio=0.9,atom_types_encoding=[C,H,O,N],functional_groups_encoding=[OH,COOH]' \\"
echo "    --n_sweeps 5"
echo ""

echo "# 分子量 200, π共役性 1.0"
echo "python eval_conditional_qm9.py \\"
echo "    --generators_path ${MODEL_PATH} \\"
echo "    --task qualitative \\"
echo "    --use_exact_conditions \\"
echo "    --property_values 'molecular_weight=200,pi_conjugation_ratio=1.0,atom_types_encoding=[C,H,O,N],functional_groups_encoding=[OH,COOH]' \\"
echo "    --n_sweeps 5"
echo ""

echo "=========================================="
echo ""
echo "すべての組み合わせを自動実行するには："
echo ""
echo "  bash generate_all_conditions.sh ${MODEL_PATH}"
echo ""
echo "または："
echo ""
echo "  python generate_molecules_with_conditions.py --model_path ${MODEL_PATH}"
echo ""
echo "=========================================="
echo ""
echo "詳細は以下のドキュメントを参照："
echo "  - QUICK_GENERATION_REFERENCE_JA.md"
echo "  - GENERATION_COMMAND_GUIDE_JA.md"
echo ""
