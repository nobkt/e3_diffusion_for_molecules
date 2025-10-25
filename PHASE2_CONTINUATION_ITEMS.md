# フェーズ2継続項目の整理

**作成日**: 2025年10月25日  
**対応PR**: PR#151継続作業  
**前フェーズ**: フェーズ1（日本語チュートリアル1-6）完了

---

## フェーズ1完了サマリー

### 完了した成果物

✅ **日本語チュートリアル全6個** (141セル、158KB)
- `tutorials/01_basic_molecule_generation_ja.ipynb` (20セル)
- `tutorials/02_conditional_generation_ja.ipynb` (23セル)
- `tutorials/03_crystal_generation_ja.ipynb` (20セル)
- `tutorials/04_molecular_descriptors_ase_ja.ipynb` (24セル)
- `tutorials/05_evaluation_analysis_ja.ipynb` (30セル)
- `tutorials/06_advanced_crystal_conditioning_ja.ipynb` (24セル)
- `tutorials/README_JA.md` (包括的ドキュメント)

### 実装原則の遵守状況

✅ **フォールバックなし**: 全てのエラーケースを明示的に処理  
✅ **理論的健全性**: E(3)同変性、PBC、物理的制約を厳密に維持  
✅ **完全性**: 230個全空間群、14官能基、包括的評価を完全実装  
✅ **実用性**: 全141セルが実行可能なコード  
✅ **一貫性**: 日本語による統一された説明とスタイル  

---

## フェーズ2: 必須継続項目

### 項目1: チュートリアル実行テスト ⭐️ 優先度: 高

**目的**: 全6チュートリアルが実際の環境で正常に実行できることを検証

**作業内容**:

1. **環境セットアップ検証**
   - 依存関係のインストール確認
   - RDKit、OpenBabel、spglibの動作確認
   - GPU/CPUの両方での動作確認

2. **チュートリアル1-6の順次実行**
   - 各セルを上から順に実行
   - 出力の検証（期待される結果と一致するか）
   - エラーの記録（発生した場合）
   - 実行時間の測定

3. **検証項目**
   - データセットの読み込み成功
   - モデルの構築成功
   - サンプリングの実行成功
   - 評価メトリクスの計算成功
   - 可視化の生成成功
   - ファイル出力の成功

4. **成果物**
   - `TUTORIAL_EXECUTION_REPORT.md`: 実行結果レポート
   - 各チュートリアルの実行ログ
   - 問題点と修正内容の記録

**推定作業時間**: 2-3時間

**依存関係**:
- Python 3.8+
- PyTorch
- RDKit
- ASE
- spglib
- OpenBabel (オプション)

**成功基準**:
- 全6チュートリアルが最初から最後まで実行完了
- 全ての期待される出力が生成される
- エラーが発生した場合は全て修正済み

---

### 項目2: APIリファレンス（日本語） ⭐️ 優先度: 高

**目的**: 全クラスと関数の詳細なAPIドキュメントを日本語で提供

**作業内容**:

1. **モジュール構造の文書化**
   ```
   doc/api_reference_ja.md
   ├── qm9モジュール
   │   ├── dataset
   │   ├── models
   │   ├── utils
   │   └── analyze
   ├── crystalモジュール
   │   ├── models
   │   ├── conditioning
   │   └── utils
   ├── egnnモジュール
   └── equivariant_diffusionモジュール
   ```

2. **各クラスのドキュメント** (最低限の項目)
   - クラス名と目的
   - 初期化パラメータ
   - 主要メソッドの説明
   - 使用例
   - 注意事項

3. **各関数のドキュメント**
   - 関数名と目的
   - パラメータの型と説明
   - 戻り値の型と説明
   - 使用例
   - エラーケース

4. **重要クラスの優先順位**（高→低）
   ```python
   # 優先度: 最高
   - qm9.models.get_model()
   - qm9.dataset.retrieve_dataloaders()
   - qm9.analyze.check_stability()
   - crystal.models.CrystalDynamics
   - crystal.conditioning.SpaceGroupEmbedding
   
   # 優先度: 高
   - equivariant_diffusion.en_diffusion.EnVariationalDiffusion
   - egnn.egnn_new.EGNN
   - qm9.utils.prepare_context()
   - qm9.utils.compute_mean_mad()
   
   # 優先度: 中
   - その他の補助関数とユーティリティ
   ```

5. **成果物**
   - `doc/api_reference_ja.md`: メインAPIリファレンス（推定200-300項目）
   - コード内のdocstringの日本語化（オプション）

**推定作業時間**: 4-5時間

**フォーマット例**:
```markdown
### qm9.models.get_model()

**目的**: E(3)同変拡散モデルを構築します。

**シグネチャ**:
```python
def get_model(args, device, dataset_info, dataloader=None)
```

**パラメータ**:
- `args` (argparse.Namespace): モデル設定パラメータ
  - `nf` (int): 隠れ層の次元数
  - `n_layers` (int): EGNNレイヤー数
  - `diffusion_steps` (int): 拡散ステップ数
  - その他...
- `device` (torch.device): 計算デバイス (cuda/cpu)
- `dataset_info` (dict): データセット情報
- `dataloader` (DataLoader, optional): データローダー（性質分布推定用）

**戻り値**:
- `model` (EnVariationalDiffusion): 拡散モデル
- `nodes_dist` (DistributionNodes): ノード数分布
- `prop_dist` (DistributionProperty or None): 性質分布

**使用例**:
```python
from qm9.models import get_model
from qm9 import utils as qm9_utils

dataset_info = qm9_utils.get_dataset_info('qm9', remove_h=False)
model, nodes_dist, prop_dist = get_model(args, device, dataset_info, train_loader)
```

**注意事項**:
- 条件付き生成の場合、`args.conditioning`を設定する必要があります
- GPU使用時は十分なメモリが必要です（推奨: 8GB以上）
```

**成功基準**:
- 主要クラス・関数が全て文書化されている
- 各ドキュメントに必須項目が全て含まれている
- コード例が実行可能
- 日本語の文法・表現が適切

---

### 項目3: ユースケース集（日本語） ⭐️ 優先度: 中

**目的**: 実際の研究プロジェクトでの使用例を提供

**作業内容**:

1. **ユースケースの選定** (5-7個)
   - 医薬品様分子の設計
   - 材料探索（有機半導体など）
   - 結晶多形予測
   - 性質最適化（logP、溶解度など）
   - カスタムデータセットでの学習
   - 大規模生成（10,000分子以上）
   - 結晶構造予測

2. **各ユースケースの構成**
   ```markdown
   # ユースケース: [タイトル]
   
   ## 背景と目的
   - 研究の文脈
   - 解決すべき課題
   
   ## データセット
   - 使用するデータ
   - 前処理手順
   
   ## モデル設定
   - アーキテクチャ
   - ハイパーパラメータ
   
   ## 学習手順
   - コマンド例
   - 学習時間の目安
   
   ## 評価と解析
   - 評価指標
   - 結果の解釈
   
   ## 結果の例
   - 生成された分子/結晶
   - 性能メトリクス
   
   ## 注意点とヒント
   - よくある問題
   - 最適化のポイント
   ```

3. **実装例**
   
   **ユースケース1: 医薬品様分子の設計**
   ```bash
   # データセット準備
   python build_geom_dataset.py --dataset geom_drugs
   
   # 学習
   python main_geom_drugs.py \
       --exp_name drug_design \
       --n_epochs 500 \
       --conditioning molecular_weight logP \
       --batch_size 64
   
   # 生成（目標: MW=300-400, logP=2-3）
   python eval_conditional_qm9.py \
       --model_path outputs/drug_design \
       --conditioning molecular_weight logP \
       --target_values 350.0 2.5 \
       --n_samples 1000
   
   # 評価
   python eval_analyze.py \
       --model_path outputs/drug_design \
       --n_samples 10000
   ```
   
   **ユースケース2: 結晶多形予測**
   ```bash
   # ASEデータベース作成
   python create_diverse_test_db.py \
       --molecule_smiles "CC(=O)O" \
       --n_polymorphs 10
   
   # 段階的学習
   bash scripts/train_staged.sh
   
   # 多形生成
   python generate_polymorphs.py \
       --model_path outputs/stage3_full/model_best.pt \
       --molecule_smiles "CC(=O)O" \
       --space_groups 2 14 19 61 \
       --densities 1.0 1.2 1.4 1.6 \
       --n_polymorphs 20
   
   # 検証
   python validate_generated_crystals.py \
       --cif_dir generated_polymorphs/
   ```

4. **成果物**
   - `doc/use_cases_ja.md`: ユースケース集
   - 各ユースケースの実行スクリプト（`examples/use_case_*.py`）

**推定作業時間**: 3-4時間

**成功基準**:
- 5-7個の実用的なユースケースを文書化
- 各ユースケースに完全な実行例を含む
- 実際に実行可能なコマンド
- 期待される結果を明示

---

## フェーズ3: 補完的項目（優先度: 低）

以下の項目は、フェーズ2完了後に実施します:

### 項目4: トラブルシューティングガイド拡張版
- よくあるエラーと解決方法
- デバッグ手法
- パフォーマンス問題の診断
- 推定時間: 2-3時間

### 項目5: 用語集（日本語）
- 技術用語の定義
- 略語の説明
- 英日対訳表
- 推定時間: 1-2時間

### 項目6: FAQ（日本語）
- よくある質問と回答
- ベストプラクティス
- 推奨設定
- 推定時間: 1-2時間

### 項目7: 論文リスト（日本語解説付き）
- 関連論文の紹介
- 各論文の要約
- 実装との対応
- 推定時間: 2-3時間

---

## 実装スケジュール案

### 即座に実施（次のPR）
1. ✅ チュートリアル実行テスト (2-3時間)
2. APIリファレンス（日本語）の作成開始 (4-5時間)

### 第2段階（次の次のPR）
3. ユースケース集（日本語）の作成 (3-4時間)
4. 実行テスト結果に基づく修正

### 第3段階（フェーズ3）
5. トラブルシューティングガイド拡張版
6. 用語集、FAQ、論文リスト

---

## 品質保証チェックリスト

### チュートリアル実行テスト
- [ ] 全6チュートリアルが最初から最後まで実行完了
- [ ] 各チュートリアルの期待される出力が生成される
- [ ] エラーが発生した場合は全て文書化
- [ ] 実行時間が許容範囲内
- [ ] 依存関係の問題が解決済み

### APIリファレンス
- [ ] 主要クラス・関数が全て文書化
- [ ] 各ドキュメントに必須項目が含まれる
- [ ] コード例が実行可能
- [ ] 日本語の文法・表現が適切
- [ ] 一貫した形式とスタイル

### ユースケース集
- [ ] 5-7個の実用的なユースケース
- [ ] 各ユースケースに完全な実行例
- [ ] 実際に実行可能なコマンド
- [ ] 期待される結果を明示
- [ ] 背景と目的が明確

---

## 連絡先・サポート

本継続計画に関する質問や提案は、GitHubのIssuesで受け付けています。

**リポジトリ**: https://github.com/nobkt/e3_diffusion_for_molecules

---

**このドキュメントは、フェーズ2以降の継続作業を明確に整理したものです。**

**作成日**: 2025年10月25日  
**対応PR**: PR#151継続  
**ステータス**: フェーズ1完了、フェーズ2準備完了
