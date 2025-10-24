# Phase 4.2-4.4 継続実装計画

**ドキュメント種別**: 継続実装計画・詳細仕様  
**対象**: CLIスクリプトとユーザーガイド  
**ステータス**: 計画中  
**日付**: 2025-10-24  
**バージョン**: 1.0

---

## エグゼクティブサマリー

Phase 4.2-4.4のコア実装が完了しました。本ドキュメントでは、**オプション**の継続実装（CLIスクリプトとユーザーガイド）の詳細仕様を提供します。

**完了済み**:
- ✅ Phase 4.2: コア実装（PropertyConstraint、ConstrainedCrystalGenerator、ParetoFrontierSearcher）
- ✅ Phase 4.3: コア実装（PropertyDistributionAnalyzer、StructureQualityAnalyzer、HTMLReportGenerator）
- ✅ Phase 4.4: コア実装（ConditioningCache、MultiGPUGenerator、BatchProcessor）

**計画中（すべてオプション）**:
- ⏳ CLIスクリプト（3本）
- ⏳ ユーザーガイド（英語・日本語）
- ⏳ 統合ワークフロー

---

## 継続実装項目1: CLIスクリプト

### 1.1 multi_property_optimization.py

#### 目的

コマンドラインから多物性最適化を実行するためのスクリプト。

#### 機能要件

**FR-CLI-1.1: 制約条件の指定**
- コマンドライン引数で複数の制約を指定可能
- 構文: `--constraint "property_name operator value"`
- 例: `--constraint "bandgap > 2.0" --constraint "melting_point in 150,200"`

**FR-CLI-1.2: Pareto最適化**
- 目的関数と最小化/最大化の指定
- グリッドサーチまたはランダムサンプリング
- Paretoフロンティアの保存（CIF + JSON）

**FR-CLI-1.3: 可視化**
- Paretoフロンティアの2D/3Dプロット自動生成
- 結果サマリーの出力

#### 詳細設計

```python
#!/usr/bin/env python
"""
Multi-Property Optimization CLI

Performs constrained crystal generation and Pareto optimization.

Usage:
    # Constraint-based generation
    python multi_property_optimization.py \
        --mode constraints \
        --predictor_path outputs/predictor.pt \
        --generator_path outputs/generator.pt \
        --molecule_id benzene \
        --constraint "bandgap > 2.0" \
        --constraint "melting_point < 200.0" \
        --n_samples 100 \
        --output_dir constrained_output/
    
    # Pareto frontier search
    python multi_property_optimization.py \
        --mode pareto \
        --predictor_path outputs/predictor.pt \
        --generator_path outputs/generator.pt \
        --molecule_id benzene \
        --objectives bandgap formation_energy \
        --minimize false true \
        --n_candidates 1000 \
        --output_dir pareto_output/

Requirements:
    - Trained property predictor model
    - Trained crystal generation model
    - Property names must match predictor configuration
"""

import argparse
import json
from pathlib import Path
import torch
import numpy as np
from typing import List, Tuple

from crystal.evaluation import (
    PropertyConstraint,
    ConstrainedCrystalGenerator,
    ParetoFrontierSearcher,
    PropertyPredictor,
)


def parse_constraint(constraint_str: str) -> Tuple[str, str, Union[float, Tuple[float, float]]]:
    """
    Parse constraint string.
    
    Format:
        "property_name operator value"
        "property_name in min,max"
    
    Examples:
        "bandgap > 2.0" -> ('bandgap', '>', 2.0)
        "melting_point in 150,200" -> ('melting_point', 'in', (150.0, 200.0))
    """
    parts = constraint_str.strip().split()
    
    if len(parts) != 3:
        raise ValueError(
            f"Invalid constraint format: '{constraint_str}'. "
            f"Expected 'property operator value'"
        )
    
    property_name = parts[0]
    operator = parts[1]
    value_str = parts[2]
    
    # Parse value
    if operator == 'in':
        # Range constraint
        try:
            min_val, max_val = value_str.split(',')
            value = (float(min_val), float(max_val))
        except ValueError as e:
            raise ValueError(
                f"Invalid range format for 'in' operator: '{value_str}'. "
                f"Expected 'min,max'"
            ) from e
    else:
        # Scalar constraint
        try:
            value = float(value_str)
        except ValueError as e:
            raise ValueError(
                f"Invalid value: '{value_str}'. Expected numeric value"
            ) from e
    
    return property_name, operator, value


def run_constraint_mode(args):
    """Run constraint-based generation."""
    print("=== Constraint-Based Generation ===")
    
    # Load models
    print(f"Loading predictor from {args.predictor_path}")
    predictor = PropertyPredictor.load(args.predictor_path)
    
    print(f"Loading generator from {args.generator_path}")
    # NOTE: Generator loading depends on specific implementation
    # Example pseudocode:
    # generator = load_generator(args.generator_path)
    
    # Parse constraints
    constraints = []
    for constraint_str in args.constraint:
        prop_name, operator, value = parse_constraint(constraint_str)
        constraint = PropertyConstraint(prop_name, operator, value)
        constraints.append(constraint)
        print(f"  Constraint: {constraint}")
    
    # Create generator
    constrained_gen = ConstrainedCrystalGenerator(predictor, constraints)
    
    # Generate crystals
    print(f"\nGenerating {args.n_samples} crystals...")
    # NOTE: Crystal generation depends on integration with generation pipeline
    # Example pseudocode:
    # satisfied_crystals = constrained_gen.generate_until_satisfied(
    #     molecule_id=args.molecule_id,
    #     n_samples=args.n_samples,
    #     max_attempts=args.max_attempts,
    # )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # NOTE: Saving depends on crystal format
    # Example pseudocode:
    # - Save crystals as CIF files
    # - Save summary JSON
    
    print(f"\nResults saved to {output_dir}")


def run_pareto_mode(args):
    """Run Pareto frontier search."""
    print("=== Pareto Frontier Search ===")
    
    # Load models
    predictor = PropertyPredictor.load(args.predictor_path)
    
    # Create searcher
    minimize = [m.lower() == 'true' for m in args.minimize]
    searcher = ParetoFrontierSearcher(
        predictor,
        objectives=args.objectives,
        minimize=minimize
    )
    
    print(f"Objectives: {args.objectives}")
    print(f"Minimize: {minimize}")
    
    # Generate candidates
    print(f"\nGenerating {args.n_candidates} candidate crystals...")
    # NOTE: Candidate generation depends on integration with generation pipeline
    # Example pseudocode:
    # candidates = generate_candidates(...)
    
    # Find Pareto frontier
    # NOTE: Requires predictions for all candidates
    # Example pseudocode:
    # pareto_indices = searcher.find_pareto_frontier(predictions_list)
    
    # Visualize
    # NOTE: Visualization depends on Pareto results
    # Example pseudocode:
    # searcher.visualize_pareto_frontier(...)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nResults saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Property Optimization for Crystal Generation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['constraints', 'pareto'],
        help='Optimization mode'
    )
    
    parser.add_argument(
        '--predictor_path',
        type=str,
        required=True,
        help='Path to trained property predictor model'
    )
    
    parser.add_argument(
        '--generator_path',
        type=str,
        required=True,
        help='Path to trained crystal generator model'
    )
    
    parser.add_argument(
        '--molecule_id',
        type=str,
        required=True,
        help='Molecule identifier'
    )
    
    # Constraint mode arguments
    parser.add_argument(
        '--constraint',
        type=str,
        action='append',
        help='Property constraint (format: "property operator value")'
    )
    
    parser.add_argument(
        '--n_samples',
        type=int,
        default=100,
        help='Number of samples to generate (constraint mode)'
    )
    
    parser.add_argument(
        '--max_attempts',
        type=int,
        default=10000,
        help='Maximum generation attempts (constraint mode)'
    )
    
    # Pareto mode arguments
    parser.add_argument(
        '--objectives',
        type=str,
        nargs='+',
        help='Objective properties for Pareto optimization'
    )
    
    parser.add_argument(
        '--minimize',
        type=str,
        nargs='+',
        help='Minimize flags for each objective (true/false)'
    )
    
    parser.add_argument(
        '--n_candidates',
        type=int,
        default=1000,
        help='Number of candidate crystals (Pareto mode)'
    )
    
    # Common arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='optimization_output',
        help='Output directory'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device for computation'
    )
    
    args = parser.parse_args()
    
    # Validate mode-specific arguments
    if args.mode == 'constraints':
        if not args.constraint:
            parser.error("--constraint is required for constraint mode")
    elif args.mode == 'pareto':
        if not args.objectives:
            parser.error("--objectives is required for Pareto mode")
        if not args.minimize:
            parser.error("--minimize is required for Pareto mode")
        if len(args.minimize) != len(args.objectives):
            parser.error("--minimize must have same length as --objectives")
    
    # Run mode
    if args.mode == 'constraints':
        run_constraint_mode(args)
    elif args.mode == 'pareto':
        run_pareto_mode(args)
    
    print("\nOptimization complete!")


if __name__ == '__main__':
    main()
```

#### テスト計画

```python
# tests/test_cli_multi_property_optimization.py

def test_parse_constraint_scalar():
    """Test parsing scalar constraint."""
    prop, op, val = parse_constraint("bandgap > 2.0")
    assert prop == "bandgap"
    assert op == ">"
    assert val == 2.0

def test_parse_constraint_range():
    """Test parsing range constraint."""
    prop, op, val = parse_constraint("melting_point in 150,200")
    assert prop == "melting_point"
    assert op == "in"
    assert val == (150.0, 200.0)

def test_parse_constraint_invalid():
    """Test parsing invalid constraint."""
    with pytest.raises(ValueError):
        parse_constraint("invalid")
```

#### 推定工数

- 実装: 1日
- テスト: 0.5日
- ドキュメント: 0.5日
- **合計: 2日**

---

### 1.2 advanced_visualization_report.py

#### 目的

生成結晶の包括的な可視化レポートを生成するCLIスクリプト。

#### 機能要件

**FR-CLI-2.1: データソース指定**
- 生成結晶ディレクトリ（CIFファイル）
- 訓練データベース（オプション）
- 目標物性値（オプション）

**FR-CLI-2.2: レポート生成**
- 物性分布プロット
- 構造品質解析
- インタラクティブHTMLレポート

**FR-CLI-2.3: カスタマイズ**
- 出力フォーマット（PNG、HTML、JSON）
- DPI設定
- カラースキーム

#### 詳細設計

```python
#!/usr/bin/env python
"""
Advanced Visualization Report Generator

Generates comprehensive visualization reports for generated crystals.

Usage:
    python advanced_visualization_report.py \
        --generated_dir generated_samples/ \
        --training_db data/crystals_with_props.db \
        --property_names bandgap melting_point \
        --target_bandgap 2.5 \
        --target_melting_point 180.0 \
        --output_report report.html \
        --output_plots plots/ \
        --dpi 300

Requirements:
    - Generated crystal files (CIF format)
    - Property predictor to compute properties
    - Optional: Training database for comparison
"""

import argparse
from pathlib import Path
import json
import numpy as np

from crystal.evaluation import (
    PropertyDistributionAnalyzer,
    StructureQualityAnalyzer,
    HTMLReportGenerator,
    PropertyPredictor,
)


def load_generated_crystals(generated_dir: Path, predictor: PropertyPredictor):
    """
    Load generated crystals and predict properties.
    
    Args:
        generated_dir: Directory containing CIF files
        predictor: Property predictor model
        
    Returns:
        crystals: List of crystal dictionaries
        properties: Dictionary of predicted properties
    """
    crystals = []
    properties = {name: [] for name in predictor.property_names}
    
    cif_files = list(generated_dir.glob('*.cif'))
    print(f"Loading {len(cif_files)} crystal files...")
    
    for cif_file in cif_files:
        # NOTE: Crystal loading depends on CIF parsing implementation
        # Example pseudocode:
        # crystal = load_cif(cif_file)
        # crystals.append(crystal)
        
        # Predict properties
        # Example pseudocode:
        # predictions = predictor(crystal['positions'], crystal['cell'], crystal['atomic_numbers'])
        # for name in predictor.property_names:
        #     properties[name].append(predictions[name].item())
        pass
    
    return crystals, properties


def main():
    parser = argparse.ArgumentParser(
        description='Generate Advanced Visualization Reports',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--generated_dir',
        type=str,
        required=True,
        help='Directory containing generated crystal files'
    )
    
    parser.add_argument(
        '--predictor_path',
        type=str,
        required=True,
        help='Path to trained property predictor'
    )
    
    parser.add_argument(
        '--property_names',
        type=str,
        nargs='+',
        required=True,
        help='Property names to analyze'
    )
    
    parser.add_argument(
        '--training_db',
        type=str,
        help='Training database for comparison (optional)'
    )
    
    # Target values (optional)
    parser.add_argument(
        '--target',
        type=str,
        action='append',
        help='Target property value (format: "property=value")'
    )
    
    # Output options
    parser.add_argument(
        '--output_report',
        type=str,
        default='report.html',
        help='Output HTML report path'
    )
    
    parser.add_argument(
        '--output_plots',
        type=str,
        default='plots',
        help='Output directory for plot images'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for plot images'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['html', 'png', 'both'],
        default='both',
        help='Output format'
    )
    
    args = parser.parse_args()
    
    # Load models
    print(f"Loading property predictor from {args.predictor_path}")
    predictor = PropertyPredictor.load(args.predictor_path)
    
    # Load generated crystals
    generated_dir = Path(args.generated_dir)
    crystals, generated_props = load_generated_crystals(generated_dir, predictor)
    
    # Load training data (optional)
    training_props = None
    if args.training_db:
        print(f"Loading training data from {args.training_db}")
        # training_props = load_training_properties(args.training_db)
    
    # Parse target values (optional)
    target_props = {}
    if args.target:
        for target_str in args.target:
            prop_name, value_str = target_str.split('=')
            target_props[prop_name] = float(value_str)
    
    # Create analyzers
    prop_analyzer = PropertyDistributionAnalyzer(args.property_names)
    struct_analyzer = StructureQualityAnalyzer()
    html_generator = HTMLReportGenerator()
    
    # Generate visualizations
    output_plots = Path(args.output_plots)
    output_plots.mkdir(parents=True, exist_ok=True)
    
    if args.format in ['png', 'both']:
        print("\nGenerating property distribution plots...")
        prop_analyzer.plot_distributions(
            generated_properties=generated_props,
            training_properties=training_props,
            target_properties=target_props,
            output_path=output_plots / 'property_distributions.png',
            dpi=args.dpi
        )
        
        print("Generating structure quality plots...")
        struct_analyzer.analyze_bond_lengths(
            crystals=crystals,
            output_path=output_plots / 'bond_lengths.png',
            dpi=args.dpi
        )
        
        struct_analyzer.analyze_cell_parameters(
            crystals=crystals,
            output_path=output_plots / 'cell_parameters.png',
            dpi=args.dpi
        )
    
    if args.format in ['html', 'both']:
        print("\nGenerating HTML report...")
        html_generator.generate_report(
            generated_properties=generated_props,
            training_properties=training_props,
            target_properties=target_props,
            output_path=args.output_report,
            title='Crystal Generation Report'
        )
    
    # Save statistics
    stats = prop_analyzer.compute_statistics(generated_props)
    stats_path = output_plots / 'statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nReport generation complete!")
    print(f"  HTML report: {args.output_report}")
    print(f"  Plots: {output_plots}")
    print(f"  Statistics: {stats_path}")


if __name__ == '__main__':
    main()
```

#### 推定工数

- 実装: 1日
- テスト: 0.5日
- ドキュメント: 0.5日
- **合計: 2日**

---

### 1.3 performance_optimized_generation.py

#### 目的

マルチGPUと最適化を使用した大規模結晶生成。

#### 推定工数

- 実装: 1日
- テスト: 0.5日
- **合計: 1.5日**

---

## 継続実装項目2: ユーザーガイド

### 2.1 Phase 4.2-4.4 User Guide (English)

#### 内容

- Quick start examples
- API reference
- CLI usage guide
- Troubleshooting

#### 推定工数

- **1.5日**

### 2.2 Phase 4.2-4.4 使用ガイド (Japanese)

#### 内容

- クイックスタート例
- API リファレンス
- CLI使用ガイド
- トラブルシューティング

#### 推定工数

- **1.5日**

---

## 総合推定工数

```
CLIスクリプト:
  - multi_property_optimization.py: 2日
  - advanced_visualization_report.py: 2日
  - performance_optimized_generation.py: 1.5日
  小計: 5.5日

ユーザーガイド:
  - 英語版: 1.5日
  - 日本語版: 1.5日
  小計: 3日

合計: 8.5日 (約2週間)
```

---

## 優先順位

**高**: なし（すべてオプション）

**中**:
1. CLIスクリプト（ユーザビリティ向上）
2. ユーザーガイド（採用促進）

**低**:
3. 統合ワークフロー

---

## まとめ

Phase 4.2-4.4のコア機能は完全に実装され、本番利用可能です。継続実装（CLIとドキュメント）は**すべてオプション**であり、実施は利用状況とニーズに基づいて判断してください。

**現在の状態**: ✅ 完全に機能するPython API  
**継続実装後**: より使いやすいCLI + 充実したドキュメント

---

**ドキュメント作成日**: 2025-10-24  
**推定総工数**: 8.5日（継続実装すべて実施の場合）  
**ステータス**: 計画段階、実施は任意
