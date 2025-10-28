#!/usr/bin/env python3
"""
条件付き分子生成スクリプト

このスクリプトは、指定された条件の組み合わせで分子を生成します。

使用例:
    python generate_molecules_with_conditions.py --model_path outputs/molecular_descriptor_model
    
    python generate_molecules_with_conditions.py \
        --model_path outputs/molecular_descriptor_model \
        --molecular_weights 100 200 300 \
        --pi_conjugations 0.8 0.9 1.0 \
        --n_sweeps 10
"""

import argparse
import subprocess
import os
from datetime import datetime
from pathlib import Path


def run_generation(model_path, property_values, n_sweeps, output_dir):
    """
    指定された条件で分子生成を実行
    
    Args:
        model_path: 訓練済みモデルのパス
        property_values: プロパティ値の文字列
        n_sweeps: 生成するサンプル数
        output_dir: 出力ディレクトリ
        
    Returns:
        tuple: (成功フラグ, 標準出力, 標準エラー出力)
    """
    cmd = [
        "python", "eval_conditional_qm9.py",
        "--generators_path", model_path,
        "--task", "qualitative",
        "--use_exact_conditions",
        "--property_values", property_values,
        "--n_sweeps", str(n_sweeps)
    ]
    
    print(f"実行コマンド: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1時間のタイムアウト
        )
        
        success = result.returncode == 0
        return success, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        print("エラー: 実行がタイムアウトしました（1時間）")
        return False, "", "Timeout"
    except Exception as e:
        print(f"エラー: {e}")
        return False, "", str(e)


def format_property_values(molecular_weight, pi_conjugation, atom_types, functional_groups):
    """
    プロパティ値を文字列形式にフォーマット
    
    Args:
        molecular_weight: 分子量
        pi_conjugation: π共役性
        atom_types: 原子種のリスト
        functional_groups: 官能基のリスト
        
    Returns:
        str: フォーマットされたプロパティ値文字列
    """
    atom_types_str = "[" + ",".join(atom_types) + "]"
    functional_groups_str = "[" + ",".join(functional_groups) + "]"
    
    return (
        f"molecular_weight={molecular_weight},"
        f"pi_conjugation_ratio={pi_conjugation},"
        f"atom_types_encoding={atom_types_str},"
        f"functional_groups_encoding={functional_groups_str}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="条件付き分子生成スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # デフォルトの条件で実行
  python generate_molecules_with_conditions.py --model_path outputs/my_model
  
  # カスタム条件で実行
  python generate_molecules_with_conditions.py \\
      --model_path outputs/my_model \\
      --molecular_weights 50 100 150 200 \\
      --pi_conjugations 0.6 0.8 1.0 \\
      --atom_types C H O N S \\
      --functional_groups OH COOH NH2 \\
      --n_sweeps 10
        """
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/molecular_descriptor_model",
        help="訓練済みモデルのパス（デフォルト: outputs/molecular_descriptor_model）"
    )
    
    parser.add_argument(
        "--molecular_weights",
        type=float,
        nargs="+",
        default=[100, 200],
        help="分子量のリスト（デフォルト: 100 200）"
    )
    
    parser.add_argument(
        "--pi_conjugations",
        type=float,
        nargs="+",
        default=[0.8, 0.9, 1.0],
        help="π共役性のリスト（デフォルト: 0.8 0.9 1.0）"
    )
    
    parser.add_argument(
        "--atom_types",
        type=str,
        nargs="+",
        default=["C", "H", "O", "N"],
        help="原子種のリスト（デフォルト: C H O N）"
    )
    
    parser.add_argument(
        "--functional_groups",
        type=str,
        nargs="+",
        default=["OH", "COOH"],
        help="官能基のリスト（デフォルト: OH COOH）"
    )
    
    parser.add_argument(
        "--n_sweeps",
        type=int,
        default=5,
        help="各条件でのサンプル数（デフォルト: 5）"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="出力ディレクトリ（デフォルト: model_path/generated_samples_TIMESTAMP）"
    )
    
    args = parser.parse_args()
    
    # モデルパスの確認
    if not os.path.exists(args.model_path):
        print(f"エラー: モデルディレクトリが見つかりません: {args.model_path}")
        return 1
    
    # 出力ディレクトリの作成
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(args.model_path, f"generated_samples_{timestamp}")
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 実行情報の表示
    print("=" * 60)
    print("条件付き分子生成スクリプト")
    print("=" * 60)
    print(f"モデルパス: {args.model_path}")
    print(f"出力ディレクトリ: {args.output_dir}")
    print()
    print("生成条件:")
    print(f"  原子種: {args.atom_types}")
    print(f"  官能基: {args.functional_groups}")
    print(f"  分子量: {args.molecular_weights}")
    print(f"  π共役性: {args.pi_conjugations}")
    print(f"  サンプル数/条件: {args.n_sweeps}")
    print()
    
    # ログファイル
    log_file = os.path.join(args.output_dir, "generation_log.txt")
    print(f"ログファイル: {log_file}")
    print("=" * 60)
    print()
    
    # 結果の記録
    results = []
    total_combinations = len(args.molecular_weights) * len(args.pi_conjugations)
    current = 0
    
    start_time = datetime.now()
    
    # すべての組み合わせで生成
    for mw in args.molecular_weights:
        for pc in args.pi_conjugations:
            current += 1
            
            print(f"[{current}/{total_combinations}] 分子生成中...")
            print(f"  分子量: {mw}")
            print(f"  π共役性: {pc}")
            
            # プロパティ値の構築
            property_values = format_property_values(
                mw, pc, args.atom_types, args.functional_groups
            )
            
            # 生成実行
            with open(log_file, "a") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"条件 [{current}/{total_combinations}]\n")
                f.write(f"開始時刻: {datetime.now()}\n")
                f.write(f"分子量: {mw}, π共役性: {pc}\n")
                f.write(f"プロパティ値: {property_values}\n")
                f.write(f"{'='*60}\n\n")
            
            success, stdout, stderr = run_generation(
                args.model_path, property_values, args.n_sweeps, args.output_dir
            )
            
            # ログに記録
            with open(log_file, "a") as f:
                f.write(stdout)
                if stderr:
                    f.write("\nエラー出力:\n")
                    f.write(stderr)
                f.write(f"\n終了時刻: {datetime.now()}\n")
                f.write(f"結果: {'成功' if success else '失敗'}\n")
            
            # 結果の記録
            results.append({
                "molecular_weight": mw,
                "pi_conjugation": pc,
                "success": success
            })
            
            if success:
                print("  ✓ 成功")
            else:
                print("  ✗ 失敗")
                print(f"  エラー: {stderr[:200]}")
            
            print()
    
    end_time = datetime.now()
    elapsed = end_time - start_time
    
    # サマリーの作成
    print("=" * 60)
    print("全ての生成が完了しました")
    print("=" * 60)
    print(f"実行時間: {elapsed}")
    print(f"出力ディレクトリ: {args.output_dir}")
    print(f"ログファイル: {log_file}")
    print()
    
    # 成功/失敗のサマリー
    success_count = sum(1 for r in results if r["success"])
    print(f"成功: {success_count}/{len(results)}")
    print(f"失敗: {len(results) - success_count}/{len(results)}")
    print()
    
    # サマリーファイルの作成
    summary_file = os.path.join(args.output_dir, "summary.txt")
    with open(summary_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("条件付き分子生成サマリー\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"実行日時: {start_time}\n")
        f.write(f"モデルパス: {args.model_path}\n")
        f.write(f"出力ディレクトリ: {args.output_dir}\n\n")
        f.write("生成条件:\n")
        f.write(f"  原子種: {args.atom_types}\n")
        f.write(f"  官能基: {args.functional_groups}\n")
        f.write(f"  分子量: {args.molecular_weights}\n")
        f.write(f"  π共役性: {args.pi_conjugations}\n")
        f.write(f"  サンプル数/条件: {args.n_sweeps}\n\n")
        f.write(f"総組み合わせ数: {total_combinations}\n")
        f.write(f"実行時間: {elapsed}\n")
        f.write(f"成功: {success_count}/{len(results)}\n")
        f.write(f"失敗: {len(results) - success_count}/{len(results)}\n\n")
        f.write("詳細結果:\n")
        for i, r in enumerate(results, 1):
            status = "✓" if r["success"] else "✗"
            f.write(f"  [{i}] MW={r['molecular_weight']}, PC={r['pi_conjugation']}: {status}\n")
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"サマリーファイル: {summary_file}")
    print()
    print("次のステップ:")
    print(f"  1. 生成された分子の可視化:")
    print(f"     python eval_sample.py --model_path {args.model_path}")
    print()
    print(f"  2. 分子の評価と解析:")
    print(f"     python eval_analyze.py --model_path {args.model_path} --n_samples 1000")
    print()
    
    return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    exit(main())
