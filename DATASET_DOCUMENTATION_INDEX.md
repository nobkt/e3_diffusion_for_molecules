# データセット要件に関するドキュメント索引 / Dataset Requirements Documentation Index

## 📚 ドキュメント一覧 / Document List

### 🚀 クイックスタート / Quick Start

**まずはこちらから！ / Start here!**

1. **[クイックリファレンス](./DATASET_REQUIREMENTS_QUICK_REF.md)** 
   - 最も重要なポイントを1ページにまとめた簡潔なガイド
   - Quick one-page reference with the most important points

### ❓ よくある質問 / FAQ

2. **[データセット要件FAQ](./DATASET_REQUIREMENTS_FAQ.md)**
   - SMILESとformulaは必須？という質問への包括的な回答
   - Comprehensive answer to "Are SMILES and formula required?"
   - 実装の詳細、コード例、優先順位付き要件リスト
   - Implementation details, code examples, prioritized requirements

### 📋 詳細な説明 / Detailed Information

3. **[対応まとめ](./DATASET_CLARIFICATION_SUMMARY.md)**
   - 調査結果の詳細
   - Detailed investigation results
   - コードレビューの結果
   - Code review findings
   - 実施した対応の記録
   - Documentation of actions taken

### 📖 使用ガイド / Usage Guide

4. **[単分子条件付き分子性結晶生成 - 使用ガイド](./SINGLE_MOLECULE_CONDITIONED_USAGE_GUIDE.md)**
   - 完全なチュートリアル（FAQ付き）
   - Complete tutorial with FAQ section
   - インストールから生成まで
   - From installation to generation
   - トラブルシューティング
   - Troubleshooting guide

---

## 🎯 シナリオ別ガイド / Scenario-Based Guide

### シナリオ1: 「SMILESは必要？」とすぐに知りたい
**Scenario 1: Quick answer to "Do I need SMILES?"**

→ [クイックリファレンス](./DATASET_REQUIREMENTS_QUICK_REF.md)を読む

**回答: いいえ、不要です / Answer: No, not required**

---

### シナリオ2: OpenBabelを使いたい、RDKitは使いたくない
**Scenario 2: Want to use OpenBabel, not RDKit**

→ [データセット要件FAQ](./DATASET_REQUIREMENTS_FAQ.md) のセクション5を読む

**要点: OpenBabel完全対応、SMILESなしでOK / Key point: Full OpenBabel support, no SMILES needed**

---

### シナリオ3: データベースの作り方を一から学びたい
**Scenario 3: Learn database creation from scratch**

→ [使用ガイド](./SINGLE_MOLECULE_CONDITIONED_USAGE_GUIDE.md) のセクション2を読む

**セクション 2.1と2.2で詳しく説明 / Detailed explanation in sections 2.1 and 2.2**

---

### シナリオ4: なぜSMILESが不要なのか技術的理由を知りたい
**Scenario 4: Want to understand the technical reasons**

→ [対応まとめ](./DATASET_CLARIFICATION_SUMMARY.md) の調査結果セクションを読む

**コードレビューの結果を記載 / Contains code review results**

---

## 🔍 キーワード検索 / Keyword Search

以下のキーワードでドキュメントを探す：

### SMILES関連 / SMILES-related
- "SMILESは必須？" → [FAQ](./DATASET_REQUIREMENTS_FAQ.md), [クイックリファレンス](./DATASET_REQUIREMENTS_QUICK_REF.md)
- "SMILES不要" → すべてのドキュメントで説明

### OpenBabel関連 / OpenBabel-related
- "OpenBabel" → [FAQ](./DATASET_REQUIREMENTS_FAQ.md) セクション5
- "RDKit代替" → [FAQ](./DATASET_REQUIREMENTS_FAQ.md) セクション5

### データベース作成 / Database creation
- "最小限のデータ" → [クイックリファレンス](./DATASET_REQUIREMENTS_QUICK_REF.md)
- "必須フィールド" → [FAQ](./DATASET_REQUIREMENTS_FAQ.md) セクション4
- "molecule.db作成" → [使用ガイド](./SINGLE_MOLECULE_CONDITIONED_USAGE_GUIDE.md) セクション2.1

### トラブルシューティング / Troubleshooting
- "エラー" → [使用ガイド](./SINGLE_MOLECULE_CONDITIONED_USAGE_GUIDE.md) トラブルシューティングセクション
- "動作しない" → まず[クイックリファレンス](./DATASET_REQUIREMENTS_QUICK_REF.md)のチェックリストを確認

---

## 📊 ドキュメントの関係図 / Document Relationship

```
┌─────────────────────────────────────────┐
│  DATASET_REQUIREMENTS_QUICK_REF.md      │  ← 最初に読む / Read first
│  (クイックリファレンス / Quick Ref)      │
└────────────────┬────────────────────────┘
                 │
                 ├─→ 詳しく知りたい / Want more details
                 │
┌────────────────▼────────────────────────┐
│  DATASET_REQUIREMENTS_FAQ.md            │  ← 包括的な説明 / Comprehensive
│  (FAQ)                                  │
└────────────────┬────────────────────────┘
                 │
                 ├─→ 調査の背景を知りたい / Want background
                 │
┌────────────────▼────────────────────────┐
│  DATASET_CLARIFICATION_SUMMARY.md       │  ← 技術的詳細 / Technical details
│  (対応まとめ / Summary)                  │
└─────────────────────────────────────────┘
                 │
                 ├─→ 実際に使いたい / Want to use it
                 │
┌────────────────▼────────────────────────┐
│  SINGLE_MOLECULE_CONDITIONED_           │  ← 完全なチュートリアル
│  USAGE_GUIDE.md                         │     Full tutorial
│  (使用ガイド / Usage Guide)              │
└─────────────────────────────────────────┘
```

---

## 💡 推奨読む順番 / Recommended Reading Order

### 初めての方 / First-time users
1. [クイックリファレンス](./DATASET_REQUIREMENTS_QUICK_REF.md) - 5分
2. [FAQ](./DATASET_REQUIREMENTS_FAQ.md) の結論セクション - 5分
3. [使用ガイド](./SINGLE_MOLECULE_CONDITIONED_USAGE_GUIDE.md) セクション2 - 10分

### 詳しく知りたい方 / Deep dive
1. [FAQ](./DATASET_REQUIREMENTS_FAQ.md) 全体 - 15分
2. [対応まとめ](./DATASET_CLARIFICATION_SUMMARY.md) - 10分
3. [使用ガイド](./SINGLE_MOLECULE_CONDITIONED_USAGE_GUIDE.md) 全体 - 30分

### 技術者・開発者 / Developers
1. [対応まとめ](./DATASET_CLARIFICATION_SUMMARY.md) - コードレビュー結果
2. [FAQ](./DATASET_REQUIREMENTS_FAQ.md) - 実装の詳細
3. `crystal/data/molecular_crystal_loader.py` のソースコード

---

## 📞 さらにサポートが必要な場合 / Need More Help?

- **GitHub Issues**: バグ報告や機能リクエスト / Bug reports and feature requests
- **GitHub Discussions**: 一般的な質問 / General questions
- このドキュメントで不明な点があれば Issue を開いてください

---

## 🔄 最終更新 / Last Updated

- **日付 / Date**: 2025-10-11
- **バージョン / Version**: 1.0
- **ステータス / Status**: ✅ Complete

---

## 📝 ドキュメント作成の経緯 / Document Creation History

ユーザーから「SMILESとformulaは必須か？」という質問があり、コードベースを調査した結果、以下が明らかになりました：

- SMILESとformulaは**オプション**であり、必須ではない
- モデルは3D座標と原子番号のみを使用
- OpenBabelで分子を準備可能（RDKit不要）

この調査結果を元に、ユーザーが簡単に情報にアクセスできるよう、複数のドキュメントを作成しました。

---

**このインデックスドキュメント自体も含めて、すべてのドキュメントは日本語と英語の両方で記載されています。**

**All documents, including this index, are written in both Japanese and English.**
