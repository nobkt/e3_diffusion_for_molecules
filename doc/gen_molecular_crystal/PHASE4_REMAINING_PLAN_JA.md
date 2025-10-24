# Phase 4 継続実装計画: 残りのオプション拡張

**ドキュメント種別**: 継続実装計画・詳細仕様  
**フェーズ**: Phase 4.2-4.4 (Optional Enhancements)  
**ステータス**: 計画中  
**日付**: 2025-10-24  
**バージョン**: 1.0

---

## エグゼクティブサマリー

Phase 4.1（Property Validation System）の実装が完了しました。本ドキュメントでは、残りの**オプション拡張コンポーネント**の詳細実装計画と仕様を提供します。

**完了済み**:
- ✅ Phase 4.1: Property Validation System

**計画中（すべてオプション）**:
- ⏳ Phase 4.2: Multi-Property Optimization（多物性最適化）
- ⏳ Phase 4.3: Advanced Visualization（高度な可視化）
- ⏳ Phase 4.4: Performance Optimization（パフォーマンス最適化）

**重要**: これらのコンポーネントはすべて**オプション**です。Phase 4.1の完了により、システムは物性検証を含む完全な機能を提供しています。

---

## Phase 4.2: Multi-Property Optimization

### 概要

複数の物性を同時に最適化し、Pareto frontierを探索する機能を提供します。

### 目的

- 複数の物性制約を満たす結晶の生成
- 物性間のトレードオフの可視化
- Pareto最適解の探索

### 要件定義

#### 機能要件

**FR-P4-2.1: 制約条件の定義**
- 複数の物性制約を受け入れ
- 制約タイプ: 等号(`==`)、不等号(`>`, `<`, `>=`, `<=`)、範囲(`in`)
- 例: `bandgap > 2.0 AND melting_point < 200.0`

**FR-P4-2.2: 制約充足生成**
- 制約を満たす結晶を生成
- 制約範囲内での目標物性値のサンプリング
- 生成後の制約検証（Phase 4.1の予測器使用）

**FR-P4-2.3: Pareto frontier探索**
- 2つ以上の目的関数を最適化
- 非劣解（Pareto最適解）の探索
- グリッドサーチまたは進化アルゴリズム

**FR-P4-2.4: 結果の可視化と出力**
- Pareto frontierの2D/3D可視化
- 最適解の保存（CIF形式）
- トレードオフ分析レポート

#### 非機能要件

**NFR-P4-2.1: 探索効率**
- 1,000サンプルの探索: < 1時間（GPU使用時）

**NFR-P4-2.2: メモリ効率**
- メモリ使用量: < 8GB（バッチ処理）

**NFR-P4-2.3: スケーラビリティ**
- 最大5つの物性を同時最適化可能

### 詳細設計

#### 1. 制約条件パーサー

```python
class PropertyConstraint:
    """
    Single property constraint.
    
    Examples:
        - PropertyConstraint('bandgap', '>', 2.0)
        - PropertyConstraint('melting_point', 'in', (150.0, 200.0))
    """
    
    def __init__(
        self,
        property_name: str,
        operator: str,  # '>', '<', '>=', '<=', '==', 'in'
        value: Union[float, Tuple[float, float]]
    ):
        self.property_name = property_name
        self.operator = operator
        self.value = value
        
        # Validate operator
        valid_operators = ['>', '<', '>=', '<=', '==', 'in']
        if operator not in valid_operators:
            raise ValueError(f"Invalid operator: {operator}")
        
        # Validate value for 'in' operator
        if operator == 'in' and not isinstance(value, tuple):
            raise ValueError("'in' operator requires tuple (min, max)")
    
    def check(self, predicted_value: float) -> bool:
        """Check if predicted value satisfies constraint."""
        if self.operator == '>':
            return predicted_value > self.value
        elif self.operator == '<':
            return predicted_value < self.value
        elif self.operator == '>=':
            return predicted_value >= self.value
        elif self.operator == '<=':
            return predicted_value <= self.value
        elif self.operator == '==':
            return abs(predicted_value - self.value) < 1e-6
        elif self.operator == 'in':
            return self.value[0] <= predicted_value <= self.value[1]
    
    def sample_target_value(self) -> float:
        """Sample a target value that satisfies the constraint."""
        if self.operator == 'in':
            # Uniform sampling within range
            return np.random.uniform(self.value[0], self.value[1])
        elif self.operator == '>':
            # Sample above threshold with some margin
            return self.value + abs(self.value) * np.random.uniform(0.1, 0.5)
        elif self.operator == '<':
            # Sample below threshold with some margin
            return self.value - abs(self.value) * np.random.uniform(0.1, 0.5)
        elif self.operator == '>=':
            return self.value + abs(self.value) * np.random.uniform(0.0, 0.5)
        elif self.operator == '<=':
            return self.value - abs(self.value) * np.random.uniform(0.0, 0.5)
        elif self.operator == '==':
            return self.value
```

#### 2. 制約充足生成器

```python
class ConstrainedCrystalGenerator:
    """
    Generate crystals satisfying multiple property constraints.
    
    Uses rejection sampling with property predictor for validation.
    """
    
    def __init__(
        self,
        generator_model: nn.Module,
        property_predictor: PropertyPredictor,
        constraints: List[PropertyConstraint],
    ):
        self.generator_model = generator_model
        self.property_predictor = property_predictor
        self.constraints = constraints
    
    def generate_batch(
        self,
        molecule_id: str,
        batch_size: int = 32,
    ) -> List[Dict]:
        """
        Generate a batch of crystals attempting to satisfy constraints.
        
        Returns:
            crystals: List of generated crystals with predicted properties
        """
        # Sample target properties from constraint ranges
        target_properties = {}
        for constraint in self.constraints:
            target_properties[constraint.property_name] = constraint.sample_target_value()
        
        # Generate crystals with target properties
        crystals = generate_crystals(
            self.generator_model,
            molecule_id=molecule_id,
            target_properties=target_properties,
            n_samples=batch_size
        )
        
        # Predict properties
        for crystal in crystals:
            predictions = self.property_predictor(
                crystal['positions'],
                crystal['cell'],
                crystal['atomic_numbers']
            )
            crystal['predicted_properties'] = predictions
            
            # Check constraint satisfaction
            crystal['satisfies_constraints'] = all(
                constraint.check(predictions[constraint.property_name].item())
                for constraint in self.constraints
            )
        
        return crystals
    
    def generate_until_satisfied(
        self,
        molecule_id: str,
        n_samples: int,
        max_attempts: int = 10000,
    ) -> List[Dict]:
        """
        Generate crystals until n_samples satisfy all constraints.
        
        Uses rejection sampling.
        """
        satisfied_crystals = []
        attempts = 0
        
        pbar = tqdm(total=n_samples, desc="Generating constrained crystals")
        
        while len(satisfied_crystals) < n_samples and attempts < max_attempts:
            batch = self.generate_batch(molecule_id, batch_size=32)
            
            for crystal in batch:
                if crystal['satisfies_constraints']:
                    satisfied_crystals.append(crystal)
                    pbar.update(1)
                    
                    if len(satisfied_crystals) >= n_samples:
                        break
            
            attempts += len(batch)
        
        pbar.close()
        
        if len(satisfied_crystals) < n_samples:
            print(f"Warning: Only generated {len(satisfied_crystals)}/{n_samples} "
                  f"satisfying crystals after {attempts} attempts")
        
        return satisfied_crystals
```

#### 3. Pareto Frontier探索器

```python
class ParetoFrontierSearcher:
    """
    Search for Pareto frontier in multi-objective property space.
    """
    
    def __init__(
        self,
        generator_model: nn.Module,
        property_predictor: PropertyPredictor,
        objectives: List[str],
        minimize: List[bool] = None,
    ):
        self.generator_model = generator_model
        self.property_predictor = property_predictor
        self.objectives = objectives
        
        # Default: minimize all objectives
        if minimize is None:
            minimize = [True] * len(objectives)
        self.minimize = minimize
    
    def grid_search(
        self,
        molecule_id: str,
        n_points_per_dim: int = 10,
        property_ranges: Dict[str, Tuple[float, float]] = None,
    ) -> List[Dict]:
        """
        Grid search in property space.
        
        Args:
            molecule_id: Molecule identifier
            n_points_per_dim: Number of grid points per dimension
            property_ranges: Optional property ranges for grid
            
        Returns:
            crystals: All generated crystals with properties
        """
        # Create grid
        if property_ranges is None:
            # Use default ranges from training data
            property_ranges = self._get_default_ranges()
        
        grids = [
            np.linspace(property_ranges[obj][0], property_ranges[obj][1], n_points_per_dim)
            for obj in self.objectives
        ]
        
        # Generate crystals at each grid point
        crystals = []
        
        for grid_point in itertools.product(*grids):
            target_properties = {
                obj: val for obj, val in zip(self.objectives, grid_point)
            }
            
            # Generate one crystal per grid point
            crystal = generate_crystals(
                self.generator_model,
                molecule_id=molecule_id,
                target_properties=target_properties,
                n_samples=1
            )[0]
            
            # Predict properties
            predictions = self.property_predictor(
                crystal['positions'],
                crystal['cell'],
                crystal['atomic_numbers']
            )
            crystal['predicted_properties'] = predictions
            crystal['target_properties'] = target_properties
            
            crystals.append(crystal)
        
        return crystals
    
    def find_pareto_frontier(
        self,
        crystals: List[Dict]
    ) -> List[Dict]:
        """
        Find Pareto-optimal crystals from a set of candidates.
        
        A crystal is Pareto-optimal if no other crystal is better in all objectives.
        
        Args:
            crystals: List of crystals with predicted properties
            
        Returns:
            pareto_crystals: List of Pareto-optimal crystals
        """
        # Extract objective values
        objective_values = np.array([
            [crystal['predicted_properties'][obj].item() for obj in self.objectives]
            for crystal in crystals
        ])
        
        # Flip signs for maximization objectives
        for i, minimize in enumerate(self.minimize):
            if not minimize:
                objective_values[:, i] *= -1
        
        # Find Pareto frontier
        is_pareto = np.ones(len(crystals), dtype=bool)
        
        for i, vals_i in enumerate(objective_values):
            if is_pareto[i]:
                # Check if any other point dominates this one
                dominates = np.all(objective_values <= vals_i, axis=1) & \
                           np.any(objective_values < vals_i, axis=1)
                is_pareto[i] = not np.any(dominates)
        
        pareto_crystals = [crystal for i, crystal in enumerate(crystals) if is_pareto[i]]
        
        return pareto_crystals
    
    def visualize_pareto_frontier(
        self,
        crystals: List[Dict],
        pareto_crystals: List[Dict],
        output_path: str,
    ):
        """
        Visualize Pareto frontier (2D or 3D).
        
        Args:
            crystals: All generated crystals
            pareto_crystals: Pareto-optimal crystals
            output_path: Path to save plot
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # Extract objective values
        all_vals = np.array([
            [c['predicted_properties'][obj].item() for obj in self.objectives]
            for c in crystals
        ])
        
        pareto_vals = np.array([
            [c['predicted_properties'][obj].item() for obj in self.objectives]
            for c in pareto_crystals
        ])
        
        if len(self.objectives) == 2:
            # 2D plot
            plt.figure(figsize=(10, 8))
            plt.scatter(all_vals[:, 0], all_vals[:, 1], 
                       c='lightblue', alpha=0.5, label='All crystals')
            plt.scatter(pareto_vals[:, 0], pareto_vals[:, 1],
                       c='red', s=100, marker='*', label='Pareto frontier')
            plt.xlabel(self.objectives[0])
            plt.ylabel(self.objectives[1])
            plt.title('Pareto Frontier')
            plt.legend()
            plt.grid(True)
            
        elif len(self.objectives) == 3:
            # 3D plot
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(all_vals[:, 0], all_vals[:, 1], all_vals[:, 2],
                      c='lightblue', alpha=0.5, label='All crystals')
            ax.scatter(pareto_vals[:, 0], pareto_vals[:, 1], pareto_vals[:, 2],
                      c='red', s=100, marker='*', label='Pareto frontier')
            ax.set_xlabel(self.objectives[0])
            ax.set_ylabel(self.objectives[1])
            ax.set_zlabel(self.objectives[2])
            ax.set_title('Pareto Frontier')
            ax.legend()
        
        else:
            raise ValueError("Visualization only supports 2 or 3 objectives")
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Pareto frontier plot saved to {output_path}")
```

### 実装計画

**Week 1: 制約条件システム**
- `PropertyConstraint`クラスの実装
- 制約パーサー
- 制約サンプリング
- ユニットテスト

**Week 2: Pareto探索**
- `ParetoFrontierSearcher`の実装
- グリッドサーチ
- 非劣解の検出
- ユニットテスト

**Week 3: 可視化とCLI**
- 2D/3D可視化
- CLIスクリプト作成
- ドキュメント作成
- 統合テスト

### 成功基準

- 制約充足率 > 50%（適切な制約設定時）
- Pareto frontier検出の正確性 100%
- 1,000サンプル探索: < 1時間

---

## Phase 4.3: Advanced Visualization

### 概要

生成された結晶の物性分布と品質を可視化する高度なツールを提供します。

### 目的

- 物性分布の理解
- 生成品質の評価
- 学習進捗の追跡

### 要件定義

#### 機能要件

**FR-P4-3.1: 物性分布プロット**
- 生成結晶の物性ヒストグラム
- 訓練データとの比較
- 目標物性との比較

**FR-P4-3.2: 構造品質メトリクス**
- 結合長分布
- 結合角分布
- セルパラメータ分析

**FR-P4-3.3: 学習進捗追跡**
- Loss曲線
- 物性統計の時間変化
- サンプル品質の進化

**FR-P4-3.4: インタラクティブ可視化**
- HTMLレポート生成
- インタラクティブプロット（Plotly）
- ダッシュボード

#### 非機能要件

**NFR-P4-3.1: レポート生成速度**
- 100サンプルのレポート: < 30秒

**NFR-P4-3.2: 可読性**
- 研究発表・論文に使用可能な品質

### 詳細設計

#### 1. 物性分布アナライザー

```python
class PropertyDistributionAnalyzer:
    """
    Analyze and visualize property distributions.
    """
    
    def __init__(self, property_names: List[str]):
        self.property_names = property_names
    
    def plot_distributions(
        self,
        generated_properties: Dict[str, List[float]],
        training_properties: Dict[str, List[float]] = None,
        target_properties: Dict[str, float] = None,
        output_path: str = 'property_distributions.png',
    ):
        """
        Plot property distributions.
        
        Args:
            generated_properties: Properties of generated crystals
            training_properties: Properties from training data (optional)
            target_properties: Target property values (optional)
            output_path: Path to save plot
        """
        import matplotlib.pyplot as plt
        
        n_props = len(self.property_names)
        fig, axes = plt.subplots(1, n_props, figsize=(6*n_props, 5))
        
        if n_props == 1:
            axes = [axes]
        
        for ax, prop_name in zip(axes, self.property_names):
            # Generated distribution
            gen_vals = generated_properties[prop_name]
            ax.hist(gen_vals, bins=30, alpha=0.7, label='Generated', 
                   color='blue', density=True)
            
            # Training distribution
            if training_properties and prop_name in training_properties:
                train_vals = training_properties[prop_name]
                ax.hist(train_vals, bins=30, alpha=0.5, label='Training',
                       color='green', density=True)
            
            # Target value
            if target_properties and prop_name in target_properties:
                target_val = target_properties[prop_name]
                ax.axvline(target_val, color='red', linestyle='--',
                          linewidth=2, label='Target')
            
            ax.set_xlabel(prop_name)
            ax.set_ylabel('Density')
            ax.set_title(f'{prop_name} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Distribution plot saved to {output_path}")
    
    def compute_statistics(
        self,
        properties: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute statistics for each property.
        
        Returns:
            stats: Dictionary of statistics for each property
        """
        stats = {}
        
        for prop_name in self.property_names:
            vals = np.array(properties[prop_name])
            stats[prop_name] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
                'median': float(np.median(vals)),
                'q25': float(np.percentile(vals, 25)),
                'q75': float(np.percentile(vals, 75)),
            }
        
        return stats
```

#### 2. 構造品質アナライザー

```python
class StructureQualityAnalyzer:
    """
    Analyze structural quality of generated crystals.
    """
    
    def analyze_bond_lengths(
        self,
        crystals: List[Dict],
        output_path: str = 'bond_lengths.png',
    ):
        """
        Analyze and plot bond length distributions.
        """
        from ase import Atoms
        
        all_bond_lengths = []
        
        for crystal in crystals:
            # Convert to ASE Atoms
            atoms = Atoms(
                numbers=crystal['atomic_numbers'],
                positions=crystal['positions'],
                cell=crystal['cell'],
                pbc=True
            )
            
            # Compute neighbor list
            from ase.neighborlist import neighbor_list
            i, j, d = neighbor_list('ijd', atoms, cutoff=3.0)
            
            # Filter out self-interactions
            mask = i != j
            bond_lengths = d[mask]
            all_bond_lengths.extend(bond_lengths)
        
        # Plot
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.hist(all_bond_lengths, bins=50, alpha=0.7, color='blue')
        plt.xlabel('Bond Length (Å)')
        plt.ylabel('Count')
        plt.title('Bond Length Distribution')
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Bond length plot saved to {output_path}")
```

#### 3. HTMLレポート生成器

```python
class HTMLReportGenerator:
    """
    Generate comprehensive HTML reports with interactive plots.
    """
    
    def generate_report(
        self,
        crystals: List[Dict],
        output_path: str = 'report.html',
    ):
        """
        Generate HTML report with all visualizations.
        
        Uses Plotly for interactive plots.
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create HTML content
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Crystal Generation Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .plot {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Crystal Generation Report</h1>
            <p>Generated {n_crystals} crystals</p>
            
            <h2>Property Distributions</h2>
            <div id="property_dist" class="plot"></div>
            
            <h2>Structure Quality</h2>
            <div id="structure_quality" class="plot"></div>
            
            <script>
                {plot_scripts}
            </script>
        </body>
        </html>
        """
        
        # Generate plots...
        # (Implementation details)
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"HTML report saved to {output_path}")
```

### 実装計画

**Week 1: 物性可視化**
- 物性分布プロット
- 統計計算
- 比較プロット

**Week 2: 構造品質**
- 結合長・角度分析
- セルパラメータ分析
- 品質メトリクス

**Week 3: HTMLレポート**
- Plotly統合
- インタラクティブプロット
- ダッシュボード

### 成功基準

- レポート生成: < 30秒（100サンプル）
- 研究発表に使用可能な図の品質
- インタラクティブ機能の動作

---

## Phase 4.4: Performance Optimization

### 概要

生成速度とメモリ効率を最適化します。

### 目的

- 大規模生成の高速化
- メモリ使用量の削減
- スケーラビリティの向上

### 要件定義

#### 機能要件

**FR-P4-4.1: マルチGPU生成**
- 複数GPUへの生成分散
- 効率的なバッチ処理
- 結果の統合

**FR-P4-4.2: キャッシュ最適化**
- 条件付けベクトルのキャッシュ
- 分子特徴の再利用
- メモリ効率の向上

**FR-P4-4.3: 最適化サンプリング**
- 拡散ステップ数の削減（精度検証付き）
- 高速ODEソルバー
- 適応的ステップサイズ

#### 非機能要件

**NFR-P4-4.1: 生成速度**
- 2倍以上の高速化（マルチGPU）

**NFR-P4-4.2: メモリ効率**
- メモリ使用量の30%削減

### 詳細設計

#### 1. マルチGPU生成器

```python
class MultiGPUGenerator:
    """
    Distribute crystal generation across multiple GPUs.
    """
    
    def __init__(
        self,
        model: nn.Module,
        n_gpus: int = None,
    ):
        if n_gpus is None:
            n_gpus = torch.cuda.device_count()
        
        self.n_gpus = n_gpus
        
        # Replicate model to all GPUs
        self.models = [
            model.to(f'cuda:{i}')
            for i in range(n_gpus)
        ]
    
    def generate_parallel(
        self,
        molecule_id: str,
        target_properties: Dict[str, float],
        n_samples: int,
    ) -> List[Dict]:
        """
        Generate crystals in parallel across GPUs.
        """
        samples_per_gpu = n_samples // self.n_gpus
        
        # Create processes for each GPU
        with multiprocessing.Pool(self.n_gpus) as pool:
            results = pool.starmap(
                self._generate_on_gpu,
                [(gpu_id, molecule_id, target_properties, samples_per_gpu)
                 for gpu_id in range(self.n_gpus)]
            )
        
        # Concatenate results
        all_crystals = []
        for crystals in results:
            all_crystals.extend(crystals)
        
        return all_crystals
    
    def _generate_on_gpu(
        self,
        gpu_id: int,
        molecule_id: str,
        target_properties: Dict[str, float],
        n_samples: int,
    ) -> List[Dict]:
        """Generate on a specific GPU."""
        model = self.models[gpu_id]
        # Generation logic...
```

#### 2. 条件付けキャッシュ

```python
class ConditioningCache:
    """
    Cache conditioning vectors for reuse.
    """
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
    
    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable,
    ) -> torch.Tensor:
        """
        Get conditioning from cache or compute if not present.
        """
        if key not in self.cache:
            if len(self.cache) >= self.max_size:
                # Remove oldest entry
                self.cache.pop(next(iter(self.cache)))
            
            self.cache[key] = compute_fn()
        
        return self.cache[key]
```

### 実装計画

**Week 1: マルチGPU**
- プロセス並列化
- GPUメモリ管理
- 結果統合

**Week 2: キャッシュ最適化**
- 条件付けキャッシュ
- メモリプロファイリング
- 最適化

### 成功基準

- 2倍以上の高速化（マルチGPU使用時）
- メモリ使用量30%削減

---

## 実装優先順位

### 優先度評価

**Phase 4.2 (Multi-Property Optimization)**: 中
- 研究用途で有用
- 複雑な材料設計に対応
- 推定工数: 2週間

**Phase 4.3 (Advanced Visualization)**: 低
- ユーザー体験向上
- 論文発表に有用
- 推定工数: 1週間

**Phase 4.4 (Performance Optimization)**: 低
- 現在の性能で十分
- 大規模生成時のみ必要
- 推定工数: 2週間

### 推奨実装順序

1. **Phase 4.2を優先** - 多物性最適化は研究価値が高い
2. **Phase 4.3を次に** - 可視化は論文発表に役立つ
3. **Phase 4.4は必要時のみ** - 性能ボトルネックが確認されてから

---

## まとめ

### Phase 4ステータス

**完了**:
- ✅ Phase 4.1: Property Validation System

**計画中（すべてオプション）**:
- ⏳ Phase 4.2: Multi-Property Optimization
- ⏳ Phase 4.3: Advanced Visualization  
- ⏳ Phase 4.4: Performance Optimization

### 推奨事項

**研究利用の場合**:
1. Phase 4.1で物性検証を実施
2. 必要に応じてPhase 4.2で多物性最適化
3. 論文執筆時にPhase 4.3で可視化

**本番利用の場合**:
1. Phase 4.1で品質保証
2. Phase 4コンポーネントよりデプロイインフラを優先
3. スケール要件に応じてPhase 4.4を検討

**探索利用の場合**:
1. Phase 4.1で十分
2. 実際の使用でフィードバックを収集
3. ニーズに基づいて追加実装を決定

---

**ドキュメント作成日**: 2025-10-24  
**総推定工数**: Phase 4.2-4.4合計で5-6週間（1開発者）  
**ステータス**: Phase 4.1完了、Phase 4.2-4.4は計画段階  
**次のステップ**: ユーザーフィードバックに基づき実装判断
