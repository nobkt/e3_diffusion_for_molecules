# E3同変拡散モデル 完全詳細理論説明書

**Version 1.0 | 最終更新: 2025年10月24日**

---

## 目次

1. [理論的基礎](#理論的基礎)
2. [E(3)同変性の数学的定義](#e3同変性の数学的定義)
3. [拡散モデルの理論](#拡散モデルの理論)
4. [EGNNアーキテクチャ](#egnnアーキテクチャ)
5. [ノイズスケジュールと最適化](#ノイズスケジュールと最適化)
6. [条件付け機構の理論](#条件付け機構の理論)
7. [結晶拡張の理論](#結晶拡張の理論)
8. [損失関数の詳細](#損失関数の詳細)
9. [サンプリング理論](#サンプリング理論)
10. [数値安定性と実装](#数値安定性と実装)

---

## 理論的基礎

### はじめに

E3同変拡散モデル（E3 Equivariant Diffusion Model, EDM）は、3次元分子構造を生成するための生成モデルです。本モデルは以下の理論的原則に基づいています：

1. **E(3)同変性**: 3次元ユークリッド群の作用に対する同変性
2. **拡散過程**: スコアベースの生成モデリング
3. **物理的制約**: 化学的・物理的妥当性の保証
4. **厳密な数学的定式化**: 近似やフォールバックを使用しない

### 記号の定義

#### 基本記号

- $\mathcal{G} = (\mathcal{V}, \mathcal{E})$: 分子グラフ（頂点集合$\mathcal{V}$、辺集合$\mathcal{E}$）
- $N = |\mathcal{V}|$: 原子数（頂点数）
- $\mathbf{x}_i \in \mathbb{R}^3$: 原子$i$の3次元座標
- $h_i \in \mathbb{R}^{d_h}$: 原子$i$の特徴ベクトル
- $e_{ij} \in \mathbb{R}^{d_e}$: 辺$(i,j)$の特徴ベクトル

#### データ表現

分子は以下のタプルで表現されます：

$$z = (\mathbf{x}, h) = \{(\mathbf{x}_1, h_1), ..., (\mathbf{x}_N, h_N)\}$$

ここで：
- $\mathbf{x} = \{\mathbf{x}_1, ..., \mathbf{x}_N\} \in \mathbb{R}^{N \times 3}$: 座標行列
- $h = \{h_1, ..., h_N\} \in \mathbb{R}^{N \times d_h}$: 特徴行列

#### 群論的記号

- $G = E(3)$: 3次元ユークリッド群
- $R \in SO(3)$: 回転行列（3次元特殊直交群）
- $\mathbf{t} \in \mathbb{R}^3$: 平行移動ベクトル
- $g = (R, \mathbf{t}) \in E(3)$: E(3)群の要素

---

## E(3)同変性の数学的定義

### E(3)群の定義

3次元ユークリッド群$E(3)$は、3次元空間における等長変換の群です：

$$E(3) = SO(3) \ltimes \mathbb{R}^3$$

ここで$\ltimes$は半直積を表します。

### 群の作用

E(3)群の要素$g = (R, \mathbf{t})$は座標に以下のように作用します：

$$g \cdot \mathbf{x} = R\mathbf{x} + \mathbf{t}$$

### 同変性の定義

関数$f: \mathbb{R}^{N \times 3} \times \mathbb{R}^{N \times d_h} \rightarrow \mathbb{R}^{N \times 3} \times \mathbb{R}^{N \times d_h}$が**E(3)同変**であるとは、任意の$g = (R, \mathbf{t}) \in E(3)$に対して：

$$f(R\mathbf{x} + \mathbf{t}, h) = (Rf(\mathbf{x}, h)_{\mathbf{x}} + \mathbf{t}, f(\mathbf{x}, h)_h)$$

が成立することを意味します。ここで$f(\mathbf{x}, h)_{\mathbf{x}}$と$f(\mathbf{x}, h)_h$はそれぞれ座標部分と特徴部分を表します。

### 不変量の構成

E(3)不変な量（スカラー）の基本的な構成要素：

1. **距離**: $d_{ij} = \|\mathbf{x}_i - \mathbf{x}_j\|$
2. **内積**: $\langle \mathbf{v}, \mathbf{w} \rangle$ （ベクトル$\mathbf{v}, \mathbf{w}$の内積）
3. **ノルム**: $\|\mathbf{v}\|$

### 同変層の構成

E(3)同変な層を構成するための基本原理：

**定理1（同変層の構成）**:
座標の更新が以下の形式を取る場合、E(3)同変性が保たれる：

$$\mathbf{x}_i' = \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \phi_{ij} \cdot (\mathbf{x}_j - \mathbf{x}_i)$$

ここで$\phi_{ij} \in \mathbb{R}$はスカラー関数であり、E(3)不変な量のみに依存します。

**証明**:
回転$R$と平行移動$\mathbf{t}$を適用すると：

$$
\begin{align}
R\mathbf{x}_i' + \mathbf{t} &= R\mathbf{x}_i + \mathbf{t} + \sum_{j} \phi_{ij} \cdot R(\mathbf{x}_j - \mathbf{x}_i) \\
&= R\mathbf{x}_i + \mathbf{t} + \sum_{j} \phi_{ij} \cdot (R\mathbf{x}_j - R\mathbf{x}_i)
\end{align}
$$

これは$f(R\mathbf{x} + \mathbf{t})$に等しい。□

### 重心制約

並進自由度を除去するため、重心制約を課します：

$$\sum_{i=1}^N \mathbf{x}_i = \mathbf{0}$$

この制約下では、実効的な自由度は$3N - 3$となります。

---

## 拡散モデルの理論

### 前向き拡散過程

前向き拡散過程は、データに徐々にノイズを加える確率過程です：

$$q(z_t | z_0) = \mathcal{N}(z_t; \alpha_t z_0, \sigma_t^2 I)$$

ここで：
- $z_0 = (\mathbf{x}_0, h_0)$: クリーンなデータ
- $z_t = (\mathbf{x}_t, h_t)$: 時刻$t$のノイズ付きデータ
- $\alpha_t$: 信号のスケール係数
- $\sigma_t$: ノイズのスケール係数

### ノイズスケジュールの定義

信号対雑音比（SNR）を用いてノイズスケジュールを定義：

$$\text{SNR}(t) = \frac{\alpha_t^2}{\sigma_t^2}$$

対数SNRとして$\gamma(t)$を定義：

$$\gamma(t) = \log \text{SNR}(t) = \log \frac{\alpha_t^2}{\sigma_t^2}$$

### パラメータ化

スケール係数を$\gamma(t)$で表現：

$$
\begin{align}
\alpha_t &= \sqrt{\sigma(\gamma(t))} = \sqrt{\frac{1}{1 + e^{-\gamma(t)}}} \\
\sigma_t &= \sqrt{1 - \sigma(\gamma(t))} = \sqrt{\frac{e^{-\gamma(t)}}{1 + e^{-\gamma(t)}}}
\end{align}
$$

ここで$\sigma(x) = \frac{1}{1 + e^{-x}}$はシグモイド関数です。

### 逆向き拡散過程

生成過程は逆向き拡散として定式化されます：

$$p_\theta(z_{t-\Delta t} | z_t) = \mathcal{N}(z_{t-\Delta t}; \mu_\theta(z_t, t), \Sigma_\theta(z_t, t))$$

### スコア関数

スコア関数は対数確率の勾配として定義：

$$s_\theta(z_t, t) = \nabla_{z_t} \log p(z_t)$$

### デノイジング目的

モデルは以下の目的でノイズを予測するように学習：

$$\epsilon_\theta(z_t, t) \approx \epsilon \sim \mathcal{N}(0, I)$$

ここで$z_t = \alpha_t z_0 + \sigma_t \epsilon$。

### 変分下界（ELBO）

変分下界を最大化することで学習を行います：

$$
\begin{align}
\log p(z_0) &\geq \mathbb{E}_{q(z_1|z_0)}[\log p(z_0|z_1)] \\
&\quad - D_{KL}(q(z_1|z_0) \| p(z_1)) \\
&\quad - \mathbb{E}_{q(z_t,z_0)}\left[\int_0^1 \frac{1}{2\sigma_t^2}\|\epsilon_\theta(z_t, t) - \epsilon\|^2 dt\right]
\end{align}
$$

### KLダイバージェンス

ガウス分布間のKLダイバージェンスは解析的に計算可能：

$$
\begin{align}
D_{KL}(\mathcal{N}(\mu_q, \Sigma_q) \| \mathcal{N}(\mu_p, \Sigma_p)) = \\
\frac{1}{2}\left[\log\frac{|\Sigma_p|}{|\Sigma_q|} - d + \text{tr}(\Sigma_p^{-1}\Sigma_q) + (\mu_p - \mu_q)^T\Sigma_p^{-1}(\mu_p - \mu_q)\right]
\end{align}
$$

簡略化された形式（対角共分散の場合）：

$$D_{KL} = \sum_i \left[\log\frac{\sigma_{p,i}}{\sigma_{q,i}} + \frac{\sigma_{q,i}^2 + (\mu_{p,i} - \mu_{q,i})^2}{2\sigma_{p,i}^2} - \frac{1}{2}\right]$$

---

## EGNNアーキテクチャ

### EGNN層の定義

E(n)同変グラフニューラルネットワーク（EGNN）層は以下のように定義されます：

#### メッセージパッシング

各エッジ$(i,j)$に対してメッセージを計算：

$$m_{ij} = \phi_e(h_i, h_j, d_{ij}^2, e_{ij})$$

ここで：
- $\phi_e$: エッジ更新関数（MLP）
- $d_{ij} = \|\mathbf{x}_i - \mathbf{x}_j\|$: ユークリッド距離
- $e_{ij}$: エッジ特徴

#### ノード特徴の更新

$$h_i' = \phi_h\left(h_i, \sum_{j \in \mathcal{N}(i)} m_{ij}\right)$$

ここで$\phi_h$はノード更新関数（MLP）。

#### 座標の更新（E(3)同変）

$$\mathbf{x}_i' = \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \phi_x(m_{ij}) \cdot \frac{\mathbf{x}_i - \mathbf{x}_j}{d_{ij} + \epsilon}$$

ここで：
- $\phi_x$: 座標更新のための重み関数（MLP → スカラー）
- $\epsilon$: 数値安定性のための小さな定数（例：$10^{-8}$）

### アテンション機構

EGNNにアテンション機構を組み込む場合：

$$\alpha_{ij} = \frac{\exp(\phi_a(h_i, h_j, d_{ij}^2))}{\sum_{k \in \mathcal{N}(i)} \exp(\phi_a(h_i, h_k, d_{ik}^2))}$$

メッセージにアテンション重みを適用：

$$m_{ij}' = \alpha_{ij} \cdot m_{ij}$$

### 完全なEGNN層

EGNN層の完全な定義：

$$
\begin{align}
m_{ij} &= \phi_e([h_i, h_j, d_{ij}^2, e_{ij}]) \\
\mathbf{x}_i' &= \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \phi_x(m_{ij}) \cdot \frac{\mathbf{x}_i - \mathbf{x}_j}{d_{ij} + \epsilon} \\
m_i &= \sum_{j \in \mathcal{N}(i)} m_{ij} \\
h_i' &= \phi_h([h_i, m_i])
\end{align}
$$

ここで$[·, ·]$は連結演算を表します。

### E(3)同変性の証明

**定理2（EGNN層の同変性）**:
上記のEGNN層はE(3)同変である。

**証明**:
回転$R$と平行移動$\mathbf{t}$を適用した場合を考えます。

1. **距離の不変性**:
   $$d_{ij}' = \|R\mathbf{x}_i + \mathbf{t} - (R\mathbf{x}_j + \mathbf{t})\| = \|R(\mathbf{x}_i - \mathbf{x}_j)\| = \|\mathbf{x}_i - \mathbf{x}_j\| = d_{ij}$$

2. **メッセージの不変性**:
   $m_{ij}$は$d_{ij}^2$に依存するため、$m_{ij}' = m_{ij}$

3. **座標更新の同変性**:
   $$
   \begin{align}
   (R\mathbf{x}_i + \mathbf{t})' &= R\mathbf{x}_i + \mathbf{t} + \sum_j \phi_x(m_{ij}) \cdot \frac{R\mathbf{x}_i - R\mathbf{x}_j}{d_{ij}} \\
   &= R\mathbf{x}_i + \mathbf{t} + R\sum_j \phi_x(m_{ij}) \cdot \frac{\mathbf{x}_i - \mathbf{x}_j}{d_{ij}} \\
   &= R\mathbf{x}_i' + \mathbf{t}
   \end{align}
   $$

したがって、EGNN層はE(3)同変です。□

### 多層EGNN

$L$層のEGNNを積み重ねた場合も同変性は保たれます：

$$f = f_L \circ f_{L-1} \circ ... \circ f_1$$

各$f_l$がE(3)同変であれば、合成関数$f$もE(3)同変です。

---

## ノイズスケジュールと最適化

### 多項式ノイズスケジュール

多項式ノイズスケジュール：

$$\bar{\alpha}_t^2 = (1 - (t/T)^p)^2$$

ここで$p$は多項式の次数（推奨値：2または3）。

### コサインノイズスケジュール

コサインスケジュール：

$$\bar{\alpha}_t = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$$

ここで$s$はオフセットパラメータ（推奨値：0.008）。

### ガンマ関数

対数SNR関数$\gamma(t)$：

$$\gamma(t) = -\log\left(\frac{1 - \bar{\alpha}_t^2}{\bar{\alpha}_t^2}\right)$$

### 数値精度の制御

数値的安定性のため、$\bar{\alpha}_t^2$をクリップ：

$$\bar{\alpha}_t^2 \leftarrow \max(\epsilon_{\text{min}}, \min(1 - \epsilon_{\text{min}}, \bar{\alpha}_t^2))$$

推奨値：$\epsilon_{\text{min}} = 10^{-5}$

### SNRの連続性

離散時間ステップ間のSNRの連続性を保証：

$$\alpha_t / \alpha_{t-1} \geq \epsilon_{\text{clip}}$$

推奨値：$\epsilon_{\text{clip}} = 0.001$

### 最適なノイズスケジュールの選択

異なるタスクに対する推奨スケジュール：

1. **小分子（QM9）**: Polynomial (power=2)
2. **大分子（GEOM）**: Polynomial (power=3)
3. **結晶構造**: Cosine

---

## 条件付け機構の理論

### 条件付き拡散の定式化

条件付き生成では、目標性質$y$を与えて分子を生成：

$$p_\theta(z_0 | y) \propto p_\theta(z_0) \cdot p(y | z_0)$$

### コンテキストベクトル

性質$y$をコンテキストベクトル$c$にエンコード：

$$c = \phi_{\text{context}}(y)$$

### 条件付きスコア

条件付きスコア関数：

$$s_\theta(z_t, t, c) = \nabla_{z_t} \log p(z_t | c)$$

### 正規化

性質値を正規化して学習を安定化：

$$y_{\text{norm}} = \frac{y - \mu_y}{\sigma_y}$$

ここで$\mu_y$と$\sigma_y$は学習データから計算される平均と標準偏差です。

### Mean Absolute Deviation（MAD）正規化

より頑健な正規化手法：

$$y_{\text{norm}} = \frac{y - \text{median}(y)}{\text{MAD}(y)}$$

ここで：

$$\text{MAD}(y) = \text{median}(|y_i - \text{median}(y)|)$$

### 複数性質の条件付け

複数の性質$\{y_1, y_2, ..., y_k\}$で条件付ける場合：

$$c = [\phi_1(y_1), \phi_2(y_2), ..., \phi_k(y_k)]$$

コンテキストベクトルを連結して使用。

### 厳密な条件付け

目標値との厳密な一致を保証する方法：

1. **ガイダンススケール**を使用：
   $$s'(z_t, t, c) = s(z_t, t) + w \cdot (s(z_t, t, c) - s(z_t, t))$$

2. **射影法**：各ステップで制約を満たすように射影

### Classifier-Free Guidance

分類器なしガイダンス：

$$\epsilon_\theta(z_t, c, t) = \epsilon_\theta(z_t, \emptyset, t) + w \cdot (\epsilon_\theta(z_t, c, t) - \epsilon_\theta(z_t, \emptyset, t))$$

ここで$w$はガイダンスの強度（推奨値：1.0〜7.0）。

---

## 結晶拡張の理論

### 周期境界条件（PBC）

結晶構造では周期境界条件を適用：

$$\mathbf{x}_{\text{periodic}} = \mathbf{x} + n_1\mathbf{a} + n_2\mathbf{b} + n_3\mathbf{c}$$

ここで$\mathbf{a}, \mathbf{b}, \mathbf{c}$は格子ベクトル、$n_1, n_2, n_3 \in \mathbb{Z}$。

### 最小イメージ規約

最短距離を計算する最小イメージ規約：

$$d_{ij} = \min_{n_1,n_2,n_3} \|\mathbf{x}_i - \mathbf{x}_j - n_1\mathbf{a} - n_2\mathbf{b} - n_3\mathbf{c}\|$$

### 格子ベクトルの拡散

格子パラメータも拡散過程に含めます：

$$\mathbf{L} = [\mathbf{a}, \mathbf{b}, \mathbf{c}] \in \mathbb{R}^{3 \times 3}$$

拡散過程：

$$q(\mathbf{L}_t | \mathbf{L}_0) = \mathcal{N}(\mathbf{L}_t; \alpha_t \mathbf{L}_0, \sigma_t^2 I)$$

### 空間群対称性

空間群$G$の対称操作$g = (W, \mathbf{w})$：

$$\mathbf{x}' = W\mathbf{x} + \mathbf{w} \pmod{1}$$

ここで$W \in O(3)$は点群操作、$\mathbf{w}$は並進ベクトル（分数座標）。

### 空間群の埋め込み

空間群番号$s \in \{1, ..., 230\}$を埋め込み：

$$e_s = \text{Embedding}(s) \in \mathbb{R}^{d_s}$$

### 密度条件付け

密度$\rho$を条件として使用：

$$\rho = \frac{M}{V}$$

ここで$M$は分子量、$V = |\det(\mathbf{L})|$は単位格子体積。

### Wyckoff位置

対称性を考慮した原子位置の制約：

$$\mathbf{x}_i \in \text{Wyckoff}(G, \text{multiplicity})$$

### 結晶品質メトリクス

1. **空間群一致度**:
   $$\text{SG-Accuracy} = \frac{\#(\text{detected} = \text{target})}{\#\text{total}}$$

2. **密度誤差**:
   $$\text{Density-Error} = \frac{|\rho_{\text{gen}} - \rho_{\text{target}}|}{\rho_{\text{target}}}$$

3. **対称性スコア**:
   $$\text{Sym-Score} = \frac{\sum_g \mathbb{1}[d(g\mathbf{x}, \mathbf{x}) < \epsilon]}{|G|}$$

---

## 損失関数の詳細

### 基本的な拡散損失

L2損失：

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, z_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(z_t, t)\|^2\right]$$

### 重み付き損失

時刻依存の重み付け：

$$\mathcal{L}_{\text{weighted}} = \mathbb{E}_{t, z_0, \epsilon}\left[w(t)\|\epsilon - \epsilon_\theta(z_t, t)\|^2\right]$$

推奨される重み関数：

$$w(t) = \frac{1}{\sigma_t^2}$$

### VLB損失

変分下界（Variational Lower Bound）：

$$
\begin{align}
\mathcal{L}_{\text{VLB}} = &\mathbb{E}_{q(z_1|z_0)}[-\log p_\theta(z_0|z_1)] \\
&+ D_{KL}(q(z_T|z_0) \| p(z_T)) \\
&+ \sum_{t=2}^T \mathbb{E}_{q(z_t|z_0)}[D_{KL}(q(z_{t-1}|z_t, z_0) \| p_\theta(z_{t-1}|z_t))]
\end{align}
$$

### ハイブリッド損失

実際の学習では、単純損失とVLB損失を組み合わせ：

$$\mathcal{L} = \mathcal{L}_{\text{simple}} + \lambda \mathcal{L}_{\text{VLB}}$$

推奨値：$\lambda = 0.001$

### 座標・特徴の分離損失

座標と特徴を別々に扱う：

$$\mathcal{L} = \mathcal{L}_{\mathbf{x}} + \lambda_h \mathcal{L}_h$$

ここで：
- $\mathcal{L}_{\mathbf{x}}$: 座標の損失
- $\mathcal{L}_h$: 特徴（原子タイプ等）の損失
- $\lambda_h$: バランス係数（推奨値：1.0〜4.0）

### 正規化項

1. **重心制約**:
   $$\mathcal{L}_{\text{COM}} = \|\sum_i \mathbf{x}_i\|^2$$

2. **距離制約**:
   $$\mathcal{L}_{\text{dist}} = \sum_{i,j} \max(0, r_{\text{min}} - \|\mathbf{x}_i - \mathbf{x}_j\|)^2$$

完全な損失関数：

$$\mathcal{L}_{\text{total}} = \mathcal{L} + \lambda_{\text{COM}}\mathcal{L}_{\text{COM}} + \lambda_{\text{dist}}\mathcal{L}_{\text{dist}}$$

---

## サンプリング理論

### DDPM サンプリング

Denoising Diffusion Probabilistic Model のサンプリング：

$$z_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(z_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}}\epsilon_\theta(z_t, t)\right) + \sigma_t \epsilon$$

ここで$\epsilon \sim \mathcal{N}(0, I)$。

### DDIM サンプリング

Denoising Diffusion Implicit Model（決定的サンプリング）：

$$z_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\left(\frac{z_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(z_t, t)}{\sqrt{\bar{\alpha}_t}}\right) + \sqrt{1-\bar{\alpha}_{t-1}}\epsilon_\theta(z_t, t)$$

### スキップステップサンプリング

計算効率のため、ステップをスキップ：

$$z_{t-k} = \text{DDIM-Step}(z_t, t, t-k, \epsilon_\theta)$$

典型的な設定：$T=1000$で学習、$T'=100$でサンプリング

### アニーリングサンプリング

温度パラメータ$\tau$を使用：

$$z_{t-1} \sim \mathcal{N}(\mu_\theta(z_t, t), \tau^2 \Sigma_\theta(z_t, t))$$

$\tau$を時間とともに減少させる：

$$\tau(t) = \tau_{\max} \cdot \left(\frac{\tau_{\min}}{\tau_{\max}}\right)^{t/T}$$

### 条件付きサンプリング

条件$c$を与えたサンプリング：

$$z_{t-1} = \mu_\theta(z_t, c, t) + \sigma_t \epsilon$$

### 自己一貫性チェック

生成された分子の自己一貫性を確認：

1. 化学的妥当性
2. E(3)同変性（回転・平行移動テスト）
3. 目標性質との一致（条件付き生成の場合）

---

## 数値安定性と実装

### 浮動小数点精度

#### 指数関数の安定化

```python
def stable_exp(x):
    # exp(x) の安定版
    return torch.exp(torch.clamp(x, min=-20, max=20))
```

#### 対数の安定化

```python
def stable_log(x, eps=1e-8):
    # log(x) の安定版
    return torch.log(torch.clamp(x, min=eps))
```

#### expm1 と log1p

小さな値に対する安定な計算：

```python
# exp(x) - 1 の安定版
y = torch.expm1(x)

# log(1 + x) の安定版
y = torch.log1p(x)
```

### グラデーションクリッピング

勾配爆発を防ぐ：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 正規化の数値安定性

#### レイヤー正規化

```python
def stable_layer_norm(x, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + eps)
```

#### バッチ正規化

結晶生成では特に重要：

```python
def stable_batch_norm(x, eps=1e-5):
    mean = x.mean(dim=0, keepdim=True)
    var = x.var(dim=0, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + eps)
```

### 距離計算の安定性

```python
def stable_distance(x_i, x_j, eps=1e-8):
    # ユークリッド距離の安定な計算
    diff = x_i - x_j
    dist_sq = (diff ** 2).sum(dim=-1)
    return torch.sqrt(dist_sq + eps)
```

### ソフトマックスの安定性

```python
def stable_softmax(x, dim=-1):
    # 最大値を引いて安定化
    x_max = x.max(dim=dim, keepdim=True)[0]
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)
```

### 行列演算の安定性

#### 行列の逆行列

```python
def stable_inverse(A, eps=1e-6):
    # 正則化を加えた逆行列
    n = A.shape[-1]
    A_reg = A + eps * torch.eye(n, device=A.device)
    return torch.inverse(A_reg)
```

#### 対称行列の固有値分解

```python
def stable_eigh(A, eps=1e-8):
    # 対称行列の固有値・固有ベクトル
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    eigenvalues = torch.clamp(eigenvalues, min=eps)
    return eigenvalues, eigenvectors
```

### 数値誤差の累積

長い拡散ステップでの誤差累積を防ぐ：

1. **倍精度の使用**（必要に応じて）:
   ```python
   model = model.double()  # float64
   ```

2. **定期的な正規化**:
   ```python
   # 重心を0に戻す
   x = x - x.mean(dim=0, keepdim=True)
   ```

3. **スケール制御**:
   ```python
   # 座標のスケールを制御
   scale = torch.sqrt((x ** 2).sum() / (N * 3))
   x = x / scale
   ```

### メモリ管理

#### 勾配チェックポイント

```python
from torch.utils.checkpoint import checkpoint

def forward_with_checkpoint(layer, x):
    return checkpoint(layer, x)
```

#### インプレース操作

メモリ効率の改善：

```python
# 通常の操作
y = x + 1

# インプレース操作
x.add_(1)
```

### 並列化の考慮

#### データ並列

```python
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
```

#### 分散データ並列

```python
from torch.nn.parallel import DistributedDataParallel as DDP
model = DDP(model, device_ids=[local_rank])
```

---

## 理論的保証

### 定理3（収束性）

適切な条件下で、拡散モデルの学習は真の分布に収束します。

**条件**:
1. ノイズスケジュールが連続的
2. ニューラルネットワークが十分な表現力を持つ
3. 学習データが十分に多い

### 定理4（E(3)不変性の保存）

E(3)同変な操作のみを使用する場合、生成される分子はE(3)不変な分布から抽出されます。

### 定理5（条件付き生成の一貫性）

条件付き生成において、目標性質$y$を与えたとき、生成される分子の性質は$y$に収束します（適切なガイダンスの下）。

---

## 実装上の注意点

### フォールバックの禁止

本システムでは、理論的に不健全なフォールバック処理を**一切使用しません**：

- ❌ 距離がゼロの場合のデフォルト値
- ❌ NaN/Infが発生した場合の無視
- ❌ 制約違反の黙認
- ❌ 近似的な対称性操作

代わりに、以下のアプローチを使用：

- ✅ 数値的に安定な計算（eps項の追加）
- ✅ 明示的なエラーハンドリング
- ✅ 入力の事前検証
- ✅ 厳密な制約の強制

### デバッグとテスト

#### 同変性テスト

```python
def test_equivariance(model, x, h):
    # ランダムな回転と平行移動
    R = random_rotation_matrix()
    t = torch.randn(3)
    
    # 変換前の出力
    out1 = model(x, h)
    
    # 変換後の入力
    x_transformed = (R @ x.T).T + t
    
    # 変換後の出力
    out2 = model(x_transformed, h)
    
    # 同変性の確認
    out1_transformed = (R @ out1.T).T + t
    assert torch.allclose(out1_transformed, out2, atol=1e-5)
```

#### 不変性テスト

```python
def test_invariance(model, x, h):
    # ランダムな変換
    R = random_rotation_matrix()
    t = torch.randn(3)
    
    # スカラー出力の不変性
    scalar1 = model.compute_scalar(x, h)
    scalar2 = model.compute_scalar((R @ x.T).T + t, h)
    
    assert torch.allclose(scalar1, scalar2, atol=1e-5)
```

---

## まとめ

本理論説明書では、E3同変拡散モデルの完全な数学的基礎を解説しました：

1. **E(3)同変性**: 厳密な数学的定義と証明
2. **拡散過程**: 前向き・逆向き過程の完全な定式化
3. **EGNN**: E(3)同変な層の構成と証明
4. **ノイズスケジュール**: 最適化手法と数値安定性
5. **条件付け**: 理論的基礎と実装方法
6. **結晶拡張**: 周期境界条件と空間群対称性
7. **損失関数**: 各種損失の定式化
8. **サンプリング**: 効率的で理論的に健全な方法
9. **数値安定性**: 実装における重要な考慮事項

全ての操作は理論的に健全であり、フォールバックや近似を一切使用していません。

---

**参考文献**

1. Hoogeboom, E., et al. (2022). "Equivariant Diffusion for Molecule Generation in 3D." ICML 2022.
2. Satorras, V. G., et al. (2021). "E(n) Equivariant Graph Neural Networks." ICML 2021.
3. Ho, J., et al. (2020). "Denoising Diffusion Probabilistic Models." NeurIPS 2020.
4. Song, Y., et al. (2021). "Score-Based Generative Modeling through Stochastic Differential Equations." ICLR 2021.
5. Bronstein, M. M., et al. (2021). "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges." arXiv:2104.13478.

---

**著作権とライセンス**

本文書はMITライセンスの下で公開されています。

**最終更新**: 2025年10月24日  
**バージョン**: 1.0  
**言語**: 日本語
