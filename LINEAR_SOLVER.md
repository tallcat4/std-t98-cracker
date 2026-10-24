# ソルバー誘導AIのAPIと判定基準

ソルバー誘導AI（Solver-guided AI）は、音声フレームの線形制約から候補を選び、hybridモデルで最終判定します。バーストごとに証拠を蓄積し、基準を満たすと系列番号と秘話コードを返します。

## 逐次API

```python
from std_t98_cracker.enhanced import HeavyScorer
from std_t98_cracker.guided_ai import GuidedAISolver
from std_t98_cracker.series_solver import SeriesSolver

scorer = HeavyScorer(device='auto')
solver = GuidedAISolver(SeriesSolver(), scorer)
session = solver.new_session(threshold=20, margin=6)

# PTT中、到着した各バーストに対して呼ぶ。
# burst49: FEC復号済みの(4, 49)の0/1配列（ThumbDV順）
# good4: 各フレームのFEC正常フラグ、形状(4,)
status = session.push(burst49, valid=good4)
if status['key'] is not None:
    series, key = status['series'], status['key']

session.reset()  # 次のPTTの前にリセット
```

`device='auto'`はCUDAが利用可能ならGPU、それ以外はCPUを選びます。CPUを指定する場合は`HeavyScorer(device='cpu', threads=1)`とします。モデルはPTTをまたいで再利用できます。

各`push()`は、その時点の証拠に基づく判断を返します。続けて入力すると、候補の競合や後続区間の判定によって`key`が`None`へ戻る場合もあります。呼び出し側で回答を採用し、その後も監視を続けられます。

## 処理と判定基準

1. 隣接するAMBEフレームでビットが変わりにくい性質から、LFSR初期状態に対する重み付きGF(2)制約を蓄積します。
2. Walsh–Hadamard変換で16系列×32768状態を評価し、上位5候補のうち線形スコアが`threshold`以上のものを選びます。
3. FEC正常な5連続バーストを1区間として、直近の重複しない最大3区間をhybridモデルで判定します。
4. 2区間なら両方、3区間なら2区間以上が平文判定で、平均スコアが正、かつ次点との差が`margin`以上なら採用します。次点のスコアが負、または次点がない場合は0と比較します。

判定に必要な最小入力は2区間、計10バースト＝音声0.8秒分です。FEC異常のフレームで線形制約の連続性を区切り、AI用の区間は全フレーム正常なバーストから作ります。同じ候補・同じ区間のAI判定はキャッシュします。

| 設定 | 既定値 | 意味 |
|---|---:|---|
| `threshold` | 20 | hybridへ渡す候補の線形スコア下限 |
| `margin` | 6 | hybridの平均スコアで必要な次点との差 |

線形スコアは、ビットの一致度から探索範囲の19 bit分を差し引いた証拠点です。hybridのスコアは`plain logit - encrypted logit`で、正なら平文判定です。どちらも正解確率ではありません。

LFSR初期状態から無線機の表示コードへの変換は内部で行います。秘話なしは全系列に共通するため、`series=0, key=0`にまとめます。

## 主な戻り値

| フィールド | 内容 |
|---|---|
| `series`, `key` | 採用した系列・表示コード。保留時は`None`、秘話なしは0 |
| `candidate_series`, `candidate` | hybridで首位の候補。評価前は`None` |
| `score`, `margin` | 首位の平均logit差、次点との差 |
| `candidates` | 評価済み候補と区間別のlogit差 |
| `linear` | 線形段階の候補・スコア |
| `verification` | 評価区間・候補選抜の閾値・AI評価数 |
| `bursts`, `traffic_seconds` | 入力バースト数と音声量（バースト数×0.08秒） |

最上位の`threshold`はhybridの平均スコア基準である0です。設定した線形スコア下限は`verification.linear_candidate_threshold`に入ります。

## 保存ファイルで確認する

```powershell
python -m std_t98_cracker crack capture.burst --series auto --method guided-ai --llr capture.llr --threshold 20 --margin 6 --trace progress.jsonl --json result.json
```

`--trace`は毎バーストの判断をJSONLに保存します。結果の`first_confirmation`は最初に採用した回答、`key`は入力全体を処理した時点の判断です。

既定では1入力を1PTTとして扱います。`--window N`を指定するとNバーストごとに状態をリセットします。`traffic_seconds`は保存された音声量で、計算時間や受信待ち時間とは別です。

[性能評価](experiments/BLIND_SOLVER_RESULTS.md) / [強化秘話の仕組み](README.md#alinco強化秘話の仕組み)
