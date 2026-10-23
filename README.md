# std-t98-cracker

STD-T98の`.burst`から鍵を解析し、指定した鍵で秘話を解除するCLIです。線形ソルバーとAI探索を使えます。

## 主な機能

- **線形解析**：音声のビット制約から鍵候補を求め、バーストごとに証拠を蓄積します。スコアと次点との差で採用基準を調整できます。
- **AI探索**：学習済みモデルで鍵候補を評価します。線形解析で解けない入力にも、別コマンドで試せます。
- **指定鍵での解除**：既知の鍵を適用し、音声化に渡せる49-bitの`.burst`を生成します。
- **入力の確認とFEC**：`inspect`で形式・フレーム数を確認できます。72-bit入力の誤り訂正は、LLRによるsoft FECにも対応します。
- **解析結果の保存**：結果をJSONに、線形解析の各バーストの候補・スコアをJSONLに保存できます。
- **秘話系列の追加**：Pythonの系列生成関数を差し替えられます。受信側や音声化側の変更は不要です。

## セットアップ

Python 3.11以上と64-bit MinGW（GCC）が必要です。

```powershell
.\build.ps1
python -m pip install -e .
# AIを使う場合のみ
python -m pip install -e '.[ai]'
```

## 使い方

```powershell
python -m std_t98_cracker inspect capture.burst
python -m std_t98_cracker crack capture.burst --method linear --json result.json
python -m std_t98_cracker crack capture.burst --method ai
python -m std_t98_cracker descramble capture.burst --key 12345 -o clear
```

`12345`は使用する鍵に置き換えます。解除後の出力は`clear.burst`です。
LLRがあれば、解析と解除の両方に`--llr capture.llr`を追加します。
線形解析は1入力を1PTTとして扱い、`--window N`でNバーストごとに状態をリセットできます。
採用基準は`--threshold`（既定20）と`--margin`（既定6）、逐次記録は`--trace progress.jsonl`で指定します。
スコアは正解確率ではありません。リアルタイムにバーストを渡すAPIも用意しています。[閾値・逐次API](LINEAR_SOLVER.md) / [評価結果](experiments/README.md)

秘話系列の追加は[examples/sequence.py](examples/sequence.py)を編集し、`--sequence`で指定します。
強化秘話の候補比較・実機試験は[実験手順](experiments/ENHANCED_TEST.md)を参照してください。
受信は[std-t98-receiver](https://github.com/tallcat4/std-t98-receiver)、音声化は[std-t98-vocode](https://github.com/tallcat4/std-t98-vocode)を使います。
[ファイル形式](FORMAT.md) / [ソース・モデルの由来](PROVENANCE.md)
