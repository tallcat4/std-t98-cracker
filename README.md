# std-t98-cracker

STD-T98の`.burst`をオフラインで解析する開発用CLIです。
SPEARの無音プロファイルによる線形ソルバーと、旧std-t98-toolsのAI鍵探索を分離して使えます。
受信、音声生成、方式間の自動フォールバックはありません。

## Windowsで準備

Python 3.11以上と64-bit Cコンパイラ（MinGW/GCC）:

```powershell
.\build.ps1
python -m pip install -e .
# AIを使用する場合だけ
python -m pip install -e '.[ai]'
```

このPCではPython 3.14 / GCC 13.2で動作確認済み。49-bit入力の線形解析/指定鍵変換にはネイティブFECもTorchも不要です。
72-bit入力を扱う際だけFEC DLLが必要です。MSVC/LinuxでもCMakeでビルド可能:

```text
cmake -S . -B build
cmake --build build --config Release
```

DLLは`std_t98_cracker/`、`build/`、`build/Release/`の順で探します。`STD_T98_FEC`で明示できます。

## 操作は別々に実行

```powershell
python -m std_t98_cracker inspect capture.burst
python -m std_t98_cracker crack capture.burst --method linear --json capture.result.json
# receiverが--llr付きで保存した場合
python -m std_t98_cracker crack capture.burst --method linear --llr capture.llr
# AIは明示的に選ぶ
python -m std_t98_cracker crack capture.burst --method ai --threads 4
# 結果を確認した後、指定した鍵で49-bitファイルを作る
python -m std_t98_cracker descramble capture.burst --key 12345 -o clear
```

最後の出力は`clear.burst`。音声化は[std-t98-vocode](https://github.com/tallcat4/std-t98-vocode)で:

```powershell
python -m std_t98_vocode decode clear.burst -o clear.wav
```

受信は[std-t98-receiver](https://github.com/tallcat4/std-t98-receiver)の仕事です。
3者のバースト/ビット順は[FORMAT.md](FORMAT.md)で定義しています。`.ambe`にはバースト境界がないため解析入力には使いません。

## 結果の意味

- `linear`: 無音・無声・有声の制約を組み合わせ、バーストごとに候補鍵と証拠スコアを更新します。
  `--threshold`（既定20）と`--margin`（次点との差、既定6）で採用基準を指定します。スコアは正解確率ではありません。
  `--trace progress.jsonl`で逐次結果を保存できます。最初の判断と最終判断は矛盾により変わる場合があります。
  PTTごとのAPI・適用範囲は[LINEAR_SOLVER.md](LINEAR_SOLVER.md)、実測は[評価記録](experiments/README.md)。`0`は平文です。
- `ai`: FFNNで候補を絞り、hybridモデルで順位付けします。結果は`ai-candidate`であり正解保証ではありません。
  連続した訂正可能な5バースト以上が必要です。キー0/平文はこのAI実装では「候補なし」と区別しません。
- `--window`は状態をリセットするバースト数。linearは既定で入力全体を1セッションとするため、異なる鍵のPTTを混ぜないでください。AIの既定は10バーストです。
  JSONの`first_burst`は0始まり、`end_burst`は終端を含みません。
  時刻ではなく、保存ファイル中のレコード番号です。
- 終了コードは0=候補あり/変換成功、2=探索完了・候補なし、1=入力/依存/実行エラー。
- JSONと変換先ファイルは上書きしません。入力も変更しません。

指定鍵変換は訂正不能フレームを保持し、件数をstderrに表示します。その部分の音声データは信頼できません。
LLRを使った解析の後は、指定鍵変換にも同じ`--llr`を渡すと同じFEC方式になります。

## 秘話系列を追加する

[examples/sequence.py](examples/sequence.py)をコピーし、`generate(key)`だけを編集します。
0..32767の鍵から、バースト先頭でリセットされる196個の0/1（4×49、AMBE自然順）を返してください。

```powershell
python -m std_t98_cracker crack capture.burst --method linear --sequence my_sequence.py
python -m std_t98_cracker descramble capture.burst --key 12345 --sequence my_sequence.py -o clear
```

線形ソルバーはGF(2)上の線形性と十分なランクを必要とします。満たさない系列には明示的AI探索または指定鍵変換を使います。
`--sequence`はローカルPythonコードを実行します。任意長鍵や状態を呼間で引き継ぐ方式は今回の範囲外です。

## 開発・検証

```powershell
python -m unittest discover -s tests -v
```

実測無音フレームによる位相ごとの鍵復元、競合/破損の棄却、バイナリ形式、指定鍵変換、FEC入力を検証します。
ソースとモデルの由来は[PROVENANCE.md](PROVENANCE.md)。
