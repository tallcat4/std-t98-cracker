# 評価データと再現手順

- [ソルバー誘導AIの性能評価](BLIND_SOLVER_RESULTS.md)：512試行の回答時間、F1、CPU性能。
- [強化秘話の実機検証](ENHANCED_RESULTS.md)：15系列のマスク・コード変換の確認結果。
- [実機試験の手順](ENHANCED_TEST.md)：収録、系列・鍵の解析、音声確認。

以下のコマンドは`std-t98-cracker`直下で実行します。音声・NPZ・個別ログの保存先は`build/`です。

## コーパス

ThumbDVのAMBE-3000Rで符号化した335発話、自然音声32.24分（96,722フレーム）を使用しました。

| 音声 | 調整用 | 評価用 | 分割 |
|---|---:|---:|---|
| JSUT basic5000 | 100発話 | 101発話 | 発話が重複しないよう分割、話者は同一 |
| LibriSpeech | 66発話 | 68発話 | dev-clean／test-clean、29人／33人の話者に重複なし |

PCMは8 kHz、実効レベルは−20・−26・−32 dBFSから選択。発話ごとにチップをリセットし、ECMODEを0または0x0800、RATETを34（49-bit ThumbDV形式）に設定しました。末尾に収録した1秒のゼロPCMは自然音声と区別して保存し、ソルバー誘導AIの評価には自然音声部分を使います。

チップのファームウェアは`V120.E100.XXXX.C106.G514.R009.B0010411.C0020208`です。

## ThumbDVで符号化する

[JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)と[LibriSpeech](https://www.openslr.org/12/)を用意します。チップ制御には別プロジェクト`neural-ambe2`の`neural_ambe.chip`・`neural_ambe.audio`を使います。測定時のドライバー版は`eeacbd2f4d3336a5f97bbf2c515584ac03d228fe`です。

```powershell
python -m pip install numpy scipy soundfile pyserial
python experiments/capture_chip.py --driver-root ../../neural-ambe2 --source-root ../../neural-ambe2/data/raw --port COM3 --minutes-per-group 8 --output build/chip-corpus
```

各NPZには入力PCM・AMBEビット列・条件・入力音声のSHA-256を保存します。同じ収録計画ならファイル単位で再開できます。

## ambe2-cで符号化する

元のNPZに含まれるPCMをソフトウェアエンコーダーへ渡します。

```powershell
python experiments/encode_corpus.py --source ../std-t98-vocode/ambe2-c-source --corpus build/chip-corpus --output build/software-corpus
```

測定にはambe2-c `70e32cbfc1cf5a08f3e9a3fbe41f6a9873cc4ca6`のdouble版・RATET 34を使用し、元のECMODEに合わせて背景雑音モードを設定しました。評価用169発話、自然音声969.72秒の符号化に7.12秒（約136倍速、起動・保存を含みビルドを除く）かかりました。

参照ベクトル4件の一致と、RATET 33からFEC復号した1,404フレームのRATET 34出力との一致を確認しています。

## ソルバー誘導AIを評価する

同じ区間・系列・鍵で2つのエンコーダーを比較し、図を作成します。

```powershell
python experiments/benchmark_guided_ai.py --chip build/chip-corpus --software build/software-corpus --output build/paired-eval
python experiments/plot_guided_ai.py build/paired-eval
```

単一のコーパスを使う場合：

```powershell
python experiments/evaluate_series_corpus.py build/chip-corpus --method guided-ai --json build/corpus-result.json
```

実録音を使う場合は、JSON配列に`input`、任意の`llr`、採点用の`series`と`key`を記載します。

```json
[
  {"input": "capture.burst", "llr": "capture.llr", "series": 5, "key": 1}
]
```

```powershell
python experiments/evaluate_series.py manifest.json --method guided-ai --json build/guided-result.json
python experiments/evaluate_series.py manifest.json --method guided-ai --device cpu --threads 1 --json build/cpu-result.json
```
