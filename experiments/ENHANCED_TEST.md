# 強化秘話の実機試験

`experiment/enhanced-scramble` の実験用コマンドです。通常のCLIは変更していません。
候補番号0は標準系列、1〜16は比較用の系列です。無線機の強化秘話番号1〜15との対応は未確定です。
[準備試験の結果・GPUベンチ](ENHANCED_RESULTS.md)

## 1回の試験

CH1で設定を記録し、自然な発話を10〜20秒送信します。1ファイルは1回のPTT・1設定にします。
receiverで保存した `.burst` と `.llr` を使い、cracker直下で実行します。

```powershell
# 無線機の設定が「5-00001」の場合。17系列すべてを既知の鍵00001で比較。
python -m std_t98_cracker.enhanced probe capture.burst --llr capture.llr --setting 5-00001 --json build/enhanced/5-00001-a.json
# 合わない場合、17系列それぞれの全非ゼロ初期状態を探索。
python -m std_t98_cracker.enhanced probe capture.burst --llr capture.llr --setting 5-00001 --all-keys --json build/enhanced/5-00001-all.json
```

判定は **FP32のhybridモデルのみ**。既定ではFEC正常の5連続バーストを3組（計1.2秒）使い、
全候補を全3区間で評価します。軽量モデルによる足切り、平文判定での早期終了はありません。
`--blocks 5` で区間数、`--block-starts 10,30,50` で元ファイル上の開始バースト位置を指定できます。
失敗したFECの前後はつなぎません。既定の先頭区間が無音や不安定なら、別の発話区間でも再試験します。

JSONには入力・モデルのハッシュ、候補ごとのスコア、鍵、初期状態、平文の対照スコアを保存します。
スコアは `plain logit - encrypted logit`（正なら平文判定）で、正解確率ではありません。
順位は各区間の最小スコア、次に平均スコアで決まります。入力と既存出力は上書きしません。

## 対応表の確認

最初に一致した設定で別の発話をもう一度試します。その後、全15系列について
`00001`・`12345`・`32767` の3鍵を試し、設定ごとに独立した送信を2回記録します（90録音）。
秘話OFFと標準秘話も対照として保存します。

```powershell
$reports = (Get-ChildItem build/enhanced/*-a.json,build/enhanced/*-b.json).FullName
python -m std_t98_cracker.enhanced summary @reports --json build/enhanced/coverage.json
```

集計は15系列の欠測・鍵の種類・録音数・候補の食い違いを示します。AIの順位だけで対応を確定しません。
複数鍵で同じ候補系列が再現し、復元鍵が設定値に一致し、実際の発話が復元されることを確認します。
全鍵探索で別の鍵が出た場合は、系列の候補にはなりますが、表示鍵とオフセットの対応は未確認です。

## 選んだ候補で音声を確認

```powershell
# NとKEYをレポートで選んだ候補番号・鍵に置き換える
python -m std_t98_cracker.enhanced plugin --family N -o build/enhanced/selected.py
python -m std_t98_cracker descramble capture.burst --llr capture.llr --key KEY --sequence build/enhanced/selected.py -o build/enhanced/clear
python -m std_t98_vocode decode build/enhanced/clear.burst -o build/enhanced/clear.wav
```

追加仮説は `--initialization state`（鍵を初期状態として扱う）、`--skip N`（系列の先頭を捨てる）、
`--reset frame`（各49-bitフレームでリセット）です。probeとpluginの両方に同じ条件を指定します。
全初期状態を探索すれば、加算オフセットや系列の位相は状態の並び替えになります。
既知の表示鍵との対応確認には、既知鍵での試験が必要です。

## 性能と合成データでの確認

```powershell
python -m unittest discover -s tests
python experiments/validate_enhanced.py --corpus build/chip-corpus-20261023 --output build/enhanced-validation --all-keys
python -m std_t98_cracker.enhanced benchmark --seconds 60 --json build/enhanced-benchmark.json
```

ベンチはバッチサイズを比較後、PN生成・転送・XOR・重いモデルの判定・スコア回収まで測ります。
GPU使用率と電力も記録し、2^30候補を1区間／3区間で判定する時間を外挿します。
録音・FEC・ファイルI/Oを含まず、実際に2^30候補を完走した測定ではありません。

## 任意マスクとオフセットの全探索

候補17系列以外も試す場合は、別コマンドで15-bitフィードバックマスク全32,768通りを探索します。
各マスクの初期状態は1〜32767すべてを試し、既知の表示鍵から加算オフセットを逆算します。
計1,073,709,056組です。オフセットはmod 32767なので、0と32767は同じ値です。
状態0は平文の対照であり、非ゼロ表示鍵のこのオフセット規則では生じません。

```powershell
# 足切りなし。重いモデルで全候補・全3区間を評価。
python -m std_t98_cracker.enhanced_sweep capture.burst --llr capture.llr --known-key 1 --output build/mask-search
# 保守的な足切りを明示的に使う場合（最終判定は同じ重いモデル）。
python -m std_t98_cracker.enhanced_sweep capture.burst --llr capture.llr --known-key 1 --prefilter-margin -10 --output build/mask-filtered
# 中断した探索は同じ引数に --resume を追加。完了済み128マスク単位から再開。
```

`--rank-bound` を追加すると、途中の最小スコアが既存上位20件を下回った候補について、残りの区間を省略します。
最終的な最小スコアは途中の最小値を超えないため、この省略で上位候補の順位は変わりません。
同点は省略せず、平均スコアまで比較します。既存チェックポイントにこのオプションを追加して再開できます。

足切りは、3区間の**全部**でFFNNの平文スコアが−10未満のときだけ行います。
初回コーパス測定では調整用729組・評価用726組の自然音声をすべて残しました。
RF誤りや未測定の音声を含む入力での保証ではありません。区間数を変えた場合も別途検証が必要です。
足切りありの処理時間は残った候補数に依存します。すべて残る最悪ケースでは高速化しません。

合成試験はランダムな最大周期マスク・オフセット・鍵・評価用発話を選びます。
短周期による識別不能を避けるため、生成側のマスクだけは最大周期を条件とします。
探索側にはマスク・オフセットを渡さず、**すべての15-bitマスク**を試します。

```powershell
python experiments/random_mask_trial.py make --corpus build/chip-corpus-20261023 --output build/random-trial
# makeが表示した既知鍵をKEYに入れる。truth.jsonは探索へ渡さない。
python -m std_t98_cracker.enhanced_sweep build/random-trial/encrypted.burst --known-key KEY --prefilter-margin -10 --rank-bound --output build/random-trial/search
python experiments/random_mask_trial.py check --truth build/random-trial/truth.json --result build/random-trial/search/result.json --json build/random-trial/check.json
# 復元したmask/offsetを選び、通常のdescrambleで使用できる。
python -m std_t98_cracker.enhanced plugin --mask MASK --offset OFFSET -o build/random-trial/selected.py
```

マスクは15-bit部分を指定します。レポートの`polynomial`は先頭ビット付きの16進表記です。
判定条件の再測定は `python experiments/calibrate_prefilter.py --corpus build/chip-corpus-20261023 --output build/prefilter.json`。
GPU上でのPN生成を含む測定は `python experiments/benchmark_mask_sweep.py build/random-trial/encrypted.burst --output build/gpu-sweep-bench.json`。
足切りを全候補が通過する負荷も別に測定し、通常の高速化率と最悪ケースを分けて記録します。
