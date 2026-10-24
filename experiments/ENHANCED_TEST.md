# 強化秘話の実機試験

1回のPTT・1設定につき、`.burst`と`.llr`を1組保存します。ファイル名を`ch1-5-00001`のようにすると、チャネル・強化秘話系列・秘話コードを対応付けられます。系列0は通常秘話です。

## 解析と音声確認

次の例はCH1・強化秘話5・コード00001の録音です。解析には系列・鍵を渡さず、ソルバー誘導AIの回答が設定値に一致するか確認します。

```powershell
python -m std_t98_cracker crack ch1-5-00001.burst --series auto --method guided-ai --llr ch1-5-00001.llr --trace build/ch1-5-00001.jsonl --json build/ch1-5-00001.json
```

結果の系列・鍵を使って解除し、`std-t98-vocode`でWAVへ変換します。

```powershell
python -m std_t98_cracker descramble ch1-5-00001.burst --series 5 --key 1 --llr ch1-5-00001.llr -o build/clear
python -m std_t98_vocode decode build/clear.burst -o build/ch1-5-00001.wav
```

送信時に話した内容と復元音声を照合します。系列・コードの組ごとに別の発話を録音し、同じ手順を繰り返します。

## 確認する設定

各系列で複数のコードを試します。`00001`・`12345`・`32767`に加え、初期状態が32767から1へ折り返すコードの前後を選ぶと、コード変換も確認できます。

通常秘話と秘話OFFを対照にし、同じ設定を別チャネルでも録音します。CH15は呼出チャネルで実機が秘話を無効にするため、秘話OFFの確認に使います。

録音をまとめて採点する場合は、[評価スクリプト](README.md#ソルバー誘導aiを評価する)に設定一覧を渡します。

[マスク・オフセットと変換式](../README.md#alinco強化秘話の仕組み) / [実機検証の結果](ENHANCED_RESULTS.md)
