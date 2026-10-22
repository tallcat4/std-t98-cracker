# STD-T98 debug tools: file contract v1

receiver / cracker / vocodeで同じ内容を保持します。旧std-t98-tools / std-t98-vocodeと互換です。

## .burst

ファイルヘッダーはありません。同じ形式のレコードを連結します。
1バースト = 4 AMBEフレーム = 80 msのtraffic payload。マーカーは16進表記です。

| 形式 | 1レコード | 長さ |
|---|---|---|
| 72-bit、3600 bps、FEC付き | `FF (48 + 9-byte payload) × 4` | 41 byte |
| 49-bit、2450 bps、FEC除去後 | `FF (31 + 7-byte payload) × 4` | 33 byte |

`48`と`31`はASCII文字列ではなく、ビット数を示す1 byteのマーカーです。
49-bit payloadはThumbDV / AMBE-3000R rate-34順、MSB first。最後のbyteの下位7 bitは0。
72-bit payloadはAMBE-3000R rate-33の送信bit順。秘話はFEC除去後の49 bitに作用します。

`FF`はpayloadにも現れます。破損時にFFを検索して再同期してはいけません。
長さ、境界、フレームマーカー、paddingが一致しないファイルはエラーです。
空ファイルは受信0件として生成可能ですが、解析/音声化では「trafficなし」のエラーです。

時刻、チャネル、呼ID、鍵ID、品質、欠落区間は保持しません。
保存された隣接レコードがRF上で連続している保証はありません。
解析windowはファイル上のレコード範囲であり、実時間の連続性を示しません。

## .ambe（互換用）

`.burst`から各レコードの先頭FFだけを取り除いたものです。
1フレームが`48 + 9 byte`または`31 + 7 byte`。
receiverは`--ambe`を指定したときだけ72-bit版を保存します。
バースト位相を復元できないため、crackerの入力には`.burst`を使います。

## .llr（オプション）

receiverの`--llr`出力。ヘッダーなし、1バースト288 byte = 4×72個のsigned int8。
`.burst`と同じバースト/フレーム/送信bit順。**正=1、負=0、絶対値=信頼度**。
同時に生成された72-bit `.burst`と組み合わせます。byte数はバースト数×288と厳密に一致させます。
サイズが同じ別のcaptureとの取り違えは形式だけでは検出できません。

## 標準秘話と系列拡張

鍵は0..32767。0は平文。標準PNは15-bit LFSR、出力bit=state LSB、feedback=bit0 XOR bit1。
stateを右shiftしfeedbackをbit14へ入れ、各バースト先頭で鍵からリセットして196 bit生成します。
4×49のAMBE自然順をThumbDV順へ並べ替えてXORします。

crackerの`--sequence`は`generate(key) -> 196 binary values`を提供するローカルPythonファイル。
受信ファイルの形式やvocodeを変更せず、系列だけを差し替えられます。

## 処理の境界

receiverは72-bit `.burst`を作るだけ。crackerの鍵探索はJSONを返すだけ。
crackerの指定鍵変換は49-bit `.burst`を作るだけ。vocodeのdecodeはWAVを作るだけ。
各処理は利用者が個別に実行します。
