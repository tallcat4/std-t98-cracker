# ソース・モデルの由来

- 線形ソルバー: SPEAR `d1611fd408939ad238372d58828639e8fde28400` の`apps/std_t98/secret/silence.cpp`をPythonへ移植。
  22個の制約と全9フィールドの確認を保持。採用判断は下記の逐次スコア方式へ更新。
- 2026-10-23の実機録音とThumbDVによるJSUT/LibriSpeech符号化実験を基に、
  無声区間の証拠集約、有声区間の制約、閾値を設定できる逐次スコアを同じ線形ソルバーへ追加。
  適用範囲と検証条件は[LINEAR_SOLVER.md](LINEAR_SOLVER.md)。実録音・コーパスは含めません。
- FEC: 同SPEARの`apps/std_t98/blip25c/src/blip25.c`からhard/Chase-II soft FECだけを抽出。
  `b25_golay.c`は同梱テーブル。音声合成とmbelib互換層は含めません。
  原著[blip25-vocoder](https://github.com/OpenBLIP25/blip25-vocoder)のMITライセンスを`vendor/blip25c/LICENSE`に保持。
- AI: 旧[std-t98-tools](https://github.com/tallcat4/std-t98-tools) `09f7e59` の`core/secret/cracker.py`と
  `models/secret_voice`のFFNN/hybrid safetensorsとJSON。重みは変更していません。
  importをローカル化し、系列生成関数を注入可能にしました。CLIではwindowをまたぐ鍵キャッシュを使いません。
- バースト形式・PN・ThumbDVのマッピングは旧std-t98-toolsおよびstd-t98-vocodeと共通。

`std_t98_cracker/models/`のJSONは元の学習メタデータです。validation accuracyは元の学習時の値で、
今回の実機データでの鍵正答率ではありません。AI結果は候補として報告します。

本体はGPL-3.0。実録音は含めていません。
