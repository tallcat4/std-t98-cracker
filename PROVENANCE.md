# ソース・モデルとライセンス

本体はGPL-3.0です。

- `linear.py`：SPEAR `d1611fd408939ad238372d58828639e8fde28400`の`apps/std_t98/secret/silence.cpp`を基に移植・拡張。
- FEC：同SPEARの`apps/std_t98/blip25c/src/blip25.c`からhard／Chase-II soft FECを抽出。`b25_golay.c`は同梱テーブル。原著[blip25-vocoder](https://github.com/OpenBLIP25/blip25-vocoder)のMITライセンスを[vendor/blip25c/LICENSE](vendor/blip25c/LICENSE)に保持。
- AI：旧[std-t98-tools](https://github.com/tallcat4/std-t98-tools) `09f7e59`の`core/secret/cracker.py`と`models/secret_voice`を使用。モデルの重みは元のsafetensorsを使用。
- バースト形式・通常秘話のPN・ThumbDVマッピングはstd-t98-toolsおよびstd-t98-vocodeと共通。

`std_t98_cracker/models/`のJSONは元の学習メタデータです。本ツールでの系列・鍵の正答率は[性能評価](experiments/BLIND_SOLVER_RESULTS.md)を参照してください。
