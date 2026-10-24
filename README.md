# std-t98-cracker

STD-T98の録音ファイルから秘話の系列・鍵を特定し、秘話を解除するCLIツールです。通常秘話とALINCOの強化秘話15系列に対応します。

**ソルバー誘導AI**が線形計算で鍵候補を絞り、hybridモデルで音声としての妥当性を判定します。系列や鍵が不明な状態から解析でき、GPU・CPUのどちらでも動作します。

512件の逐次入力テストでは誤判定0件、全件で1.8秒以内に正しい系列・鍵を特定しました（F1 = 1.000）。[評価条件と結果](experiments/BLIND_SOLVER_RESULTS.md)

## セットアップ

Python 3.11以上と64-bit MinGW（GCC）が必要です。

```powershell
.\build.ps1
python -m pip install -e '.[ai]'
```

## 使い方

### 系列と鍵を調べる

[std-t98-receiver](https://github.com/tallcat4/std-t98-receiver)で保存した、1回のPTT分の録音を指定します。

```powershell
python -m std_t98_cracker crack capture.burst --series auto --method guided-ai --llr capture.llr --json result.json
```

結果の`series`は通常秘話なら0、強化秘話なら1〜15、`key`は無線機に設定する秘話コードです。`key: 0`は秘話なし、`key: null`は判定保留を表します。系列が既知なら、`auto`をその番号に置き換えられます。

`.llr`は誤り訂正に使います。手元にない場合は`--llr capture.llr`を省略してください。

### 秘話を解除する

得られた系列と鍵を指定します。たとえば強化秘話10・コード12345なら：

```powershell
python -m std_t98_cracker descramble capture.burst --series 10 --key 12345 --llr capture.llr -o clear
```

出力の`clear.burst`を[std-t98-vocode](https://github.com/tallcat4/std-t98-vocode)に渡すと音声化できます。

### 入力確認・解析の調整

| 操作 | コマンド／オプション |
|---|---|
| ファイル形式・フレーム数の確認 | `python -m std_t98_cracker inspect capture.burst` |
| 各バーストの候補とスコアを保存 | `crack`に`--trace progress.jsonl` |
| CPUで実行 | `crack`に`--device cpu --threads 1` |
| 候補の選抜・採用基準を調整 | `crack`に`--threshold 20 --margin 6` |

バーストを逐次渡すPython APIもあります。[APIと判定基準](LINEAR_SOLVER.md)

## ALINCO強化秘話の仕組み

### 通常秘話との違い

通常秘話・強化秘話ともに、15-bit LFSR（線形帰還シフトレジスタ）で生成したビット列をAMBE音声データへXORします。強化秘話では、選んだ系列に応じて**帰還マスク**と**秘話コードに加えるオフセット**が変わります。

| 項目 | 通常秘話 | 強化秘話 |
|---|---|---|
| 系列番号 | 0 | 1〜15 |
| 秘話コード | 00001〜32767 | 00001〜32767 |
| LFSRの状態幅 | 15 bit | 15 bit |
| 帰還マスク | `0x0003` | 系列ごとに異なる |
| 初期状態 | 秘話コードそのもの | 秘話コードに系列固有のオフセットを加算 |
| 1バーストに適用する系列長 | 196 bit（49 bit × 4フレーム） | 同左 |
| 初期状態に戻すタイミング | 各バーストの先頭 | 同左 |

### マスクとオフセットの対応表

表の系列1〜15は、無線機の強化秘話設定の番号に対応します。

「多項式マスク」は最高次の`x^15`を表すビットを含む16-bit表記です。LFSRの更新には、そのビットを除いた下位15 bitの「帰還マスク」を使います。通常秘話の`0x8003`は`x^15 + x + 1`に対応します。

| 系列 | 多項式マスク | 帰還マスク | オフセット（16進） | オフセット（10進） |
|---|---|---|---|---:|
| 0（通常秘話） | `0x8003` | `0x0003` | `0x0000` | 0 |
| 1 | `0xFB8D` | `0x7B8D` | `0x1B95` | 7061 |
| 2 | `0xDC6F` | `0x5C6F` | `0x741B` | 29723 |
| 3 | `0x8999` | `0x0999` | `0x3679` | 13945 |
| 4 | `0xF499` | `0x7499` | `0x4D57` | 19799 |
| 5 | `0x9331` | `0x1331` | `0x55E8` | 21992 |
| 6 | `0xAE1D` | `0x2E1D` | `0x3444` | 13380 |
| 7 | `0xCC85` | `0x4C85` | `0x3471` | 13425 |
| 8 | `0xF4F9` | `0x74F9` | `0x68A3` | 26787 |
| 9 | `0xE7AD` | `0x67AD` | `0x7B2C` | 31532 |
| 10 | `0xD9A3` | `0x59A3` | `0x76FD` | 30461 |
| 11 | `0x944B` | `0x144B` | `0x7B59` | 31577 |
| 12 | `0xC929` | `0x4929` | `0x0C53` | 3155 |
| 13 | `0xDC71` | `0x5C71` | `0x7769` | 30569 |
| 14 | `0xB62D` | `0x362D` | `0x4597` | 17815 |
| 15 | `0x8A7B` | `0x0A7B` | `0x5748` | 22344 |

### 秘話コードから初期状態への変換

系列を`s`、秘話コードを`K`、表のオフセットを`O[s]`とすると、初期状態`S`は次の式です。

```text
S = 1 + ((K + O[s] - 1) mod 32767)       K = 1..32767
K = 1 + ((S - O[s] - 1) mod 32767)       S = 1..32767
```

加算結果が32767を超えたら32767を引き、1〜32767の範囲に折り返します。XORによる変換ではなく、整数の加算です。逆の式を使えば、解析で得たLFSRの初期状態から無線機の表示コードへ戻せます。

たとえば系列1のオフセットは7061です。

| 秘話コード `K` | 初期状態 `S` |
|---|---:|
| 00001 | 7062 |
| 25706 | 32767 |
| 25707 | 1 |
| 32767 | 7061 |

本ツールでは`key=0`を秘話なしとして扱い、初期状態と出力ビット列をすべて0にします。

### LFSRの更新と出力順序

状態のbit 0を最下位ビットとします。**更新前の最下位ビットを出力**し、帰還マスクで選んだビットのXORをbit 14へ入れて右シフトします。

```python
def pn196(key, polynomial_mask, offset):
    state = 0 if key == 0 else 1 + ((key + offset - 1) % 32767)
    taps = polynomial_mask & 0x7fff
    bits = []
    for _ in range(196):
        bits.append(state & 1)
        feedback = (state & taps).bit_count() & 1
        state = (state >> 1) | (feedback << 14)
    return bits
```

通常秘話では`taps=0x0003`なので、帰還値はbit 0とbit 1のXORです。強化秘話も同じ更新処理を使い、表からマスクとオフセットを選びます。

### 音声フレームへの適用

1バーストには4個のAMBEフレームがあり、FEC復号後の音声データは各49 bitです。上の関数で生成した196 bitを、先頭から49 bitずつ4フレームへ割り当てます。1バースト内ではLFSRを進め続け、**次のバーストの先頭で同じ初期状態へ戻します**。

生成したビット列はAMBEの自然順です。本ツールの49-bit `.burst`はThumbDV順で格納するため、各フレーム内で次の対応表を適用してからXORします。各数値は、出力位置に対応する自然順のビット番号（0始まり）です。

```python
bits = pn196(key=1, polynomial_mask=0x9331, offset=0x55E8)  # 5-00001
THUMBDV_MAP = [
    0,18,36,1,19,37,2,20,38,3,21,39,4,22,40,5,23,41,6,24,42,
    7,25,43,8,26,44,9,27,45,10,28,46,11,29,47,12,30,48,13,31,
    14,32,15,33,16,34,17,35,
]
pn_thumbdv = [
    [bits[49 * frame + i] for i in THUMBDV_MAP]
    for frame in range(4)
]
# clear[frame][i] = encrypted[frame][i] ^ pn_thumbdv[frame][i]
```

同じ系列をもう一度XORすれば元に戻るため、秘話の適用と解除は同じ処理です。72-bitの受信フレームは、先にFEC復号して49-bit音声データを取り出してから解除します。

実装は[enhanced_sequence.py](std_t98_cracker/enhanced_sequence.py)、ビット順の変換は[sequence.py](std_t98_cracker/sequence.py)にあります。

### 実機での確認

強化秘話15系列を各3コード、計45録音で照合し、上表の系列番号・マスク・オフセットで一致しました。通常秘話3録音も同じ処理で確認しています。系列10はCH1に加え、CH2・CH30でも同じマスクとコード変換で復元できました。[実機検証の詳細](experiments/ENHANCED_RESULTS.md)

## 関連資料

- [ソルバー誘導AIのAPI・判定基準](LINEAR_SOLVER.md)
- [性能評価とヒストグラム](experiments/BLIND_SOLVER_RESULTS.md)
- [評価データの作成と再現手順](experiments/README.md)
- [ファイル形式](FORMAT.md)
- [ソース・モデルとライセンス](PROVENANCE.md)
