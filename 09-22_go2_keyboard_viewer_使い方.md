# Go2キーボードビューアのWSL実行手順

対象スクリプト：`09-22_go2_keyboard_viewer.py`

## 1. 必要なファイル

スクリプトと同じフォルダに次を置きます。`WORK_DIR` はスクリプト自身の場所へ自動設定されます。

```text
09-22_go2_keyboard_viewer.py
scene_flat.xml
go2.xml
assets/
params_MM-DD/
  go2_imitation_10Kit.pt
  go2_imitation_norm_10Kit.npz
  go2_residual_stateful_10000it.pt   # 残差制御を使う場合
```

現在のWindows側ワークスペースには `go2.xml`、`assets/`、`params_MM-DD/` がないため、学習に使用した環境からコピーする必要があります。

## 2. WSL環境の準備

UbuntuのWSLターミナルで実行します。

```bash
cd '/mnt/c/大学/4年/コード'

sudo apt update
sudo apt install -y python3-venv libglfw3 libgl1-mesa-dri libglx-mesa0

python3 -m venv .venv-viewer
source .venv-viewer/bin/activate

python -m pip install -U pip
python -m pip install numpy pillow mujoco
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

このビューアではMuJoCo物理計算をCPUで行います。SNN推論もCPUで動作できます。

## 3. WSLgの確認

```bash
echo "$DISPLAY"
echo "$WAYLAND_DISPLAY"
```

通常のWSLg環境では、たとえば `DISPLAY=:0`、`WAYLAND_DISPLAY=wayland-0` のように表示されます。

## 4. スクリプト設定

`09-22_go2_keyboard_viewer.py` の先頭付近にある設定を確認します。

```python
WORK_DIR = Path(__file__).resolve().parent
PARAMS_DIR = None
BASE_ACTOR_CHECKPOINT = None
BASE_NORM_CHECKPOINT = None
RESIDUAL_CHECKPOINT = None
START_TERRAIN = "easy"
USE_RESIDUAL_AT_START = True

# 軽量表示設定
WINDOW_WIDTH, WINDOW_HEIGHT = 320, 180
RENDER_HZ = 15.0
VSYNC = False
TORCH_THREADS = 1
```

`None` のチェックポイントは、更新日時が最も新しい `params_*` フォルダから自動検索されます。別のファイルを使う場合は明示的にパスを指定します。

```python
PARAMS_DIR = WORK_DIR / "params_09-22"
BASE_ACTOR_CHECKPOINT = PARAMS_DIR / "go2_imitation_10Kit.pt"
BASE_NORM_CHECKPOINT = PARAMS_DIR / "go2_imitation_norm_10Kit.npz"
RESIDUAL_CHECKPOINT = PARAMS_DIR / "go2_residual_stateful_10000it.pt"
```

学習ノートブックは `MUJOCO_GL=egl` を使いますが、このスクリプトは起動時に `MUJOCO_GL=glfw` を設定してWSLgウィンドウを開きます。

表示は320×180、15 fpsです。歩行制御は50 Hz、内部のPD制御と物理計算は500 Hzのままなので、描画を軽くしても方策へ与える制御周期は変わりません。影、反射、霧、空、アンチエイリアス、VSyncも無効化しています。

## 5. 実行

WSLターミナルから次を実行します。

```bash
cd '/mnt/c/大学/4年/コード'
source .venv-viewer/bin/activate
python 09-22_go2_keyboard_viewer.py
```

WSLgのMuJoCoウィンドウが開きます。

キー入力は、ブラウザではなくMuJoCoウィンドウをクリックしてから行います。

| キー | 操作 |
|---|---|
| `W` / `S` | 前進 / 後退速度を段階的に変更 |
| `A` / `D` | 左 / 右移動速度を段階的に変更 |
| `Q` / `E` | 左 / 右旋回速度を段階的に変更 |
| `X` | 速度指令をゼロにする |
| `R` | ロボットを開始位置へ戻す |
| `B` | 平地priorのみ / prior＋残差を切り替える |
| `P` | 一時停止 / 再開 |
| `1` | 平地へ切り替える |
| `2` | easy、最大2 cmへ切り替える |
| `3` | medium、最大5 cmへ切り替える |
| `4` | hard、最大10 cmへ切り替える |
| `H` | 操作一覧を再表示する |
| `Esc` | ビューアを終了する |

速度指令の範囲は、学習時と同じ次の範囲に制限されます。

```text
前後速度 vx : -0.60 ～ +0.60 m/s
左右速度 vy : -0.30 ～ +0.30 m/s
旋回速度 wz : -1.00 ～ +1.00 rad/s
```

## 6. 地形切替の動作

数字キーを押すと、現在のMuJoCo viewerを閉じ、選択したXMLを読み込んで新しいviewerを自動的に開きます。地形変更時は速度指令をゼロへ戻します。

不整地XMLとPNGはノートブック実行時に毎回生成されます。このため、Windowsで生成したXMLにWindows形式の絶対パスが入っていても、WSL上のパスへ自動的に置き換わります。

## 7. 確認方法

最初は次の順序で確認します。

1. `1` の平地で `W` を数回押し、前進できるか確認
2. `2` のeasyへ切り替え、同じ前進指令を与える
3. `B` でpriorのみと残差ありを交互に比較
4. `3`、`4` の順に段差を高くする
5. 転倒した場合は `R` でリセットする

`B` を押したときに「残差checkpointが読み込まれていない」と表示された場合は、`RESIDUAL_CHECKPOINT` のパスを確認します。

実行中は5秒ごとに次のような実測速度を表示します。

```text
実測: control=50.0 Hz  render=15.0 fps  遅延=0
```

`control` が50 Hz付近ならリアルタイム制御できています。まだ重い場合は、スクリプト先頭の `RENDER_HZ` を `10.0`、画面を `256, 144` へ下げられます。また、`B` で残差を無効にするとSNN推論が1本だけになるため軽くなりますが、残差学習の効果は使われなくなります。

## 8. 主なエラー

### `scene_flat.xml がありません`

`scene_flat.xml` がスクリプトと同じフォルダにあるか確認します。別の場所を使う場合は `WORK_DIR` を変更します。

### `Error opening file 'go2.xml'` またはasset関連エラー

`go2.xml` と `assets/` を `scene_flat.xml` から参照できる位置へ置きます。

### `平地actorが見つかりません`

`params_MM-DD/` を配置するか、`BASE_ACTOR_CHECKPOINT` を明示します。

### `GLFW initialization failed` またはviewerが開かない

WSLターミナルからスクリプトを起動しているか、`DISPLAY` と `WAYLAND_DISPLAY` が設定されているかを確認します。SSH経由やGUIソケットを渡していないDocker内では、そのままではWSLgウィンドウを開けません。

