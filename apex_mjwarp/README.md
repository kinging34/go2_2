# APEX Go2 → MuJoCo Warp 移植プロトタイプ

このフォルダは [marmotlab/APEX](https://github.com/marmotlab/APEX) の Go2・平坦地形の既定設定を対象にした JupyterLab 用の移植です。元リポジトリのコミット `f35c54a4fec00c03751e0d28187278313cde51d9` を参照します。`rsl_rl` の Multi-Critic Actor-Critic、Multi-Critic PPO、rollout storage は元コードをそのまま読み込みます。物理環境と状態テンソルの取得部分を `env.py` に置き換えています。

## 実行環境

- Linux、NVIDIA CUDA GPU、Python 3.10 以降
- CUDA 対応 PyTorch、`mujoco-warp==3.14.0`、`pandas`、`pyyaml`、`wandb`、`tensorboard`、JupyterLab
- このフォルダで `APEX_MuJoCo_Warp.ipynb` を開き、上から順に実行

ノートブックは元 APEX を取得して指定コミットに切り替えます。`train.py` を直接使う場合は、先に APEX を取得して `python train.py --apex-root /path/to/APEX --num-envs 64 --iterations 1` を実行します。W&B は既定でオフラインです。

## 対応表

| 項目 | 元 APEX | この移植 |
|---|---|---|
| 物理 | Isaac Gym PhysX | MuJoCo Warp `put_model` / `make_data` / `step` |
| Go2 形状 | URDF、円柱は capsule 化 | 同じ URDF から collision-only MJCF を生成、円柱は capsule 化 |
| 制御 | 5 ms 物理刻み、4 回 decimation、APEX position prior | 同じ刻み・制御式・減衰係数 |
| 模倣 | Go2 CSV、列 6:18 関節、18:21 コマンド、21 高さ、22:34 足先、36:40 quat | 同じ CSV と列範囲 |
| 観測 | actor 45、critic 77 | 同じ次元・並び・スケール |
| 報酬 | group1 模倣、group2 追跡と正則化 | 既定の非ゼロ項を同じ係数で計算し `dt` を乗算 |
| 学習 | APEX 内 `rsl_rl` の 2 critic PPO | 同じコードを変更せず使用 |

## ファイル

- `go2_mjcf.py`: URDF の質量、慣性、関節、primitive collision を MuJoCo MJCF へ変換
- `env.py`: MuJoCo Warp 上で並列 Go2 を動かし、APEX の観測・既定報酬を返す環境
- `train.py`: 元 APEX の Multi-Critic PPO runner への接続
- `APEX_MuJoCo_Warp.ipynb`: セットアップ、モデル検証、環境スモークテスト、1 更新、本学習用セル

## 再現性の限界と残作業

これは**学習結果の再現を実証した完成版ではありません**。この作業環境では CUDA 実行環境を確認できず、MuJoCo Warp のステップと PPO 更新を実機実行できていません。MuJoCo CPU 上では生成 MJCF がコンパイルでき、`nq=19, nv=18, nu=12` を確認しました。

1. **接触力**: Isaac Gym の net contact force に対し、MuJoCo の `cfrc_ext` の並進成分を使用します。足接地判定、滑り、衝突終了の値と閾値を実機上で照合する必要があります。
2. **物理エンジン差**: 同じ URDF と時刻幅でも PhysX と MuJoCo の接触、摩擦、積分結果は一致しません。報酬曲線や学習済み重みの一致は期待できません。
3. **ドメインランダム化**: motor offset、Kp/Kd 係数、周期的 push は実装済みです。元既定設定の環境別 friction、base mass、link mass のランダム化はまだ未実装です。耐外乱性能を評価する前に追加してください。
4. **地形と拡張**: 平坦地形の Go2 既定設定のみです。rough terrain、height scan、他ロボット、他の観測モード、single-critic ablation、reference-state init、複数モーション切替には未対応です。
5. **表示**: MJCF は衝突形状のみで visual DAE mesh を省いています。見た目の確認には別途 mesh の変換と renderer の設定が必要です。
6. **定量検証**: 同一初期状態とアクション列で元 Isaac Gym 環境と、関節順序、姿勢座標、足先位置、接触力、報酬各項をログ比較してください。少数環境のスモークテストが通ってから 4096 環境へ増やします。

MuJoCo Warp の公開 API: https://mujoco.readthedocs.io/en/latest/mjwarp/ と https://mujoco.readthedocs.io/en/latest/mjwarp/api.html
