# Go2 APEX + MuJoCo Warp + SNN Actor

この実装は、既存の `apex_mjwarp/APEX` にあるモーションCSV、Multi-Critic PPO、runner、storage、Criticを使います。物理はMuJoCo Warp、ActorだけSNNです。APEXの新規cloneやチェックアウト変更は行いません。

作業場所: `/home/hoge/university/laboratory/go2_2/apex_mjwarp/`。既存の `env.py`、`train.py`、`go2_mjcf.py` はそのまま残します。SNN用環境は `env_snn.py` です。

## 設定

| 項目 | 値 |
|---|---|
| SNN Actor | encoder 64、hidden 256/256、decoder 256、1行動につき4 tick |
| PPO / Critic | 既存APEXの2-Critic PPO / 45D Actor観測・77D Critic観測 |
| モーション | `go2_retarget_canter_2ms.csv` |
| 制御 | Kp 20、Kd 0.5、action scale 0.25、DecAPを最終トルクに加算 |
| 既定の学習 | 1024環境、24 steps、5 PPO epochs、4 minibatches、1200 iterations |
| 評価 | Action Prior OFF、ノイズ・外乱OFF、速度・行動・トルクをCSVと動画に出力 |

公式PPOのstorageは時系列の膜電位を保存しないため、SNN膜電位を各行動で初期化します。これはスパイクを持つfeed-forward方策であり、制御ステップ間に膜電位を保持するSNNとは異なります。

## 実行

Linux/WSLのCUDA対応PyTorchと、`requirements_snn.txt` のMuJoCo Warp等が必要です。既存の `APEX/rsl_rl` はスクリプトから読み込むため、別のAPEXをダウンロードする必要はありません。

```bash
cd /home/hoge/university/laboratory/go2_2/apex_mjwarp
python3 probe_csv_prior_cpu.py
python3 train_snn.py --smoke --log-dir runs/apex_snn_smoke
python3 train_snn.py --num-envs 1024 --iterations 1200 --log-dir runs/apex_snn_full
python3 evaluate_snn.py --checkpoint runs/apex_snn_full/model_1200.pt --vx 1.0 --output runs/forward.csv --video runs/forward.mp4
```

Notebookは `APEX_SNN_MuJoCo_Warp.ipynb` です。最初のセルで同じフォルダの既存 `APEX` を確認します。環境数を公式設定の4096に合わせるには `--num-envs 4096` を指定します。

学習前に独立した1環境で200-stepのPrior単体挙動とActor出力・勾配を検査します。CPU MuJoCo簡易プローブでは、この変換MJCFと公式canter CSVの組み合わせで、Prior ONかつゼロActor行動では10ステップで転倒しました。これはActorを含むPPOの失敗を確定する結果ではありませんが、Warp移植の接触・制御の確認点です。

SNN用環境では、行動と報酬にreference `k` を使用した後、次の観測を作る前に`k+1`へ進めます。リセットされた環境では位相と前回行動・速度を初期化します。

このWSLにはCUDA版PyTorchとMuJoCo Warpが見つからないため、GPU上でのPPO更新や学習完走は未検証です。`--smoke`で1回のPPO更新を確認し、学習後はPrior OFFの速度と動画で歩行を判定してください。
