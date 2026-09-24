# Go2 SNN APEX: モーターが動かない原因とv5の検証

## 今回のログで確定したこと

- Priorだけを動かす事前診断では4指令とも500/500 step。平均指令軸速度は前進+0.31、後退−0.46、左移動+0.40 m/s、左旋回+0.54 rad/s。PDとMuJoCo Warpの環境側には、少なくとも歩容を動かす能力があります。
- 同じ実行のPrior OFF評価では開始時の実速度がほぼ0で、500〜700 iteration後も4指令とも追従スコア0、実速度は概ね0.003 m/sまたはrad/s以下。動画の静止と一致します。
- 開始時のSNN第3層スパイク率は0.000、Actor平均行動 `|mu|=0.000`。旧デコーダは最終スパイク率の正負差だけを使い、bias初期値0なので、この場合の決定論的行動は厳密に0です。
- 500 iteration時点でも `|mu|≈0.059`。v4の行動スケールを掛けると関節目標偏差の平均はおよそ0.03〜0.05 rad、PDの対応トルクはおよそ1.8〜2.9 Nmです。足を振り出すための関節命令には小さい値です。
- PPO学習率は100 iterationまでに下限`2e-5`へ達し、KL早期停止も継続しています。これはActorの改善をさらに遅らせます。
- CPU MuJoCoでActor行動を全関節に+0.5だけ強制注入すると、平均トルク14.8 Nm、1制御stepの平均関節変位0.0605 radでした。少なくとも同じMJCFのモーター自体は命令に応答します（`verify_apex_action_path_cpu.py`）。

## ユーザー提供のMuJoCo Warp移植版との対応

提供された `APEX_MuJoCo_Warp.ipynb`、`env.py`、`train.py`、`go2_mjcf.py`、READMEを読み、制御から学習器まで比較しました。

| 構成 | 提供された移植版 | 従来のSNN Notebook |
|---|---|---|
| Actor | 元APEXの連続値MLP | SNN最終層の二値スパイク率だけ |
| 初期探索標準偏差 | 1.0 | v4では0.40 |
| 模倣軌道 | 公開Go2の実モーションCSV | 独自の手続き生成IK歩容 |
| Actor観測 | 45D | 45D、位相なし |
| PPO | 公開Multi-Critic PPO | 独自の時系列SNN PPO |

元APEXのMLP Actorは `nn.Linear(..., num_actions)` から連続行動を直接出すため、最終層の無発火で行動が必ず0になる構造ではありません。提供されたWarp移植版README自体は、GPUでの学習成功が未検証と明記しています。したがって、その移植版を「学習済みの正解」とは扱わず、Actor・モーション・PPOの構成差を根拠にしています。元APEXの既定Go2訓練は前進中心であり、このNotebookの前後左右・旋回4指令とは課題も異なります。

## v5の変更

1. **SNN膜電位を使う連続モーター読み出し**：第1/第2スパイク層の膜電位をLayerNormして線形ヘッドへ通し、最終スパイク率出力は小さい残差として残します。最終スパイクが全0でも、Actorが観測に応じて連続行動を出し、PPO勾配が第1層へ戻ります。
2. **歩容位相を入力**：独自の周期IK軌道を模倣するため、位相sin/cosの2DをActorへ加えます。入力は47D、Criticは79Dです。これは元APEXの45D完全再現ではありません。
3. **ActorとCriticの勾配を別々にクリップ**：Critic勾配が大きいとき、SNN Actorの勾配まで一括で縮小する事態を避けます。Actor・Critic・行動標準偏差の勾配ノルムを別途記録します。
4. **PPO前の行動経路検査**：決定論的Actor出力、第1層までの勾配、強制行動による実トルクと関節変位を検査します。いずれかがほぼ0なら長時間のPPOを始めません。
5. **評価のモーター計測**：Prior OFFで指令別のActor行動、トルク、関節速度、実移動速度を表示し、TensorBoardとグラフへ保存します。

v5は蒸留を使わずPPOで直接Actorを学習します。v4のyaw転倒・姿勢報酬・PD制御修正は維持します。
新しいActor構造と観測次元のため、以前のチェックポイントは読み込めません。

## 実行と判定

`09-24_go2_snn_apex_motor_fix_v5.ipynb` を新規セッションで先頭から実行します。
最初に `Actor motor preflight`、`Motor injection`、`Initial Actor action` の値が表示されます。強制行動とActor自身の行動の両方についてトルクを検査し、ほぼ無トルクなら学習前に停止します。
実行後、`test/*_action_abs`、`test/*_torque_abs`、`test/*_joint_speed_abs` と `test/*_axis_speed` を同時に見ます。
行動・トルク・関節速度が出ても移動しなければ、報酬・歩容・接地の問題です。
行動からトルク/変位への事前検査が失敗すれば、PD/Warpの配線問題です。

このPC/WSLにはCUDA対応のPyTorchとMuJoCo Warpがないため、v5のGPU学習完走は未検証です。

参照: https://github.com/marmotlab/APEX/blob/main/rsl_rl/rsl_rl/modules/actor_multi_critic.py
