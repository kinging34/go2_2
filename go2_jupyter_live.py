"""Browser-based live controller for go2_imitation_warp_release(1).ipynb.

Run this *after training*, in a notebook cell, with:

    %run -i go2_jupyter_live.py

The ``-i`` is required because this file reuses test_env, ac, replay_buffer,
get_action, and new_mem from the notebook kernel.
"""

import asyncio
import io
import threading
import time

import ipywidgets as widgets
import mujoco
import numpy as np
import torch
from IPython.display import display
from PIL import Image as PILImage

try:
    from ipyevents import Event
except ImportError:
    Event = None


_required = ("test_env", "replay_buffer", "get_action", "new_mem", "CMD_SCALE")
_missing = [name for name in _required if name not in globals()]
if _missing:
    raise RuntimeError(
        "Run the training notebook first. Missing notebook variables: "
        + ", ".join(_missing)
    )


# Performance profile.  Physics remains at 50 Hz; only browser video is reduced.
# Increase LIVE_RENDER_HZ to 15 or LIVE_WIDTH/HEIGHT to 480/360 if the connection
# and browser can keep up.
LIVE_WIDTH = int(globals().get("LIVE_WIDTH", 320))
LIVE_HEIGHT = int(globals().get("LIVE_HEIGHT", 240))
LIVE_RENDER_HZ = int(globals().get("LIVE_RENDER_HZ", 25))
LIVE_JPEG_QUALITY = int(globals().get("LIVE_JPEG_QUALITY", 80))


# Stop an older controller if this file is run again.
if "_go2_stop_event" in globals():
    _go2_stop_event.set()
if "_go2_live_state" in globals():
    _go2_live_state["running"] = False
if "_go2_live_task" in globals() and not _go2_live_task.done():
    _go2_live_task.cancel()
for _old_thread_name in ("_go2_control_thread", "_go2_render_thread"):
    _old_thread = globals().get(_old_thread_name)
    if _old_thread is not None and _old_thread.is_alive():
        _old_thread.join(timeout=2.0)


_go2_state_lock = threading.Lock()
_go2_stop_event = threading.Event()
_go2_live_state = {
    "running": True,
    "paused": False,
    "reset": True,
    "steps": 0,
    "command": np.zeros(3, dtype=np.float32),
    "control_hz": 0.0,
    "render_hz": 0.0,
    "last_fallen": False,
    "sim_state": None,
    "sim_seq": 0,
    "jpeg": None,
    "jpeg_seq": 0,
    "error": None,
}


def _slider(description, limit, step):
    return widgets.FloatSlider(
        description=description,
        min=-float(limit),
        max=float(limit),
        step=step,
        value=0.0,
        readout_format="+.2f",
        continuous_update=False,
        layout=widgets.Layout(width="620px"),
        style={"description_width": "90px"},
    )


_go2_vx = _slider("vx [m/s]", CMD_SCALE[0], 0.05)
_go2_vy = _slider("vy [m/s]", CMD_SCALE[1], 0.05)
_go2_wz = _slider("wz [rad/s]", CMD_SCALE[2], 0.10)

_go2_image = widgets.Image(
    format="jpeg",
    # Display the 320x240 source enlarged; this avoids transmitting 4x as many pixels.
    layout=widgets.Layout(width="640px", border="2px solid #555"),
)
_go2_status = widgets.HTML()
_go2_hint = widgets.HTML()
if Event is None:
    _go2_hint.value = (
        "<b>ボタンまたはスライダーで操作できます。</b> "
        "キー操作も使う場合は、先に <code>%pip install -q ipyevents</code> を実行してください。"
    )
else:
    _go2_hint.value = (
        "<b>映像の上にマウスを置いて操作:</b> "
        "W/S=前後, A/D=左右, Q/E=旋回, X=停止, R=リセット, Space=一時停止"
    )


def _button(label, width="92px"):
    return widgets.Button(description=label, layout=widgets.Layout(width=width))


_go2_forward = _button("前進")
_go2_backward = _button("後退")
_go2_left = _button("左移動")
_go2_right = _button("右移動")
_go2_turn_left = _button("左旋回")
_go2_turn_right = _button("右旋回")
_go2_stop = _button("停止")
_go2_reset = _button("姿勢リセット", "112px")
_go2_pause = _button("一時停止", "100px")
_go2_finish = _button("終了", "80px")


def _command():
    return np.asarray(
        [_go2_vx.value, _go2_vy.value, _go2_wz.value], dtype=np.float32
    )


def _publish_command(_change=None):
    with _go2_state_lock:
        _go2_live_state["command"] = _command()


def _set_command(vx, vy, wz):
    _go2_vx.value = float(np.clip(vx, -CMD_SCALE[0], CMD_SCALE[0]))
    _go2_vy.value = float(np.clip(vy, -CMD_SCALE[1], CMD_SCALE[1]))
    _go2_wz.value = float(np.clip(wz, -CMD_SCALE[2], CMD_SCALE[2]))


for _go2_slider in (_go2_vx, _go2_vy, _go2_wz):
    _go2_slider.observe(_publish_command, names="value")
_publish_command()


_go2_forward.on_click(lambda _b: _set_command(CMD_SCALE[0], 0, 0))
_go2_backward.on_click(lambda _b: _set_command(-CMD_SCALE[0], 0, 0))
_go2_left.on_click(lambda _b: _set_command(0, CMD_SCALE[1], 0))
_go2_right.on_click(lambda _b: _set_command(0, -CMD_SCALE[1], 0))
_go2_turn_left.on_click(lambda _b: _set_command(0, 0, CMD_SCALE[2]))
_go2_turn_right.on_click(lambda _b: _set_command(0, 0, -CMD_SCALE[2]))
_go2_stop.on_click(lambda _b: _set_command(0, 0, 0))


def _request_reset(_button=None):
    with _go2_state_lock:
        _go2_live_state["reset"] = True


def _toggle_pause(_button=None):
    with _go2_state_lock:
        _go2_live_state["paused"] = not _go2_live_state["paused"]
        paused = _go2_live_state["paused"]
    _go2_pause.description = "再開" if paused else "一時停止"


def _finish(_button=None):
    with _go2_state_lock:
        _go2_live_state["running"] = False
    _go2_stop_event.set()


_go2_reset.on_click(_request_reset)
_go2_pause.on_click(_toggle_pause)
_go2_finish.on_click(_finish)


def _increment(slider, amount):
    slider.value = float(np.clip(slider.value + amount, slider.min, slider.max))


def _handle_key(event):
    key = event.get("key", "").lower()
    if key == "w":
        _increment(_go2_vx, 0.05)
    elif key == "s":
        _increment(_go2_vx, -0.05)
    elif key == "a":
        _increment(_go2_vy, 0.05)
    elif key == "d":
        _increment(_go2_vy, -0.05)
    elif key == "q":
        _increment(_go2_wz, 0.10)
    elif key == "e":
        _increment(_go2_wz, -0.10)
    elif key == "x":
        _set_command(0, 0, 0)
    elif key == "r":
        _request_reset()
    elif key == " ":
        _toggle_pause()


# ipyevents captures keys while the pointer is over the image.  The controls
# remain fully usable through buttons/sliders when ipyevents is not installed.
if Event is not None:
    _go2_live_keyboard = Event(
        source=_go2_image,
        watched_events=["keydown"],
        prevent_default_action=True,
    )
    _go2_live_keyboard.on_dom_event(_handle_key)
else:
    _go2_live_keyboard = None


_go2_controls = widgets.VBox(
    [
        _go2_hint,
        _go2_image,
        _go2_status,
        _go2_vx,
        _go2_vy,
        _go2_wz,
        widgets.HBox(
            [_go2_forward, _go2_backward, _go2_left, _go2_right]
        ),
        widgets.HBox(
            [_go2_turn_left, _go2_turn_right, _go2_stop, _go2_reset]
        ),
        widgets.HBox([_go2_pause, _go2_finish]),
    ]
)
display(_go2_controls)


def _jpeg(frame):
    buffer = io.BytesIO()
    PILImage.fromarray(frame).save(
        buffer,
        format="JPEG",
        quality=LIVE_JPEG_QUALITY,
        optimize=False,
    )
    return buffer.getvalue()


def _worker_error(where, exc):
    message = f"{where}: {type(exc).__name__}: {exc}"
    with _go2_state_lock:
        _go2_live_state["error"] = message
        _go2_live_state["running"] = False
    _go2_stop_event.set()
    print(f"Go2 live controller stopped: {message}")


def _control_worker():
    """Own all Torch/Warp work and keep it independent from browser rendering."""
    env = test_env
    try:
        torch.cuda.set_device(device)
        with _go2_state_lock:
            command = _go2_live_state["command"].copy()
        observation = env.reset(command=command)
        membrane = new_mem(env.N)
        previous_command = command.copy()

        step_number = 0
        snapshot_every = max(1, round((1.0 / env.dt) / LIVE_RENDER_HZ))
        next_tick = time.perf_counter()
        meter_started = next_tick
        meter_steps = 0

        while not _go2_stop_event.is_set():
            with _go2_state_lock:
                running = _go2_live_state["running"]
                paused = _go2_live_state["paused"]
                reset = _go2_live_state["reset"]
                command = _go2_live_state["command"].copy()
                _go2_live_state["reset"] = False
            if not running:
                break

            if reset:
                observation = env.reset(command=command)
                membrane = new_mem(env.N)
                previous_command = command.copy()
            elif not np.allclose(command, previous_command):
                env._set_cmd(env.all_idx, command)
                observation = env._get_obs()
                previous_command = command.copy()

            if paused:
                next_tick = time.perf_counter()
                _go2_stop_event.wait(0.01)
                continue

            # no_grad (not inference_mode) keeps persistent env fields mutable.
            with torch.no_grad():
                action, next_membrane = get_action(
                    replay_buffer.normalize_obs(observation), membrane, 0
                )
                observation2, _reward, done, info = env.step(action)

            last_fallen = bool(info["fallen"][0].item())
            if bool(done.any().item()):
                observation = env.autoreset(done, command=command)
                membrane = torch.where(
                    done.unsqueeze(1), new_mem(env.N), next_membrane
                )
            else:
                observation = observation2
                membrane = next_membrane

            # Publish a tiny 37-value snapshot. Rendering consumes only the
            # latest snapshot and never blocks this control thread.
            if step_number % snapshot_every == 0:
                sim_state = (
                    torch.cat((env.qpos[0], env.qvel[0]))
                    .detach()
                    .cpu()
                    .numpy()
                    .copy()
                )
                with _go2_state_lock:
                    _go2_live_state["sim_state"] = sim_state
                    _go2_live_state["sim_seq"] += 1

            step_number += 1
            meter_steps += 1
            now = time.perf_counter()
            if now - meter_started >= 1.0:
                elapsed = now - meter_started
                with _go2_state_lock:
                    _go2_live_state["control_hz"] = meter_steps / elapsed
                    _go2_live_state["steps"] += meter_steps
                    _go2_live_state["last_fallen"] = last_fallen
                meter_started, meter_steps = now, 0

            next_tick += env.dt
            delay = next_tick - time.perf_counter()
            if delay < -0.25:
                next_tick = time.perf_counter()
                delay = 0.0
            if delay > 0:
                _go2_stop_event.wait(delay)
    except Exception as exc:
        _worker_error("control", exc)
    finally:
        with _go2_state_lock:
            _go2_live_state["running"] = False
        _go2_stop_event.set()


def _render_worker():
    """Own the EGL context; drop stale states instead of queuing frames."""
    env = test_env
    renderer = None
    try:
        cpu_data = mujoco.MjData(env.mjm)
        renderer = mujoco.Renderer(env.mjm, LIVE_HEIGHT, LIVE_WIDTH)
        nq, nv = env.mjm.nq, env.mjm.nv
        last_sim_seq = -1
        meter_started = time.perf_counter()
        meter_frames = 0

        while not _go2_stop_event.is_set():
            with _go2_state_lock:
                sim_state = _go2_live_state["sim_state"]
                sim_seq = _go2_live_state["sim_seq"]

            if sim_state is None or sim_seq == last_sim_seq:
                _go2_stop_event.wait(0.002)
                continue

            # If control published several states while rendering, this reads
            # only the newest one: latency stays bounded instead of building a queue.
            last_sim_seq = sim_seq
            cpu_data.qpos[:] = sim_state[:nq]
            cpu_data.qvel[:] = sim_state[nq : nq + nv]
            mujoco.mj_forward(env.mjm, cpu_data)
            renderer.update_scene(cpu_data, camera="track")
            jpeg = _jpeg(renderer.render())

            meter_frames += 1
            now = time.perf_counter()
            with _go2_state_lock:
                _go2_live_state["jpeg"] = jpeg
                _go2_live_state["jpeg_seq"] += 1
                if now - meter_started >= 1.0:
                    _go2_live_state["render_hz"] = meter_frames / (
                        now - meter_started
                    )
                    meter_started, meter_frames = now, 0
    except Exception as exc:
        _worker_error("render", exc)
    finally:
        if renderer is not None:
            renderer.close()


async def _go2_ui_loop():
    """Send only the newest encoded frame to Jupyter; never drive physics."""
    last_jpeg_seq = -1
    last_status_update = 0.0
    try:
        while True:
            with _go2_state_lock:
                running = _go2_live_state["running"]
                paused = _go2_live_state["paused"]
                command = _go2_live_state["command"].copy()
                control_hz = _go2_live_state["control_hz"]
                render_hz = _go2_live_state["render_hz"]
                fallen = _go2_live_state["last_fallen"]
                jpeg = _go2_live_state["jpeg"]
                jpeg_seq = _go2_live_state["jpeg_seq"]
                error = _go2_live_state["error"]

            if jpeg is not None and jpeg_seq != last_jpeg_seq:
                # Trait transmission may be slow, but control is on its own thread.
                _go2_image.value = jpeg
                last_jpeg_seq = jpeg_seq

            now = time.perf_counter()
            if now - last_status_update >= 0.2:
                if error:
                    _go2_status.value = (
                        f"<b style='color:#c00'>停止: {error}</b>"
                    )
                else:
                    label = "一時停止中" if paused else (
                        "転倒→自動復帰" if fallen else "実行中"
                    )
                    _go2_status.value = (
                        f"<b>{label}</b>　"
                        f"指令: vx={command[0]:+.2f} m/s, "
                        f"vy={command[1]:+.2f} m/s, "
                        f"wz={command[2]:+.2f} rad/s　"
                        f"制御={control_hz:.1f} Hz / 生成映像={render_hz:.1f} fps"
                    )
                last_status_update = now

            if not running:
                break
            await asyncio.sleep(0.01)
    except asyncio.CancelledError:
        pass
    finally:
        _go2_stop_event.set()
        if _go2_live_state["error"] is None:
            _go2_status.value = "<b>終了しました</b>"


# Repair tensors created by the older inference_mode implementation.
test_env.prev_target = test_env.prev_target.detach().clone()
test_env.last_target = test_env.last_target.detach().clone()
if test_env.renderer is not None:
    test_env.close()
test_env.render_w = LIVE_WIDTH
test_env.render_h = LIVE_HEIGHT
test_env.mjm.vis.global_.offwidth = max(
    test_env.mjm.vis.global_.offwidth, LIVE_WIDTH
)
test_env.mjm.vis.global_.offheight = max(
    test_env.mjm.vis.global_.offheight, LIVE_HEIGHT
)
torch.cuda.synchronize(device)

_go2_control_thread = threading.Thread(
    target=_control_worker, name="go2-control", daemon=True
)
_go2_render_thread = threading.Thread(
    target=_render_worker, name="go2-render", daemon=True
)
_go2_control_thread.start()
_go2_render_thread.start()
_go2_live_task = asyncio.get_running_loop().create_task(_go2_ui_loop())
