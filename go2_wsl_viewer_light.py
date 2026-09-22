#!/usr/bin/env python3
"""Low-overhead WSLg viewer for the CPU Go2 policy evaluator.

Unlike ``mujoco.viewer.launch_passive``, this program owns a small GLFW window
and renders only at the requested rate.  Policy inference and MuJoCo physics
remain fixed at 50 Hz while drawing defaults to 25 fps at 640 x 360.

The policy, observation, reset, and physics implementation are imported from
``go2_wsl_viewer.py`` so both viewers execute the same controller.
"""

from __future__ import annotations

import os
import sys
import time

# These must be configured before importing MuJoCo or GLFW.  The caller may
# override GALLIUM_DRIVER; d3d12 is known to work on the target WSLg machine.
os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("GALLIUM_DRIVER", "d3d12")
os.environ.pop("PYOPENGL_PLATFORM", None)

import glfw
import mujoco
import numpy as np
import torch

from go2_wsl_viewer import (
    CMD_SCALE,
    CONTROL_HZ,
    MEM_DIM,
    TERRAIN_LEVELS,
    TERRAIN_TYPES,
    TERRAIN_TYPE_NAMES,
    Go2CpuEnv,
    build_parser as build_base_parser,
    load_actor,
    load_normalizer,
    resolve_input_path,
)


def build_parser():
    parser = build_base_parser()
    parser.description = (
        "Run Go2 with 50 Hz CPU control and a low-overhead WSLg window"
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument(
        "--render-hz",
        type=float,
        default=25.0,
        help="visual update rate; control remains fixed at 50 Hz",
    )
    parser.add_argument(
        "--vsync",
        action="store_true",
        help="enable vsync (smoother display but may delay 50 Hz control)",
    )
    return parser


def disable_expensive_rendering(scene: mujoco.MjvScene) -> None:
    """Disable effects that add little value to the control preview."""
    for name in (
        "mjRND_SHADOW",
        "mjRND_REFLECTION",
        "mjRND_FOG",
        "mjRND_HAZE",
        "mjRND_SKYBOX",
    ):
        flag = getattr(mujoco.mjtRndFlag, name, None)
        if flag is not None:
            scene.flags[int(flag)] = 0


def main() -> None:
    args = build_parser().parse_args()
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        raise RuntimeError(
            "DISPLAY is not set. Run from the local WSL2/WSLg terminal."
        )
    if args.width < 160 or args.height < 120:
        raise ValueError("--width must be >=160 and --height must be >=120")
    if not 1.0 <= args.render_hz <= CONTROL_HZ:
        raise ValueError(f"--render-hz must be between 1 and {CONTROL_HZ:g}")
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be at least 1")

    xml_path = resolve_input_path(args.xml)
    checkpoint_path = resolve_input_path(args.checkpoint)
    norm_path = resolve_input_path(args.norm)
    for label, path in (
        ("XML", xml_path),
        ("checkpoint", checkpoint_path),
        ("normalizer", norm_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")

    torch.set_num_threads(args.torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(args.seed)

    actor = load_actor(checkpoint_path)
    normalize = load_normalizer(norm_path, actor.obs_dim)
    env = Go2CpuEnv(
        xml_path,
        max_steps=args.max_steps,
        preset=args.preset,
        reset_noise=args.reset_noise,
        seed=args.seed,
        actor_obs_dim=actor.obs_dim,
        prior_factor=args.prior_factor,
        terrain_type=args.terrain_type,
        terrain_level=args.terrain_level,
        terrain_seed=args.terrain_seed,
    )

    if not glfw.init():
        raise RuntimeError("GLFW initialization failed")

    # No multisampling and no UI panels: prioritize latency over visual polish.
    glfw.window_hint(glfw.SAMPLES, 0)
    glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
    window = glfw.create_window(args.width, args.height, "Go2 CPU - light", None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("Could not create the GLFW window")

    glfw.make_context_current(window)
    glfw.swap_interval(1 if args.vsync else 0)

    camera = mujoco.MjvCamera()
    option = mujoco.MjvOption()
    scene = mujoco.MjvScene(env.model, maxgeom=1000)
    context = mujoco.MjrContext(
        env.model, mujoco.mjtFontScale.mjFONTSCALE_100
    )
    camera_id = (
        -1
        if args.camera == "free"
        else mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
    )
    if camera_id >= 0 and args.camera != "free":
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = camera_id
    else:
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.azimuth = 90.0
        camera.elevation = -12.0
        camera.distance = 2.0

    state = {
        "command": np.clip(
            np.asarray(args.start, dtype=np.float32), -CMD_SCALE, CMD_SCALE
        ),
        "changed": True,
        "reset": True,
        "paused": False,
        "terrain_type": env.terrain_type,
        "terrain_level": env.terrain_level,
    }
    mouse = {"x": 0.0, "y": 0.0, "left": False, "right": False}

    def update_title() -> None:
        command = state["command"]
        status = "PAUSED" if state["paused"] else "RUN"
        terrain_name = (
            TERRAIN_TYPE_NAMES[state["terrain_type"]]
            if env.has_terrain
            else "flat"
        )
        glfw.set_window_title(
            window,
            f"Go2 {status} | vx {command[0]:+.2f}  vy {command[1]:+.2f}  "
            f"wz {command[2]:+.2f} | {terrain_name} "
            f"L{state['terrain_level']} | 50 Hz / {args.render_hz:g} fps",
        )

    def print_command(prefix: str = "command") -> None:
        command = state["command"]
        print(
            f"{prefix}: vx={command[0]:+.2f} m/s  vy={command[1]:+.2f} m/s  "
            f"wz={command[2]:+.2f} rad/s",
            flush=True,
        )

    def print_terrain() -> None:
        if env.has_terrain:
            terrain_name = TERRAIN_TYPE_NAMES[state["terrain_type"]]
            print(
                f"terrain: {terrain_name}  level={state['terrain_level']}",
                flush=True,
            )
        else:
            print("terrain: flat (terrain selection is unavailable)", flush=True)

    def key_callback(_window, key, _scancode, action, _mods) -> None:
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        command = state["command"]
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, glfw.TRUE)
            return
        if key == glfw.KEY_W:
            command[0] += args.vx_step
        elif key == glfw.KEY_S:
            command[0] -= args.vx_step
        elif key == glfw.KEY_A:
            command[1] += args.vy_step
        elif key == glfw.KEY_D:
            command[1] -= args.vy_step
        elif key == glfw.KEY_Q:
            command[2] += args.wz_step
        elif key == glfw.KEY_E:
            command[2] -= args.wz_step
        elif key == glfw.KEY_X:
            command[:] = 0.0
        elif key == glfw.KEY_1:
            command[:] = (CMD_SCALE[0], 0.0, 0.0)
        elif key == glfw.KEY_2:
            command[:] = (-CMD_SCALE[0], 0.0, 0.0)
        elif key == glfw.KEY_3:
            command[:] = (0.0, CMD_SCALE[1], 0.0)
        elif key == glfw.KEY_4:
            command[:] = (0.0, -CMD_SCALE[1], 0.0)
        elif key == glfw.KEY_5:
            command[:] = (0.0, 0.0, CMD_SCALE[2])
        elif key == glfw.KEY_6:
            command[:] = (0.0, 0.0, -CMD_SCALE[2])
        elif key == glfw.KEY_R:
            state["reset"] = True
            return
        elif key == glfw.KEY_T:
            if not env.has_terrain:
                print_terrain()
                return
            state["terrain_type"] = (
                state["terrain_type"] + 1
            ) % TERRAIN_TYPES
            state["reset"] = True
            print_terrain()
            update_title()
            return
        elif key == glfw.KEY_LEFT_BRACKET:
            if not env.has_terrain:
                print_terrain()
                return
            state["terrain_level"] = max(0, state["terrain_level"] - 1)
            state["reset"] = True
            print_terrain()
            update_title()
            return
        elif key == glfw.KEY_RIGHT_BRACKET:
            if not env.has_terrain:
                print_terrain()
                return
            state["terrain_level"] = min(
                TERRAIN_LEVELS - 1, state["terrain_level"] + 1
            )
            state["reset"] = True
            print_terrain()
            update_title()
            return
        elif key in (glfw.KEY_P, glfw.KEY_SPACE):
            if action == glfw.PRESS:
                state["paused"] = not state["paused"]
                print("paused" if state["paused"] else "running", flush=True)
                update_title()
            return
        else:
            return

        np.clip(command, -CMD_SCALE, CMD_SCALE, out=command)
        state["changed"] = True
        print_command()
        update_title()

    def mouse_button_callback(_window, button, action, _mods) -> None:
        if button == glfw.MOUSE_BUTTON_LEFT:
            mouse["left"] = action == glfw.PRESS
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            mouse["right"] = action == glfw.PRESS
        mouse["x"], mouse["y"] = glfw.get_cursor_pos(window)

    def cursor_callback(_window, xpos, ypos) -> None:
        dx, dy = xpos - mouse["x"], ypos - mouse["y"]
        mouse["x"], mouse["y"] = xpos, ypos
        if mouse["left"]:
            camera.azimuth -= 0.35 * dx
            camera.elevation = float(np.clip(camera.elevation - 0.35 * dy, -89, 20))
        elif mouse["right"]:
            camera.distance = float(np.clip(camera.distance * np.exp(0.01 * dy), 0.4, 8.0))

    def scroll_callback(_window, _xoffset, yoffset) -> None:
        camera.distance = float(
            np.clip(camera.distance * np.exp(-0.12 * yoffset), 0.4, 8.0)
        )

    glfw.set_key_callback(window, key_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    update_title()

    def render() -> None:
        width, height = glfw.get_framebuffer_size(window)
        if width <= 0 or height <= 0:
            return
        # Free camera follows manually; fixed side/track cameras use trackcom.
        if camera.type == mujoco.mjtCamera.mjCAMERA_FREE:
            camera.lookat[:] = env.data.qpos[:3]
        mujoco.mjv_updateScene(
            env.model,
            env.data,
            option,
            None,
            camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            scene,
        )
        disable_expensive_rendering(scene)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)

    print(
        "Keys: W/S=vx  A/D=vy  Q/E=yaw  X=stop  1..6=presets  "
        "T=terrain  [ / ]=level  R=reset  Space/P=pause  Esc=quit",
        flush=True,
    )
    print(
        f"control={CONTROL_HZ:.0f} Hz  render={args.render_hz:g} fps  "
        f"window={args.width}x{args.height}  preset={args.preset}  "
        f"torch_threads={args.torch_threads}  actor_obs={actor.obs_dim}  "
        f"prior={args.prior_factor:.2f}  camera={args.camera}",
        flush=True,
    )
    print_terrain()

    command = state["command"].copy()
    observation = env.reset(command)
    membrane = torch.randn(1, MEM_DIM, dtype=torch.float32)
    state["reset"] = False
    state["changed"] = False

    control_period = env.dt
    render_period = 1.0 / args.render_hz
    now = time.perf_counter()
    next_control = now
    next_render = now
    report_start = now
    control_steps = 0
    render_frames = 0
    late_steps = 0

    try:
        while not glfw.window_should_close(window):
            glfw.poll_events()
            now = time.perf_counter()

            if state["reset"]:
                env.set_terrain(state["terrain_type"], state["terrain_level"])
                observation = env.reset(state["command"])
                membrane = torch.randn(1, MEM_DIM, dtype=torch.float32)
                state["reset"] = False
                state["changed"] = False
                print_command("reset")
                update_title()
            elif state["changed"]:
                env.set_command(state["command"])
                observation = env.observation()
                state["changed"] = False

            if not state["paused"] and now >= next_control:
                if now - next_control > control_period * 0.5:
                    late_steps += 1
                with torch.no_grad():
                    action, membrane = actor(normalize(observation), membrane)
                    action = action.clamp(-1.0, 1.0)
                observation, done, fallen = env.step(action)
                control_steps += 1
                if done:
                    if fallen:
                        print("fallen -> automatic reset", flush=True)
                    observation = env.reset(state["command"])
                    membrane = torch.randn(1, MEM_DIM, dtype=torch.float32)

                next_control += control_period
                if next_control < time.perf_counter() - control_period:
                    next_control = time.perf_counter()
            elif state["paused"]:
                next_control = now

            now = time.perf_counter()
            if now >= next_render:
                render()
                render_frames += 1
                next_render += render_period
                if next_render < time.perf_counter() - render_period:
                    next_render = time.perf_counter()

            now = time.perf_counter()
            if args.report_interval > 0 and now - report_start >= args.report_interval:
                elapsed = now - report_start
                control_rate = control_steps / elapsed
                render_rate = render_frames / elapsed
                print(
                    f"control: {control_rate:5.1f} Hz  render: {render_rate:4.1f} fps  "
                    f"late: {late_steps}",
                    flush=True,
                )
                report_start = now
                control_steps = render_frames = late_steps = 0

            deadline = next_render if state["paused"] else min(next_control, next_render)
            delay = deadline - time.perf_counter()
            if delay > 0:
                time.sleep(min(delay, 0.002))
    finally:
        glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    main()
