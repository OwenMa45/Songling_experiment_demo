from collections import deque
import copy
import numpy as np
import mujoco
from .kinematics import solve, tool_rotation
from .scene import build
from .fluid import Fluid, SimulationCounter


class TitrationEnv:
    def __init__(self, cfg):
        self.cfg = copy.deepcopy(cfg)
        self.s = self.cfg["simulation"]
        self.model = mujoco.MjModel.from_xml_string(build(self.cfg))
        self.data = mujoco.MjData(self.model)
        self.renderer = None
        self.dt = 1 / self.s["control_hz"]
        self.substeps = round(self.dt / self.s["timestep"])
        self.qids = []
        self.aids = []
        for arm in ("holder", "squeezer"):
            self.qids.append([self.model.joint(f"{arm}_joint{i}").qposadr[0] for i in range(1, 9)])
            self.aids.append([self.model.actuator(f"{arm}_joint{i}").id for i in range(1, 9)])
        self.default_cam = self.model.cam_pos.copy()
        self.default_light = self.model.light_diffuse.copy()
        self.default_geom = self.model.geom_pos.copy()
        self.default_size = self.model.geom_size.copy()
        self.reset()

    def site(self, name):
        return self.data.site(name).xpos.copy()

    def ik(self, arm, position, rotation, initial=None):
        ids = self.qids[0 if arm == "holder" else 1][:6]
        limits = np.array([self.model.joint(f"{arm}_joint{i}").range for i in range(1, 7)])
        probe = mujoco.MjData(self.model)
        probe.qpos[:] = self.data.qpos
        def forward(q):
            probe.qpos[ids] = q
            mujoco.mj_forward(self.model, probe)
            site = probe.site(arm + "_tcp")
            return site.xpos.copy(), site.xmat.reshape(3, 3).copy()
        return solve(forward, position, rotation, limits, initial)

    def reset(self, target=3, seed=42, randomized=False):
        if not isinstance(target, int) or target <= 0:
            raise ValueError("target must be a positive integer")
        mujoco.mj_resetData(self.model, self.data)
        rng = np.random.default_rng(seed)
        r = self.s["randomization"]
        shift = rng.uniform(-r["position_m"], r["position_m"], 3) if randomized else np.zeros(3)
        shift[2] = 0
        self.tube = np.array(self.s["tube_center"]) + shift
        self.model.geom_pos[:] = self.default_geom
        self.model.geom_size[:] = self.default_size
        for i in range(self.model.ngeom):
            if (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("tube_"):
                self.model.geom_pos[i] += shift
        self.model.cam_pos[:] = self.default_cam + (rng.uniform(-r["camera_m"], r["camera_m"], self.default_cam.shape) if randomized else 0)
        self.model.light_diffuse[:] = self.default_light * (rng.uniform(1-r["light_fraction"], 1+r["light_fraction"]) if randomized else 1)
        fluid_cfg = dict(self.cfg["fluid"])
        if randomized:
            fluid_cfg["displacement_ml"] *= rng.uniform(1-r["flow_fraction"], 1+r["flow_fraction"])
        self.fluid = Fluid(fluid_cfg)
        self.counter = SimulationCounter(self.fluid)
        self.target = target
        self.last_count = 0
        self.reason = None
        self.stopping_at = None
        self.collisions = []
        # Local X down; arm approaches follow their respective mounting directions.
        holder_rot = tool_rotation(self.s["holder_heading"])
        self.holder_rot = holder_rot
        self.squeeze_rot = tool_rotation(self.s["squeezer_heading"])
        tcp = self.tube.copy()
        tcp[2] = self.s["tip_height"] + 0.075
        self.hold_goal = self.ik("holder", tcp, holder_rot)
        self.hold_start = self.ik("holder", tcp + [-0.035, -0.025, 0.03], holder_rot, self.hold_goal)
        bulb = tcp + [0, 0, 0.055]
        self.squeeze_goal = self.ik("squeezer", bulb, self.squeeze_rot)
        self.squeeze_start = self.ik("squeezer", bulb + [0, 0.065, 0.035], self.squeeze_rot, self.squeeze_goal)
        for ids, q, width in zip(self.qids, (self.hold_start, self.squeeze_start), (0.008, 0.065)):
            self.data.qpos[ids[:6]] = q
            self.data.qpos[ids[6:]] = [width / 2, -width / 2]
        mujoco.mj_forward(self.model, self.data)
        self.command = self.state()
        self.last_action = self.command.copy()
        self.delay = deque([self.command.copy() for _ in range(int(rng.integers(r["delay_steps"]+1)) if randomized else 0)])
        return self.state()

    def state(self):
        return np.concatenate([np.r_[self.data.qpos[ids[:6]], self.data.qpos[ids[6]] - self.data.qpos[ids[7]]] for ids in self.qids]).astype(np.float32)

    def observe(self):
        return {"state": self.state(), "image": self.render("exterior"), "wrist_image": self.render("wrist"), "prompt": f"Dispense {self.target} drops into the fixed test tube using both arms."}

    def stop(self, reason):
        if self.reason is None:
            self.reason = reason
            self.stopping_at = self.data.time
            self.command = self.state().astype(float)
            self.delay.clear()

    @property
    def done(self):
        return self.stopping_at is not None and self.data.time - self.stopping_at >= self.s["settle_seconds"]

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (14,) or not np.isfinite(action).all():
            self.stop("invalid_action")
            action = self.command.copy()
        if self.reason is None:
            self.delay.append(action.copy())
            action = self.delay.popleft()
        previous = self.command.copy()
        if self.reason is None:
            speeds = np.array([self.s["joint_speed"]]*6 + [self.s["gripper_speed"]])
            speeds = np.tile(speeds, 2)
            self.command += np.clip(action - self.command, -speeds*self.dt, speeds*self.dt)
            for index, arm in enumerate(("holder", "squeezer")):
                for j in range(6):
                    lo, hi = self.model.joint(f"{arm}_joint{j+1}").range
                    self.command[index*7+j] = np.clip(self.command[index*7+j], lo, hi)
                self.command[index*7+6] = np.clip(self.command[index*7+6], 0, 0.07)
        self.last_action = self.command.copy()
        for sub in range(self.substeps):
            ctrl = previous + (self.command - previous) * (sub+1)/self.substeps
            if self.reason:
                self.command[13] = min(.065, self.command[13]+self.s["gripper_speed"]*self.s["timestep"])
                ctrl = self.command
            for index, aids in enumerate(self.aids):
                q = ctrl[index*7:index*7+7]
                self.data.ctrl[aids] = np.r_[q[:6], q[6]/2, -q[6]/2]
            # Gravity compensation is explicit; still uses physical position actuators.
            self.data.qfrc_applied[:] = self.data.qfrc_bias
            mujoco.mj_step(self.model, self.data)
            separation = np.linalg.norm(self.site("squeezer_tcp") - self.site("bulb_center"))
            width = self.state()[13]
            # A sideways or tilted tool cannot squeeze by TCP position alone.
            holder_axis = self.data.site("holder_tcp").xmat.reshape(3, 3)[:, 0]
            squeeze_axis = self.data.site("squeezer_tcp").xmat.reshape(3, 3)[:, 0]
            aligned = separation < self.s["alignment_tolerance"] and abs(np.dot(holder_axis, squeeze_axis)) > .95
            compression = max(0, 1-width/self.s["bulb_diameter"]) if aligned else 0
            self.model.geom_size[self.model.geom("bulb").id, 1:] = self.s["bulb_diameter"]/2 * max(.3, 1-compression)
            self.fluid.step(self.s["timestep"], compression, self.site("tip"), self.tube, self.s["tube_radius"], self.data.time)
            for contact in self.data.contact:
                a, b = (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or "" for g in (contact.geom1, contact.geom2))
                # MuJoCo already excludes adjacent bodies; non-adjacent self-contact counts.
                holder_mount = (a == "glass" and b.startswith("holder_")) or (b == "glass" and a.startswith("holder_"))
                base_table = "table" in (a,b) and any("base_link" in n for n in (a,b))
                if not (holder_mount or base_table) and contact.dist < -0.001:
                    self.collisions.append((a, b))
                    self.stop("collision")
            try:
                count = self.counter.read()
                if not isinstance(count, (int, np.integer)) or count < self.last_count:
                    raise ValueError("Counter must be a monotonic nonnegative integer")
                self.last_count = count
                if count >= self.target:
                    self.stop("target_reached")
            except Exception:
                self.stop("counter_error")
            if self.fluid.spilled:
                self.stop("spill")
            if self.data.time >= self.s["timeout"]:
                self.stop("timeout")
        return self.report()

    def report(self):
        return {"target": self.target, "received": self.fluid.received, "emitted": self.fluid.emitted, "spilled": self.fluid.spilled, "reason": self.reason, "time": self.data.time, "collisions": list(set(self.collisions)), "success": bool(self.done and self.reason == "target_reached" and self.fluid.received == self.target and not self.fluid.spilled and not self.collisions)}

    def render(self, camera="exterior"):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=self.s["image_size"], width=self.s["image_size"])
        self.renderer.update_scene(self.data, camera=camera)
        scene = self.renderer.scene
        for arrival, hit, pos, start in self.fluid.pending:
            if scene.ngeom >= scene.maxgeom:
                break
            p = np.array(pos) - [0, 0, 0.5*9.81*(self.data.time-start)**2]
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.002]*3), p, np.eye(3).flatten(), np.array([0.1, 0.5, 1, 1], dtype=np.float32))
            scene.ngeom += 1
        return self.renderer.render().copy()

    def close(self):
        if self.renderer:
            self.renderer.close()
