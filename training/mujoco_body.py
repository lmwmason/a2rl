"""MuJoCo free-body integration for the simulator's measured actuator model."""
import ctypes as ct

import mujoco
import numpy as np

from .sim import RIGID_STEP


class MujocoBody:
    def __init__(self, lib):
        parameters = (ct.c_float * 6)()
        lib.sim_rigid_parameters(parameters)
        dt, mass, ix, iy, iz, gravity = parameters
        self.model = mujoco.MjModel.from_xml_string(f"""
            <mujoco model="a2rl-free-body">
              <option timestep="{dt}" gravity="0 0 {-gravity}" integrator="Euler"/>
              <worldbody>
                <body name="quadrotor">
                  <freejoint/>
                  <inertial pos="0 0 0" mass="{mass}" diaginertia="{ix} {iy} {iz}"/>
                </body>
              </worldbody>
            </mujoco>
        """)
        self.data = mujoco.MjData(self.model)
        self.vector = np.empty(3)
        self.error = None
        self.callback = RIGID_STEP(self.advance)

    def reset(self):
        self.error = None
        mujoco.mj_resetData(self.model, self.data)

    def advance(self, state_buffer, wrench_buffer, output_buffer):
        try:
            state = np.ctypeslib.as_array(state_buffer, shape=(13,))
            wrench = np.ctypeslib.as_array(wrench_buffer, shape=(6,))
            output = np.ctypeslib.as_array(output_buffer, shape=(13,))
            position, velocity = self.data.qpos, self.data.qvel
            position[:3] = state[:3]
            position[3] = state[9]
            position[4:] = state[6:9]
            velocity[:3] = state[3:6]
            velocity[3:] = state[10:]
            world_wrench = self.data.xfrc_applied[1]
            self.vector[:] = wrench[:3]
            mujoco.mju_rotVecQuat(world_wrench[:3], self.vector, position[3:])
            self.vector[:] = wrench[3:]
            mujoco.mju_rotVecQuat(world_wrench[3:], self.vector, position[3:])
            mujoco.mj_step(self.model, self.data)
            output[:3] = position[:3]
            output[3:6] = velocity[:3]
            output[6:9] = position[4:]
            output[9] = position[3]
            output[10:] = velocity[3:]
            return 0
        except Exception as error:
            self.error = error
            return -1
