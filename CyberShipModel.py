#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The file used to describe the dynamic of quadrotor UAV

By xiaobo
Contact linxiaobo110@gmail.com
Created on Thr August 22 22:51:44 2019
"""

# Copyright (C)
#
# This file is part of QuadrotorFly
#
# GWpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GWpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GWpy.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import enum
from enum import Enum
import MemoryStore
import SensorBase

"""
********************************************************************************************************
**-------------------------------------------------------------------------------------------------------
**  Compiler   : python 3.6
**  Module Name: CyberShipModel
**  Module Date: 2019-08-22
**  Module Auth: xiaobo
**  Version    : V0.1
**  Description: create the module
**-------------------------------------------------------------------------------------------------------
**  Reversion  :
**  Modified By:
**  Date       :
**  Content    :
**  Notes      :
********************************************************************************************************/
"""

# definition of key constant
D2R = np.pi / 180
state_dim = 6
action_dim = 3
action_bound = np.array([1, 1, 1])


def rk4(func, x0, action, h):
    """Runge Kutta 4 order update function
    :param func: system dynamic
    :param x0: system state
    :param action: control input
    :param h: time of sample
    :return: state of next time
    """
    k1 = func(x0, action)
    k2 = func(x0 + h * k1 / 2, action)
    k3 = func(x0 + h * k2 / 2, action)
    k4 = func(x0 + h * k3, action)
    # print('rk4 debug: ', k1, k2, k3, k4)
    x1 = x0 + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x1


class ModelPara(object):
    """Define the parameters of quadrotor model

    """

    def __init__(self, g=9.81, rotor_num=4, tim_sample=0.01):
        """init the cyberShip parameters
        These parameters are able to be estimation in web(https://flyeval.com/) if you do not have a real UAV.
        common parameters:
            -g         : s,                time para of dynamic response of motor
        """
        self.M = np.array([[25.8, 0, 0], [0, 33.8, 6.2], [0, 6.2, 2.76]])
        self.ts = tim_sample
        self.test = 1


class SimInitType(Enum):
    rand = enum.auto()
    fixed = enum.auto()


class SimPara(object):
    """contain the parameters for guiding the simulation process
    """

    def __init__(self, init_mode=SimInitType.rand, init_eta=np.array([5., 5., 0.]), init_vi=np.array([1, 1, 0]),
                 enable_sensor_sys=False):
        """ init the parameters for simulation process, focus on conditions during an episode
        :param init_mode:
        :param init_eta:
        :param init_vi:
        :param enable_sensor_sys: whether the sensor system is enable, including noise and bias of sensor
        """
        self.initMode = init_mode
        self.initEta = init_eta
        self.initEta[2] = D2R * init_eta[2]
        self.initVi = init_vi
        self.initVi[2] = D2R * init_vi[2]
        self.enableSensorSys = enable_sensor_sys


class CyberShipModel(object):
    """module interface, main class including basic dynamic of quad
    """

    def __init__(self, model_para: ModelPara, sim_para: SimPara):
        """init a cyber ship
        :param model_para:    parameters of model,maintain together
        :param sim_para:    'simple', without dynamic of motor; 'dynamic' with dynamic;
        """
        self.modelPara = model_para
        self.simPara = sim_para

        # states of quadrotor
        #   -position x, position y, direction phy
        self.eta = np.array([0, 0, 0])
        #   -velocity u, velocity v, rotation rate
        self.vi = np.array([0, 0, 0])
        # accelerate, m/(s^2)
        self.acc = np.zeros(3)

        # time control, s
        self.__ts = 0

        # initial the sensors
        if self.simPara.enableSensorSys:
            self.sensorList = list()
            # self.imu0 = SensorImu.SensorImu()
            # self.sensorList.append(self.imu0)

        # initial the states
        self.reset_states()

    @property
    def ts(self):
        """return the tick of system"""
        return self.__ts

    def generate_init_vi(self):
        """used to generate a init attitude according to simPara"""
        init_vi = self.simPara.initVi
        if self.simPara.initMode == SimInitType.rand:
            phi = (1 * np.random.random() - 0.5) * init_vi[0]
            theta = (1 * np.random.random() - 0.5) * init_vi[1]
            psi = (1 * np.random.random() - 0.5) * init_vi[2]
        else:
            phi = init_vi[0]
            theta = init_vi[1]
            psi = init_vi[2]
        return np.array([phi, theta, psi])

    def generate_init_eta(self):
        """used to generate a init position according to simPara"""
        pos = self.simPara.initEta
        if self.simPara.initMode == SimInitType.rand:
            x = (1 * np.random.random() - 0.5) * pos[0]
            y = (1 * np.random.random() - 0.5) * pos[1]
            z = (1 * np.random.random() - 0.5) * pos[2]
        else:
            x = pos[0]
            y = pos[1]
            z = pos[2]
        return np.array([x, y, z])

    def reset_states(self, eta='none', vi='none'):
        self.__ts = 0
        if isinstance(eta, str):
            self.eta = self.generate_init_eta()
        else:
            self.eta = eta
            self.eta[2] = eta[2] * D2R

        if isinstance(vi, str):
            self.vi = self.generate_init_vi()
        else:
            self.vi = vi
            self.vi[2] = vi[2] * D2R

        # sensor system reset
        if self.simPara.enableSensorSys:
            for sensor in self.sensorList:
                sensor.reset(self.state)

    def dynamic_basic(self, state, action):
        """ calculate /dot(state) = f(state) + u(state)
        :param state:
        This function will be executed many times during simulation, so high performance is necessary. :param state:
            0       1       2       3       4       5
            p_x     p_y     phy     v_u     v_v     r
        :param action: u1(torque on u), u2(torque on v), u3(torque on r)
        :return: derivatives of state inclfrom bokeh.plotting import figure
        """
        # variable used repeatedly
        # M = np.array([[25.8, 0, 0], [0, 33.8, 6.2], [0, 6.2, 2.76]])
        # state[4] = 0
        m_inv = np.array([[0.03875969,  0.,  0.],
                          [0.,  0.05032089, -0.11303967],
                          [0., -0.11303967,  0.61624854]])

        c = np.array([[0, 0, -33.8 * state[4] + 6.2 * state[5]],
                      [0, 0, 25.8 * state[3]],
                      [33.8 * state[4] - 6.2 * state[5], -25.8 * state[3], 0]])
        d = np.array([[12 + 2.5 * np.abs(state[3]), 0, 0],
                      [0, 17 + 4.5 * np.abs(state[4]), 0.2],
                      [0, 0.5, 0.5 + 0.1 * np.abs(state[5])]])
        rot = np.array([[np.cos(state[2]), -np.sin(state[2]), 0],
                        [np.sin(state[2]), np.cos(state[2]), 0],
                       [0, 0, 1]])
        # state[4] = 0
        nu = state[3:6]
        action[1] = 0
        t2 = action

        dot_state = np.zeros([state_dim])
        # dynamic of position cycle
        dot_state[0:3] = np.dot(rot, nu)
        dot_state[3:6] = np.dot(m_inv, t2 - c.dot(nu) - d.dot(nu))
        # print(t2,  c.dot(nu), d.dot(nu))

        ''' Just used for test
        temp1 = state[10] * state[11] * (para.uavInertia[1] - para.uavInertia[2]) / para.uavInertia[0]
        temp2 = - para.rotorInertia / para.uavInertia[0] * state[10] * rotor_rate_sum
        temp3 = + para.uavL * action[1] / para.uavInertia[0]
        print('dyanmic Test', temp1, temp2, temp3, action)
       '''
        return dot_state

    def observe(self):
        """out put the system state, with sensor system or without sensor system"""
        if self.simPara.enableSensorSys:
            sensor_data = dict()
            for index, sensor in enumerate(self.sensorList):
                if isinstance(sensor, SensorBase.SensorBase):
                    # name = str(index) + '-' + sensor.get_name()
                    name = sensor.get_name()
                    sensor_data.update({name: sensor.observe()})
            return sensor_data
        else:
            return np.hstack([self.eta, self.vi])

    @property
    def state(self):
        return np.hstack([self.eta, self.vi])

    def is_finished(self):
        # if (np.max(np.abs(self.position)) < self.simPara.maxPosition)\
        #         and (np.max(np.abs(self.velocity) < self.simPara.maxVelocity))\
        #         and (np.max(np.abs(self.attitude) < self.simPara.maxAttitude))\
        #         and (np.max(np.abs(self.angular) < self.simPara.maxAngular)):
        #     return False
        # else:
        #     return True
        if self.modelPara.test > 0:
            return False

    def get_reward(self):
        reward = np.sum(np.square(self.eta)) / 8 + np.sum(np.square(self.vi)) / 20
        return reward

    def step(self, action: 'int > 0'):

        self.__ts += self.modelPara.ts
        # 1.1 Actuator model, calculate the thrust and torque

        # 1.1 Basic model, calculate the basic model, the u need to be given directly in test-mode for Matlab
        state_temp = np.hstack([self.eta, self.vi])
        state_next = rk4(self.dynamic_basic, state_temp, action, self.modelPara.ts)
        [self.eta, self.vi] = np.split(state_next, 2)
        # calculate the accelerate
        # state_dot = self.dynamic_basic(state_temp, forces)
        # self.acc = state_dot[3:6]

        # 2. Calculate Sensor sensor model
        if self.simPara.enableSensorSys:
            for index, sensor in enumerate(self.sensorList):
                if isinstance(sensor, SensorBase.SensorBase):
                    sensor.update(np.hstack([state_next, self.acc]), self.__ts)
        ob = self.observe()

        # 3. Check whether finish (failed or completed)
        finish_flag = self.is_finished()

        # 4. Calculate a reference reward
        reward = self.get_reward()

        return ob, reward, finish_flag

    # def get_controller_pid(self, state, ref_state=np.array([0, 0, 1, 0])):
    #     """ pid controller
    #     :param state: system state, 12
    #     :param ref_state: reference value for x, y, z, yaw
    #     :return: control value for four motors
    #     """
    #
    #     # position-velocity cycle, velocity cycle is regard as kd
    #     kp_pos = np.array([0.3, 0.3, 0.8])
    #     kp_vel = np.array([0.15, 0.15, 0.5])
    #     # decoupling about x-y
    #     phy = state[8]
    #     # de_phy = np.array([[np.sin(phy), -np.cos(phy)], [np.cos(phy), np.sin(phy)]])
    #     # de_phy = np.array([[np.cos(phy), np.sin(phy)], [np.sin(phy), -np.cos(phy)]])
    #     de_phy = np.array([[np.cos(phy), -np.sin(phy)], [np.sin(phy), np.cos(phy)]])
    #     err_pos = ref_state[0:3] - np.array([state[0], state[1], state[2]])
    #     ref_vel = err_pos * kp_pos
    #     err_vel = ref_vel - np.array([state[3], state[4], state[5]])
    #     # calculate ref without decoupling about phy
    #     # ref_angle = kp_vel * err_vel
    #     # calculate ref with decoupling about phy
    #     ref_angle = np.zeros(3)
    #     ref_angle[0:2] = np.matmul(de_phy, kp_vel[0] * err_vel[0:2])
    #
    #     # attitude-angular cycle, angular cycle is regard as kd
    #     kp_angle = np.array([1.0, 1.0, 0.8])
    #     kp_angular = np.array([0.2, 0.2, 0.2])
    #     # ref_angle = np.zeros(3)
    #     err_angle = np.array([-ref_angle[1], ref_angle[0], ref_state[3]]) - np.array([state[6], state[7], state[8]])
    #     ref_rate = err_angle * kp_angle
    #     err_rate = ref_rate - [state[9], state[10], state[11]]
    #     con_rate = err_rate * kp_angular
    #
    #     # the control value in z direction needs to be modify considering gravity
    #     err_altitude = (ref_state[2] - state[2]) * 0.5
    #     con_altitude = (err_altitude - state[5]) * 0.25
    #     oil_altitude = 0.6 + con_altitude
    #     if oil_altitude > 0.75:
    #         oil_altitude = 0.75
    #
    #     action_motor = np.zeros(4)
    #     if self.uavPara.structureType == StructureType.quad_plus:
    #         action_motor[0] = oil_altitude - con_rate[0] - con_rate[2]
    #         action_motor[1] = oil_altitude + con_rate[0] - con_rate[2]
    #         action_motor[2] = oil_altitude - con_rate[1] + con_rate[2]
    #         action_motor[3] = oil_altitude + con_rate[1] + con_rate[2]
    #     elif self.uavPara.structureType == StructureType.quad_x:
    #         action_motor[0] = oil_altitude - con_rate[2] - con_rate[1] - con_rate[0]
    #         action_motor[1] = oil_altitude - con_rate[2] + con_rate[1] + con_rate[0]
    #         action_motor[2] = oil_altitude + con_rate[2] - con_rate[1] + con_rate[0]
    #         action_motor[3] = oil_altitude + con_rate[2] + con_rate[1] - con_rate[0]
    #     else:
    #         action_motor = np.zeros(4)
    #
    #     action_pid = action_motor
    #     return action_pid, oil_altitude


class GuideLine(object):
    """contain the functions for guiderline calculation
    """
    def __init__(self, p1=np.array([0, 0]), p2=np.array([0, 0])):
        self.k = (p2[1] - p1[1]) / (p2[0] - p1[0])  # real tan(theta)
        self.b = self.k * p1[0] - p1[1]
        self.theta = np.arctan(self.k)

    def dis_line2point(self, px, py):
        dis_result = (-self.k * px + py + self.b) / np.sqrt(self.k * self.k + 1)
        return dis_result


if __name__ == '__main__':
    " used for testing this module"
    testFlag = 3

    if testFlag == 1:
        pass
    elif testFlag == 2:
        pass
    elif testFlag == 3:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        print("PID  controller test: ")
        mPara = ModelPara()
        sPara = SimPara(init_mode=SimInitType.fixed, init_eta=np.array([0., 0., 0.]), init_vi=np.array([0, 0, 0]))
        ship1 = CyberShipModel(mPara, sPara)
        ship1.reset_states()
        record = MemoryStore.DataRecord()
        record.clear()
        step_cnt = 0

        # 0.1 prepare the guider line
        guid_p1 = np.array([0, 0])
        guid_p2 = np.array([2, 2])
        gl1 = GuideLine(guid_p1, guid_p2)
        # print(guid_k, guid_b)

        # control gain
        guid_kd = 0.3
        vi_i = 0  # i on v_u
        for i in range(5000):
            stateTemp = ship1.observe()
            # action2, oil = ship1.get_controller_pid(stateTemp, ref)

            # 1.1 calculate the v in world frame according to guider-line
            guid_d = gl1.dis_line2point(stateTemp[0], stateTemp[1])
            v_d = -guid_d * guid_kd
            # print(guid_d)
            v0 = 1
            v_X = -v_d * np.cos(gl1.theta) + v0 * np.sin(gl1.theta)
            v_Y = v_d * np.sin(gl1.theta) + v0 * np.cos(gl1.theta)
            # considering the effect of v_v
            vxx = v_X + stateTemp[4] * np.sin(stateTemp[2])
            vyy = v_Y - stateTemp[4] * np.cos(stateTemp[2])

            # 1.2 calculate the v in body frame according to the v_world
            v_u = np.sqrt(vxx * vxx + vyy * vyy) * np.sign(vxx) * np.sign(np.cos(stateTemp[2]))
            v_u = np.clip(v_u, -1.3, 1.3)
            phi = np.arctan2(vyy, vxx)
            print(guid_d, v_d, phi / D2R)

            # 1.3 calculate the final action
            action = np.zeros(3)
            error = v_u - stateTemp[3]
            vi_i = vi_i + error
            action[0] = error * 3 + vi_i * 0.016
            phi_dot = (phi - stateTemp[2]) * 2
            action[2] = (phi_dot - stateTemp[5]) * 20

            # print(phi, phi_dot, stateTemp[2], stateTemp[5])
            # action2 = np.random.rand(3)
            # print('action: ', action)
            # action2 = np.clip(action2, 0.1, 0.9)
            ship1.step(action)
            # test
            action[1] = phi
            stateTemp[4] = v_u
            # stateTemp[4] = phi_dot
            record.buffer_append((stateTemp, action))
            step_cnt = step_cnt + 1
        record.episode_append()

        data = record.get_episode_buffer()
        bs = data[0]
        ba = data[1]
        t = range(0, record.count)
        # mpl.style.use('seaborn')
        fig1 = plt.figure(1)
        plt.clf()
        plt.subplot(3, 1, 1)
        tss = range(500)
        plt.plot(bs[t, 0], bs[t, 1], label='xy')
        plt.plot(np.arange(0, 17, 1), np.arange(0, 17, 1), '--r', label='line')
        # plt.plot(t, bs[t, 1], label='y')
        # plt.plot(t, bs[t, 2] / D2R, label='yaw')
        plt.ylabel('Eta $(\circ)$', fontsize=15)
        plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
        plt.subplot(3, 1, 2)
        # plt.plot(t, bs[t, 3], label='u')
        plt.plot(t, bs[t, 3], label='v_real')
        plt.plot(t, bs[t, 4], label='v_ref')
        plt.ylabel('Vi (m)', fontsize=15)
        plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
        plt.subplot(3, 1, 3)
        plt.plot(t, bs[t, 2] / D2R, label='phi_real')
        plt.plot(t, ba[t, 1] / D2R, label='phi_ref')
        plt.ylabel('Altitude (m)', fontsize=15)
        plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
        plt.show()

        # plt.subplot(3, 1, 1)
        # plt.plot(t, bs[t, 0], label='x')
        # plt.plot(t, bs[t, 1], label='y')
        # plt.plot(t, bs[t, 2] / D2R, label='yaw')
        # plt.ylabel('Eta $(\circ)$', fontsize=15)
        # plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
        # plt.subplot(3, 1, 2)
        # plt.plot(t, bs[t, 3], label='u')
        # plt.plot(t, bs[t, 4], label='v')
        # plt.plot(t, bs[t, 5], label='r')
        # plt.ylabel('Vi (m)', fontsize=15)
        # plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
        # plt.subplot(3, 1, 3)
        # plt.plot(t, ba[t, 0], label='z')
        # plt.plot(t, ba[t, 2], label='z')
        # plt.ylabel('Altitude (m)', fontsize=15)
        # plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
        # plt.show()
