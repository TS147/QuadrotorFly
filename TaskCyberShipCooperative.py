#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""complete the cooperative task for 3-cybership

By xiaobo, wang qing
Contact linxiaobo110@gmail.com
Created on  八月 24 14:57 2019
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
import matplotlib.pyplot as plt
import MemoryStore
import time
import CommonFunctions as Cf
from enum import Enum
# import imp
import pickle
import time
import os
import CyberShipModel as Csm

"""
********************************************************************************************************
**-------------------------------------------------------------------------------------------------------
**  Compiler   : python 3.6
**  Module Name: TaskCybershipCooperativ
**  Module Date: 2019/8/24
**  Module Auth: wang qing, xiaobo
**  Version    : V0.1
**  Description: 'Replace the content between'
**-------------------------------------------------------------------------------------------------------
**  Reversion  :
**  Modified By:
**  Date       :
**  Content    :
**  Notes      :
********************************************************************************************************/
"""

D2R = Csm.D2R


class TaskCybershipCooperative(object):
    """Task for 3-cyberships with pid
    """
    def __init__(self):
        # 1. define the ships
        model_para = Csm.ModelPara()
        sim_para = Csm.SimPara(init_mode=Csm.SimInitType.fixed, init_eta=np.array([0., 0., 0.]),
                               init_vi=np.array([0, 0, 0]))
        self.ships = list()
        ship1 = Csm.CyberShipModel(model_para, sim_para)
        ship2 = Csm.CyberShipModel(model_para, sim_para)
        ship3 = Csm.CyberShipModel(model_para, sim_para)
        self.ships.append(ship1)
        self.ships.append(ship2)
        self.ships.append(ship3)

        # 2. define the guider-line
        # 0.1 prepare the guider line
        guid_p1 = np.array([0, 0])
        guid_p2 = np.array([2, 2])
        self.guidLine = Csm.GuideLine(guid_p1, guid_p2)

        # 3. init the memore record
        self.record = MemoryStore.DataRecord()
        self.record.clear()

    @classmethod
    def vec_bet2points(cls, pos=np.array([0, 0]), tgt=np.array([1, 1])):
        dy = tgt[1] - pos[1]
        dx = tgt[0] - pos[0]
        dis = np.sqrt(dx * dx + dy * dy)
        theta = np.arctan2(dy, dx)
        return dis, theta

    @classmethod
    def follower_guidance(cls, tgt_pos, tgt_v0, tgt_yaw, state, ):
        guid_kd1 = 0.2
        # calculate the distance and theta
        tgt_dis, tgt_theta = TaskCybershipCooperative.vec_bet2points(np.array([state[0], state[1]]), tgt_pos)
        # print(tgt_dis)
        # calculate the target velocity and angle
        v_d = tgt_dis * guid_kd1
        v_x = v_d * np.cos(tgt_theta) + tgt_v0 * np.sin(tgt_yaw)
        v_y = v_d * np.sin(tgt_theta) + tgt_v0 * np.cos(tgt_yaw)

        # considering the effect of v_v
        vxx = v_x + state[4] * np.sin(state[2])
        vyy = v_y - state[4] * np.cos(state[2])

        # calculate the v in body frame according to the v_world
        v_u = np.sqrt(vxx * vxx + vyy * vyy) * np.sign(vxx) * np.sign(np.cos(state[2]))
        v_u = np.clip(v_u, -1.3, 1.3)
        phi = np.arctan2(vyy, vxx)
        return v_u, phi

    def task_formation(self):
        agents = self.ships
        self.record.clear()
        # control gain
        guid_kd = 0.3
        # guid_kd1 = 0.2
        step_cnt = 0

        # 队形控制
        formation_ship1 = np.array([-2., 2.])
        formation_ship2 = np.array([2., -2.])
        for i in range(5000):
            state_temp = agents[0].observe()
            state_temp1 = agents[1].observe()
            state_temp2 = agents[2].observe()
            # action2, oil = ship1.get_controller_pid(stateTemp, ref)

            # 1 the agent[0] is the leader,
            # 1.1 calculate the v in world frame according to guider-line
            guid_d = self.guidLine.dis_line2point(state_temp[0], state_temp[1])
            v_d = -guid_d * guid_kd
            # print(guid_d)
            v0 = 1
            v_X = -v_d * np.cos(self.guidLine.theta) + v0 * np.sin(self.guidLine.theta)
            v_Y = v_d * np.sin(self.guidLine.theta) + v0 * np.cos(self.guidLine.theta)
            # considering the effect of v_v
            vxx = v_X + state_temp[4] * np.sin(state_temp[2])
            vyy = v_Y - state_temp[4] * np.cos(state_temp[2])

            # 1.2 calculate the v in body frame according to the v_world
            v_u = np.sqrt(vxx * vxx + vyy * vyy) * np.sign(vxx) * np.sign(np.cos(state_temp[2]))
            v_u = np.clip(v_u, -1.3, 1.3)
            phi = np.arctan2(vyy, vxx)
            # print(guid_d, v_d, phi / D2R)

            # 1.3 calculate the final action
            action0 = agents[0].get_speed_yaw_controller_pid(state_temp, v_u, phi)

            # 2 the agent[1] is the follower
            # 2.1 guidance for the agent1
            tgt_pos1 = np.array([state_temp[0] + formation_ship1[0], state_temp[1] + formation_ship1[1]])
            v_u1, phi1 = self.follower_guidance(tgt_pos1, v0, self.guidLine.theta, state_temp1)

            # 2.2 calculate the final action fro the agent1
            action1 = agents[1].get_speed_yaw_controller_pid(state_temp1, v_u1, phi1)

            # 3 the agent[2] is the follower
            # 3.1 guidance for the agent1
            tgt_pos2 = np.array([state_temp[0] + formation_ship2[0], state_temp[1] + formation_ship2[1]])
            v_u2, phi2 = self.follower_guidance(tgt_pos2, v0, self.guidLine.theta, state_temp2)

            # 3.2 calculate the final action fro the agent1
            action2 = agents[2].get_speed_yaw_controller_pid(state_temp2, v_u2, phi2)

            # update the system
            agents[0].step(action0)
            agents[1].step(action1)
            agents[2].step(action2)
            # test
            action1[1] = phi
            # state_temp[4] = v_u
            # stateTemp[4] = phi_dot
            state_temp1[4] = v_u

            self.record.buffer_append((state_temp, action0, state_temp1, action1, state_temp2, action2))
            step_cnt = step_cnt + 1
        self.record.episode_append()

        data = self.record.get_episode_buffer()
        bs = data[0]
        ba = data[1]
        bs1 = data[2]
        ba1 = data[3]
        bs2 = data[4]
        ba2 = data[5]
        t = range(0, self.record.count)
        # mpl.style.use('seaborn')
        fig1 = plt.figure(1)
        plt.clf()
        plt.subplot(3, 1, 1)
        tss = range(500)
        plt.plot(bs[t, 0], bs[t, 1], label='leader1')
        plt.plot(bs1[t, 0], bs1[t, 1], label='follower1')
        plt.plot(bs2[t, 0], bs2[t, 1], label='follower2')
        plt.plot(np.arange(0, 17, 1), np.arange(0, 17, 1), '--r', label='line')
        # plt.plot(t, bs[t, 1], label='y')
        # plt.plot(t, bs[t, 2] / D2R, label='yaw')
        plt.ylabel('Eta $(\circ)$', fontsize=15)
        plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))

        # 把轨迹旋转以后，方便查看
        matrix = np.array([[np.cos(self.guidLine.theta), -np.sin(self.guidLine.theta)],
                           [np.sin(self.guidLine.theta), np.cos(self.guidLine.theta)]])
        pos_leader = bs[t, 0:2]
        pos_follower1 = bs1[t, 0:2]
        pos_follower2 = bs2[t, 0:2]
        pos_leader_t = np.transpose(matrix.dot(np.transpose(pos_leader)))
        pos_fol1_t = np.transpose(matrix.dot(np.transpose(pos_follower1)))
        pos_fol2_t = np.transpose(matrix.dot(np.transpose(pos_follower2)))

        plt.subplot(3, 1, 2)
        plt.plot(t, pos_leader_t[t, 0], label='leader')
        plt.plot(t, pos_fol1_t[t, 0], label='follower1')
        plt.plot(t, pos_fol2_t[t, 0], label='follower2')
        plt.ylabel('X (m)', fontsize=15)
        plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
        plt.subplot(3, 1, 3)
        plt.plot(t, pos_leader_t[t, 1], label='leader')
        plt.plot(t, pos_fol1_t[t, 1], label='follower1')
        plt.plot(t, pos_fol2_t[t, 1], label='follower2')
        plt.ylabel('Y (m)', fontsize=15)
        plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
        plt.show()

    def task_formation_switch(self):
        agents = self.ships
        self.record.clear()
        # control gain
        guid_kd = 0.3
        # guid_kd1 = 0.2
        step_cnt = 0

        # 队形控制
        formation_ship1_list = list()
        formation_ship1_list.append(np.array([-2., 2.]))
        formation_ship1_list.append(np.array([-2., 0.]))

        formation_ship2_list = list()
        formation_ship2_list.append(np.array([2., -2.]))
        formation_ship2_list.append(np.array([0., -2.]))

        for i in range(5000):
            state_temp = agents[0].observe()
            state_temp1 = agents[1].observe()
            state_temp2 = agents[2].observe()
            # action2, oil = ship1.get_controller_pid(stateTemp, ref)

            # 1 the agent[0] is the leader,
            # 1.1 calculate the v in world frame according to guider-line
            guid_d = self.guidLine.dis_line2point(state_temp[0], state_temp[1])
            v_d = -guid_d * guid_kd
            # print(guid_d)
            v0 = 1
            v_X = -v_d * np.cos(self.guidLine.theta) + v0 * np.sin(self.guidLine.theta)
            v_Y = v_d * np.sin(self.guidLine.theta) + v0 * np.cos(self.guidLine.theta)
            # considering the effect of v_v
            vxx = v_X + state_temp[4] * np.sin(state_temp[2])
            vyy = v_Y - state_temp[4] * np.cos(state_temp[2])

            # 1.2 calculate the v in body frame according to the v_world
            v_u = np.sqrt(vxx * vxx + vyy * vyy) * np.sign(vxx) * np.sign(np.cos(state_temp[2]))
            v_u = np.clip(v_u, -1.3, 1.3)
            phi = np.arctan2(vyy, vxx)
            # print(guid_d, v_d, phi / D2R)

            # 1.3 calculate the final action
            action0 = agents[0].get_speed_yaw_controller_pid(state_temp, v_u, phi)

            # 1.4 control the formation for followers
            if step_cnt < 3000:
                formation_ship1 = formation_ship1_list[0]
                formation_ship2 = formation_ship2_list[0]
            else:
                formation_ship1 = formation_ship1_list[1]
                formation_ship2 = formation_ship2_list[1]

            # 2 the agent[1] is the follower
            # 2.1 guidance for the agent1
            tgt_pos1 = np.array([state_temp[0] + formation_ship1[0], state_temp[1] + formation_ship1[1]])
            v_u1, phi1 = self.follower_guidance(tgt_pos1, v0, self.guidLine.theta, state_temp1)

            # 2.2 calculate the final action fro the agent1
            action1 = agents[1].get_speed_yaw_controller_pid(state_temp1, v_u1, phi1)

            # 3 the agent[2] is the follower
            # 3.1 guidance for the agent1
            tgt_pos2 = np.array([state_temp[0] + formation_ship2[0], state_temp[1] + formation_ship2[1]])
            v_u2, phi2 = self.follower_guidance(tgt_pos2, v0, self.guidLine.theta, state_temp2)

            # 3.2 calculate the final action fro the agent1
            action2 = agents[2].get_speed_yaw_controller_pid(state_temp2, v_u2, phi2)

            # update the system
            agents[0].step(action0)
            agents[1].step(action1)
            agents[2].step(action2)
            # test
            action1[1] = phi
            # state_temp[4] = v_u
            # stateTemp[4] = phi_dot
            state_temp1[4] = v_u

            self.record.buffer_append((state_temp, action0, state_temp1, action1, state_temp2, action2))
            step_cnt = step_cnt + 1
        self.record.episode_append()

        data = self.record.get_episode_buffer()
        bs = data[0]
        ba = data[1]
        bs1 = data[2]
        ba1 = data[3]
        bs2 = data[4]
        ba2 = data[5]
        t = range(0, self.record.count)
        # mpl.style.use('seaborn')
        fig1 = plt.figure(1)
        plt.clf()
        plt.subplot(3, 1, 1)
        tss = range(500)
        plt.plot(bs[t, 0], bs[t, 1], label='leader1')
        plt.plot(bs1[t, 0], bs1[t, 1], label='follower1')
        plt.plot(bs2[t, 0], bs2[t, 1], label='follower2')
        plt.plot(np.arange(0, 17, 1), np.arange(0, 17, 1), '--r', label='line')
        # plt.plot(t, bs[t, 1], label='y')
        # plt.plot(t, bs[t, 2] / D2R, label='yaw')
        plt.ylabel('Eta $(\circ)$', fontsize=15)
        plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))

        # 把轨迹旋转以后，方便查看
        matrix = np.array([[np.cos(self.guidLine.theta), -np.sin(self.guidLine.theta)],
                           [np.sin(self.guidLine.theta), np.cos(self.guidLine.theta)]])
        pos_leader = bs[t, 0:2]
        pos_follower1 = bs1[t, 0:2]
        pos_follower2 = bs2[t, 0:2]
        pos_leader_t = np.transpose(matrix.dot(np.transpose(pos_leader)))
        pos_fol1_t = np.transpose(matrix.dot(np.transpose(pos_follower1)))
        pos_fol2_t = np.transpose(matrix.dot(np.transpose(pos_follower2)))

        plt.subplot(3, 1, 2)
        plt.plot(t, pos_leader_t[t, 0], label='leader')
        plt.plot(t, pos_fol1_t[t, 0], label='follower1')
        plt.plot(t, pos_fol2_t[t, 0], label='follower2')
        plt.ylabel('X (m)', fontsize=15)
        plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
        plt.subplot(3, 1, 3)
        plt.plot(t, pos_leader_t[t, 1], label='leader')
        plt.plot(t, pos_fol1_t[t, 1], label='follower1')
        plt.plot(t, pos_fol2_t[t, 1], label='follower2')
        plt.ylabel('Y (m)', fontsize=15)
        plt.legend(fontsize=15, bbox_to_anchor=(1, 1.05))
        plt.show()


if __name__ == '__main__':
    task = TaskCybershipCooperative()
    test_flag = 2
    if test_flag == 1:
        task.task_formation()

    if test_flag == 2:
        task.task_formation_switch()
