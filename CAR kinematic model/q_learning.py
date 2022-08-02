import math
import random

import numpy as np
import cv2
from control import Car_Dynamics, MPC_Controller, Linear_MPC_Controller



class QLearingAgent:

    def __init__(self, sx, sy, gx, gy, ox, oy, resolution, rr, alpha, gamma, epsilon):
        """
        :param sx: start point x
        :param sy: start point y
        :param gx: goal point x
        :param gy: goal point y
        :param ox: obstacles x coordinates
        :param oy: obstacles y coordinates
        :param resolution: idk
        :param rr: idk
        :param alpha: learning rate
        :param gamma: discount factor
        :param epsilon: exploration factor
        """
        self.start_x = sx
        self.start_y = sy
        self.goal_x = gx
        self.goal_y = gy
        # full motions
        self.motions = [[1, 0],
                        [0, 1],
                        [-1, 0],
                        [0, -1],
                        [-1, -1],
                        [-1, 1],
                        [1, -1],
                        [1, 1]]
        # limit motions to U,R,D,L
        # self.motions = [[1, 0],
        #                 [0, 1],
        #                 [-1, 0],
        #                 [0, -1]]
        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obs_map = None
        self.x_width, self.y_width = 0, 0
        self.calc_obstacle_map(ox, oy)

        # 3D array where (x,y,j) is the reward the agent gets when the current state is x,y and the chosen action is motion[j]
        self.rewards = np.zeros((self.x_width, self.y_width, len(self.motions)))
        for x in range(self.x_width):
            for y in range(self.y_width):
                for j in range(len(self.motions)):
                    next_state_x = x + self.motions[j][0]
                    next_state_y = y + self.motions[j][1]
                    if not self.verify_step(next_state_x, next_state_y):
                        continue
                    elif next_state_x == self.goal_x and next_state_y == self.goal_y:
                        self.rewards[x][y][j] = 10
                    elif self.obs_rewards(next_state_x, next_state_y):
                        self.rewards[x][y][j] = -10

                    else:
                        self.rewards[x][y][j] = 2
        for i in range(5):
            self.rewards[self.goal_x][self.goal_y - i][1] = 10
            self.rewards[self.goal_x][self.goal_y + i][3] = 10


        # 3D array where (x,y,j) is a flag: when the agent is in x,y state motions[j] is a legal action if the flag is 1.
        self.legal_moves = np.ones((self.x_width, self.y_width, len(self.motions)))
        for x in range(self.x_width):
            for y in range(self.y_width):
                for j in range(len(self.motions)):
                    next_state_x = x + self.motions[j][0]
                    next_state_y = y + self.motions[j][1]
                    if not self.verify_step(next_state_x, next_state_y):
                        continue
                    if self.obs_map[next_state_x][next_state_y]:
                        self.legal_moves[x][y][j] = 0

        # 3D array where (x,y,j) is the q_value for state = x,y and action = motions[j]
        self.q_table = np.zeros((self.x_width, self.y_width, len(self.motions)))

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def obs_rewards(self, x, y):
        return self.obs_map[x][y] or self.obs_map[x + 1][y] or self.obs_map[x][y + 1] \
               or self.obs_map[x - 1 ][y] or self.obs_map[x][y - 1] or self.obs_map[x + 1][y - 1] \
               or self.obs_map[x - 1][y + 1] or self.obs_map[x + 1][y + 1] or self.obs_map[x - 1][y - 1]


    def calc_obstacle_map(self, ox, oy):
        """
        helper func to calc obs_map
        :param ox:
        :param oy:
        :return:
        """

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)

        # obstacle map generation
        self.obs_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d < self.rr - 2:
                        self.obs_map[ix][iy] = True
                        break

    def calc_grid_position(self, index, min_position):
        """
        calc grid position
        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_step(self, x, y):
        px = self.calc_grid_position(x, self.min_x)
        py = self.calc_grid_position(y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obs_map[x][y]:
            return False

        return True

    def train(self, episodes):
        """
        function to train and adjust the q_table according to the algorithm we learned
        :param episodes: num of episodes to train the agent
        :return:
        """
        for i in range(episodes):
            state = [self.start_x, self.start_y]
            done = False

            while not done:
                legal_actions = np.argwhere(self.legal_moves[state[0]][state[1]] == 1).flatten()
                num_actions = len(legal_actions)
                action_probabilities = np.ones(num_actions, dtype=float) * self.epsilon / num_actions

                max_val = np.amax(self.q_table[state[0]][state[1]])
                best_motion_idx = np.random.choice(np.argwhere(self.q_table[state[0]][state[1]] == max_val).flatten())
                action_probabilities[np.argwhere(best_motion_idx == legal_actions)] += (1.0 - self.epsilon)

                next_action_idx = np.random.choice(legal_actions, p=action_probabilities)
                motion = self.motions[next_action_idx]

                next_state = (state[0] + motion[0], state[1] + motion[1])
                reward = self.rewards[state[0]][state[1]][next_action_idx]

                done = (next_state[0] == self.goal_x and next_state[1] == self.goal_y) or not self.verify_step(next_state[0], next_state[1])

                old_value = self.q_table[state[0]][state[1]][next_action_idx]
                next_max = np.max(self.q_table[next_state[0]][next_state[1]])

                # new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
                # self.q_table[state[0]][state[1]][motion_idx] = new_value
                self.q_table[state[0]][state[1]][next_action_idx] += self.alpha * (reward + self.gamma * next_max -
                                                                                   self.q_table[state[0]][state[1]][next_action_idx])

                state = next_state

    def train_and_print(self, episodes, start_x, start_y, psi_start, env, controller, logger):
        """
        function to train and adjust the q_table according to the algorithm we learned
        :param episodes: num of episodes to train the agent
        :return:
        """
        for j in range(episodes):
            state = [self.start_x, self.start_y]
            final_path = [state]
            done = False

            while not done:
                legal_actions = np.argwhere(self.legal_moves[state[0]][state[1]] == 1).flatten()
                num_actions = len(legal_actions)
                action_probabilities = np.ones(num_actions, dtype=float) * self.epsilon / num_actions

                max_val = np.amax(self.q_table[state[0]][state[1]])
                best_motion_idx = np.random.choice(np.argwhere(self.q_table[state[0]][state[1]] == max_val).flatten())
                action_probabilities[np.argwhere(best_motion_idx == legal_actions)] += (1.0 - self.epsilon)

                next_action_idx = np.random.choice(legal_actions, p=action_probabilities)
                motion = self.motions[next_action_idx]

                next_state = (state[0] + motion[0], state[1] + motion[1])
                reward = self.rewards[state[0]][state[1]][next_action_idx]

                done = (next_state[0] == self.goal_x and next_state[1] == self.goal_y) or not self.verify_step(next_state[0], next_state[1])

                old_value = self.q_table[state[0]][state[1]][next_action_idx]
                next_max = np.max(self.q_table[next_state[0]][next_state[1]])

                # new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
                # self.q_table[state[0]][state[1]][motion_idx] = new_value
                self.q_table[state[0]][state[1]][next_action_idx] += self.alpha * (reward + self.gamma * next_max -
                                                                                   self.q_table[state[0]][state[1]][next_action_idx])

                state = next_state
                final_path.append(state)

            final_path = np.array(final_path)
            my_car = Car_Dynamics(start_x, start_y, 0, np.deg2rad(psi_start), length=4, dt=0.2)
            print(f"path number {j}\n")
            for i, point in enumerate(final_path):
                acc, delta = controller.optimize(my_car, final_path[i:i + 5])
                # acc, delta = accelerates[i], deltas[i]
                my_car.update_state(my_car.move(acc, delta))
                if i == 0:
                    cv2.waitKey(1000)
                if i % 2 == 0:
                    res = env.render(my_car.x, my_car.y, my_car.psi, delta)
                    logger.log(point, my_car, acc, delta)
                    cv2.imshow('environment', res)
                    key = cv2.waitKey(1)
                    if key == ord('s'):
                        cv2.imwrite('res.png', res * 255)


    def test(self, episodes):
        """
        func to test q_values
        :param episodes: how many paths to generate
        :return:
        """
        for _ in range(episodes):
            state = [self.start_x, self.start_y]
            done = False

            while not done:
                print("current location " + str(state) + "\n")
                motion_idx = np.argmax(self.q_table[state[0]][state[1]])
                print("motion chosen " + str(self.motions[motion_idx]) + "\n")
                state = [state[0] + self.motions[motion_idx][0], state[1] + self.motions[motion_idx][1]]
                done = (state[0] == self.goal_x and state[1] == self.goal_y) or (state in self.obs_map) or \
                       (state[0] >= 99 or state[0] <= 0) or (state[1] >= 99 or state[1] <= 0)

    def get_path(self):
        """
        for our purpose - generates a final path to present on GUI according to training
        :return:
        """
        state = [self.start_x, self.start_y]
        final_path = [state]
        done = False

        # retrieve q_table from file
        # q_table_2d = np.loadtxt("large_train_100000.txt")
        # self.q_table = q_table_2d.reshape(q_table_2d.shape[0], q_table_2d.shape[1] // self.q_table.shape[2], self.q_table.shape[2])

        while not done:

            legal_actions = np.argwhere(self.legal_moves[state[0]][state[1]] == 1).flatten()
            num_actions = len(legal_actions)
            action_probabilities = np.ones(num_actions, dtype=float) * self.epsilon / num_actions

            max_val = np.amax(self.q_table[state[0]][state[1]])
            best_motion_idx = np.random.choice(np.argwhere(self.q_table[state[0]][state[1]] == max_val).flatten())
            action_probabilities[np.argwhere(best_motion_idx == legal_actions)] += (1.0 - self.epsilon)

            next_action_idx = np.random.choice(legal_actions, p=action_probabilities)

            state = [state[0] + self.motions[next_action_idx][0], state[1] + self.motions[next_action_idx][1]]
            final_path.append(state)
            done = (state[0] == self.goal_x and state[1] == self.goal_y) or not self.verify_step(state[0], state[1])

        return final_path


if __name__ == '__main__':
    agent = QLearingAgent(10,20, 60,80, [(11,12), (45,65), (90,100), 500,400], 0.1, 0.6, 0.1)
    agent.train(1000)
    agent.test(1)



