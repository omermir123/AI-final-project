import math
import random
from genetic import GeneticAlg

import cv2
import numpy as np
import time
import argparse

from q_learning import *
from environment import Environment, Parking1
from helper import PathPlanning, ParkPathPlanning, interpolate_path
from control import Car_Dynamics, MPC_Controller, PurePersuit, Linear_MPC_Controller
from utils import angle_of_line, make_square, DataLogger

########################## constant ################################################
Q_LEARNING = 0
GENETIC = 1
SIMULATE = 0
TRAINING = 1

# ready q_values tables
SHORT_Q_VALUES = 'short_train_1000.txt'
MEDIUM_Q_VALUES = 'medium_train_1000000_new.txt'
LARGE_Q_VALUES = 'large_train_100000.txt'

# short route configuration: --x_start 50 --y_start 40 --psi_start 270 --parking 13
# medium route configuration: --x_start 70 --y_start 95 --psi_start 0 --parking 7
# large route configuration: --x_start 50 --y_start 65 --psi_start 90 --parking 8

#############################################################################################


def quality(car, goal):
    norm = np.linalg.norm(np.array([car.x, car.y]) - np.array([goal[0], goal[1]]))
    angle = np.deg2rad(car.psi) % 360
    return 30 * norm + 5 * min(abs(angle - 90), abs(angle - 270))



if __name__ == '__main__':

    ################### define process configurations and parse arguments ###################
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_start', type=int, default=70, help='X of start')
    parser.add_argument('--y_start', type=int, default=50, help='Y of start')
    parser.add_argument('--psi_start', type=int, default=0, help='psi of start')
    parser.add_argument('--x_end', type=int, default=90, help='X of end')
    parser.add_argument('--y_end', type=int, default=80, help='Y of end')
    parser.add_argument('--parking', type=int, default=8, help='park position in parking1 out of 24')
    parser.add_argument('--state', type = int, default=SIMULATE, help= '0 = simulate car parking with values already '
                                                                       'learned, 1 = train the algorithm' )
    parser.add_argument('--alg', type=int, default=Q_LEARNING, help='algorithm to choose path by:'
                                                           '0 = q-learning, 1 = genetic algorithm')
    parser.add_argument('--training_episodes', type=int, default=0, help='for training state enter: for genetic - '
                                                                         'number of generations, for q-learning - '
                                                                         'number of episodes')
    parser.add_argument('--file_name', type=str, default=None, help='text file name:'
                                                                    'for Q-learning in training state: '
                                                                    'the file where the q_table will be saved'
                                                                    'for Q-learning in simulate state: '
                                                                    'the file where the simulation q_table is saved'
                                                                    'for genetic in training state: '
                                                                    'the file where the final population will be saved'
                                                                    'for genetic in simulate state: '
                                                                    'the file where the population is saved')

    args = parser.parse_args()
    logger = DataLogger()
    #############################################################################################

    ########################## default variables ################################################
    start = np.array([args.x_start, args.y_start])
    end = np.array([args.x_end, args.y_end])
    state = args.state
    alg = args.alg
    training_episodes = args.training_episodes
    file_name = args.file_name
    #############################################################################################

    ########################## defining obstacles ###############################################
    parking1 = Parking1(args.parking)
    end, obs = parking1.generate_obstacles()
    #############################################################################################

    ########################### environment initialization ##################################################
    env = Environment(obs)
    my_car = Car_Dynamics(start[0], start[1], 0, np.deg2rad(args.psi_start), length=4, dt=0.2)
    MPC_HORIZON = 5

    res = env.render(my_car.x, my_car.y, my_car.psi, 0)
    cv2.imshow('environment', res)
    key = cv2.waitKey(1)
    #############################################################################################

    ########################## handle obstacle_map ###############################################

    margin = 5
    obstacles = obs[(obs[:, 0] >= 0) & (obs[:, 1] >= 0)]

    obs1 = np.concatenate([np.array([[0, i] for i in range(105)]),
                            np.array([[105, i] for i in range(105)]),
                               np.array([[i, 0] for i in range(105)]),
                               np.array([[i, 105] for i in range(105)]),
                               obstacles])

    ox = [int(item) for item in obs1[:, 0]]
    oy = [int(item) for item in obs1[:, 1]]
    grid_size = 1
    robot_radius = 4
    #############################################################################################

    ########################## init agent ###############################################
    agent = None
    if alg == Q_LEARNING:
        agent = QLearingAgent(start[0], start[1], end[0], end[1], ox, oy, grid_size, robot_radius, 1, 0.6, 0.1)
        # agent = QLearingAgent(start[0], start[1], end[0], end[1], ox, oy, grid_size, robot_radius, 1, 0.6, 0.3)
        controller = MPC_Controller()
    else:
        agent = GeneticAlg(start[0], start[1], ox, oy, grid_size, robot_radius, end[0], end[1])
        controller = PurePersuit()
    #############################################################################################

    ########################## training or loading data ###############################################
    if state == TRAINING:
        if alg == GENETIC:
            print(f'starting to train the {alg} agent')
            start_time = time.time()
            agent.run_genetics(training_episodes)
            print(f"The time it took to train the {alg} agent with {training_episodes} generations is {time.time() - start_time} seconds")
            np.savetxt(file_name, agent.population)
        else:
            print(f'starting to train the {alg} agent')
            start_time = time.time()
            agent.train(training_episodes)
            # agent.train_and_print(training_episodes, start[0], start[1], args.psi_start, env, controller, logger)
            print(f"The time it took to train the {alg} agent with {training_episodes} episodes is {time.time() - start_time} seconds")
            q_table_2d = agent.q_table.reshape(agent.q_table.shape[0], -1)
            np.savetxt(file_name, q_table_2d)
    else:
        if alg == Q_LEARNING:
            q_table_2d = np.loadtxt(file_name)
            agent.q_table = q_table_2d.reshape(q_table_2d.shape[0], q_table_2d.shape[1] // agent.q_table.shape[2],
                                              agent.q_table.shape[2])
        else:
            population = np.loadtxt(file_name)
            agent.population = population
            gen_acc, deltas, paths_x, paths_y, paths_psi = controller.calc_path(np.array(agent.get_path()), np.deg2rad(args.psi_start), 4)

        # TODO: need to add option to load data for genetic algorithm to simulate parking
    #############################################################################################

    ########################## extracting path ###############################################
    #TODO: create get_path() function for genetic agent
    final_path = np.array(agent.get_path())

    #############################################################################################

    ################################## execute ##################################################
    print('driving to destination ...')
    for i,point in enumerate(final_path):
        if alg == Q_LEARNING:
            acc, delta = controller.optimize(my_car, final_path[i:i+MPC_HORIZON])
            # acc, delta = accelerates[i], deltas[i]
            my_car.update_state(my_car.move(acc,  delta))
            res = env.render(my_car.x, my_car.y, my_car.psi, delta)
        else:
            res = env.render(paths_x[i], paths_y[i], paths_psi[i], deltas [i])
            acc, delta = gen_acc[i], deltas[i]
            my_car.update_state_pure(my_car.move(gen_acc[i], deltas[i]))
        logger.log(point, my_car, acc, delta)
        cv2.imshow('environment', res)
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite('res.png', res*255)
    fitness = quality(my_car, end)
    print(f"quality percentage is {fitness}")

    # zeroing car steer
    res = env.render(my_car.x, my_car.y, my_car.psi, 0)
    logger.save_data()
    cv2.imshow('environment', res)
    key = cv2.waitKey()
    #############################################################################################

    cv2.destroyAllWindows()





