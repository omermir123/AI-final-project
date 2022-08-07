import math
import random
from genetic import GeneticAlg

import cv2
import numpy as np
from time import sleep
import argparse

from environment import Environment, Parking1
from helper import PathPlanning, ParkPathPlanning, interpolate_path
from control import Car_Dynamics, PurePersuit, Linear_MPC_Controller
from utils import angle_of_line, make_square, DataLogger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_start', type=int, default=70, help='X of start')
    parser.add_argument('--y_start', type=int, default=50, help='Y of start')
    parser.add_argument('--psi_start', type=int, default=0, help='psi of start')
    parser.add_argument('--x_end', type=int, default=90, help='X of end')
    parser.add_argument('--y_end', type=int, default=80, help='Y of end')
    parser.add_argument('--parking', type=int, default=8, help='park position in parking1 out of 24')

    args = parser.parse_args()
    logger = DataLogger()

    ########################## default variables ################################################
    start = np.array([args.x_start, args.y_start])
    end   = np.array([args.x_end, args.y_end])
    #############################################################################################

    # environment margin  : 5
    # pathplanning margin : 5

    ########################## defining obstacles ###############################################
    parking1 = Parking1(args.parking)
    end, obs = parking1.generate_obstacles()

    # add squares
    # square1 = make_square(10,65,20)
    # square2 = make_square(15,30,20)
    # square3 = make_square(50,50,10)
    # obs = np.vstack([obs,square1,square2,square3])

    # Rahneshan logo
    # start = np.array([50,5])
    # end = np.array([35,67])
    # rah = np.flip(cv2.imread('READ_ME/rahneshan_obstacle.png',0), axis=0)
    # obs = np.vstack([np.where(rah<100)[1],np.where(rah<100)[0]]).T

    # new_obs = np.array([[78,78],[79,79],[78,79]])
    # obs = np.vstack([obs,new_obs])
    #############################################################################################

    ########################### initialization ##################################################
    env = Environment(obs)
    my_car = Car_Dynamics(start[0], start[1], 0, np.deg2rad(args.psi_start), length=4, dt=0.2)
    MPC_HORIZON = 5
    controller = PurePersuit()
    # controller = Linear_MPC_Controller()

    res = env.render(my_car.x, my_car.y, my_car.psi, 0)
    cv2.imshow('environment', res)
    key = cv2.waitKey(1)
    #############################################################################################

    ############################# path planning #################################################
    # park_path_planner = ParkPathPlanning(obs)
    # path_planner = PathPlanning(obs)
    #
    # print('planning park scenario ...')
    # new_end, park_path, ensure_path1, ensure_path2 = park_path_planner.generate_park_scenario(int(start[0]),int(start[1]),int(end[0]),int(end[1]))
    #
    # print('routing to destination ...')
    # path = path_planner.plan_path(int(start[0]),int(start[1]),int(new_end[0]),int(new_end[1]))
    # path = np.vstack([path, ensure_path1])
    #
    # print('interpolating ...')
    # interpolated_path = interpolate_path(path, sample_rate=5)
    # interpolated_park_path = interpolate_path(park_path, sample_rate=2)
    # interpolated_park_path = np.vstack([ensure_path1[::-1], interpolated_park_path, ensure_path2[::-1]])
    #
    # env.draw_path(interpolated_path)
    # env.draw_path(interpolated_park_path)

    # final_path = np.vstack([interpolated_path, interpolated_park_path, ensure_path2])

    margin = 5
    # sacale obstacles from env margin to pathplanning margin

    obstacles = obs + np.array([margin, margin])
    # obstacles = obs[(obs[:, 0] >= 0) & (obs[:, 1] >= 0)]

    obs1 = np.concatenate([np.array([[0, i] for i in range(105)]),
                            np.array([[105, i] for i in range(105)]),
                               np.array([[i, 0] for i in range(105)]),
                               np.array([[i, 105] for i in range(105)]),
                               obstacles])

    ox = [int(item) for item in obs1[:, 0]]
    oy = [int(item) for item in obs1[:, 1]]
    grid_size = 1
    robot_radius = 4

    gen = GeneticAlg(start[0], start[1], ox, oy, grid_size, robot_radius, end[0], end[1])
    pop1 = gen.population
    for i in range(400):
        print(f"Generation number {i}")
        gen.run_genetics(i)
        my_car = Car_Dynamics(start[0], start[1], 0, np.deg2rad(args.psi_start), length=4, dt=0.2)
    pop2 = gen.population
    # np.array(gen.population).tofile(r"C:\Users\Admin\Desktop\targilim\semester B Year 2\AI 67842\AI-final-project")
    #############################################################################################

    ################################## control ##################################################
    my_car = Car_Dynamics(start[0], start[1], 0, np.deg2rad(args.psi_start), length=4, dt=0.2)
    price_of_paths = []
    for path in gen.population:
        price_of_paths.append(gen.fitness(path, my_car, 100))
        my_car = Car_Dynamics(start[0], start[1], 0, np.deg2rad(args.psi_start), length=4, dt=0.2)
    good_indexes = np.argsort(np.array(price_of_paths))
    final_paths = gen.population[good_indexes[:30]]
    for k, final_path1 in enumerate(final_paths):
        print('driving to destination ...')
        print(f"the score is {price_of_paths[good_indexes[k]]}")
        # final_path = max(pop, key=lambda x:len(x))
        my_car = Car_Dynamics(start[0], start[1], 0, np.deg2rad(args.psi_start), length=4, dt=0.2)
        acc, delta, paths_x, paths_y, paths_psi = controller.calc_path(final_path1, np.deg2rad(args.psi_start), 4)
        for i in range(len(acc)):

        # for i,point in enumerate(final_path1):
            # acc, delta = controller.optimize(my_car, final_path1[i:i+MPC_HORIZON])
            # acc, delta = accelerates[i], deltas[i]
            # my_car.update_state(my_car.move(acc,  delta))

            res = env.render(paths_x[i], paths_y[i], paths_psi[i], delta[i])
            # logger.log(point, my_car, acc[i], delta[i])
            cv2.imshow('environment', res)
            key = cv2.waitKey(1)
            if key == ord('s'):
                cv2.imwrite('res.png', res*255)
        # acc, delta = controller.optimize(my_car, final_path1[-5:])
        degree = my_car.psi
        print(f"the angle is {np.rad2deg(paths_psi[-1]) % 360}")
        print(f"the angle in rad is {paths_psi[-1]}")
        print(f"render : the final point is ({paths_x[-1]}, {paths_y[-1]})")
        print(f"real : the final point is ({final_path1[-1]}, {final_path1[-1]})")
        # zeroing car steer
        cv2.waitKey(100000)
        res = env.render(my_car.x, my_car.y, my_car.psi, 0)
        # logger.save_data()
        cv2.imshow('environment', res)
        key = cv2.waitKey()
        #####
        # env = Environment(obs)
        my_car = Car_Dynamics(start[0], start[1], 0, np.deg2rad(args.psi_start), length=4, dt=0.2)
        # MPC_HORIZON = 5
        # controller = MPC_Controller()
        # controller = Linear_MPC_Controller()

        res = env.render(my_car.x, my_car.y, my_car.psi, 0)
        cv2.imshow('environment', res)
        # key = cv2.waitKey(1)
        #####

    #############################################################################################

    cv2.destroyAllWindows()

