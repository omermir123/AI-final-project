import random
import math
import numpy as np
import control


class GeneticAlg:

    def __init__(self, start_x, start_y, ox, oy, resolution, rr, goal_x, goal_y):
        self.motion = [[1, 0],
                       [0, 1],
                       [-1, 0],
                       [0, -1],
                       [-1, -1],
                       [-1, 1],
                       [1, -1],
                       [1, 1]]
        self.directions_ = []
        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.calc_obstacle_map(ox, oy)
        self.population = []
        self.start_x = start_x
        self.start_y = start_y
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.create_init_population()

    def direction_to_path(self, directions):
        final_path = [np.array([self.calc_xy_index(self.start_x, self.min_x),
                                self.calc_xy_index(self.start_y, self.min_y)])]
        for direction in directions:
            xx = final_path[-1][0] + direction[0]
            yy = final_path[-1][1] + direction[1]
            if not self.verify_step(xx, yy):
                break
            final_path.append(np.array([xx, yy]))
        return np.array(final_path)

    def create_init_population(self):
        for route in range(40):
            directions = [self.motion[random.randrange(0, 7, 1)] for _ in range(900)]
            self.population.append(self.direction_to_path(directions))
            self.directions_.append(directions[:len((self.population[-1]))])

        self.population = np.array(self.population)
        self.directions_ = np.array(self.directions_)

    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d < self.rr - 2:
                        self.obstacle_map[ix][iy] = True
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
        if self.obstacle_map[x][y]:
            return False

        return True

    def flatten_path(self, path):
        new_path = np.empty(len(path), dtype=object)
        new_path[:] = [tuple(point) for point in path]
        return new_path

    def get_split_indexes(self, path1, path2):
        left, right = self.flatten_path(path1), self.flatten_path(path2)
        same_points = np.array([x for x in np.intersect1d(left, right, return_indices=True)])[1:]
        split_index = random.randint(0, len(same_points[0]) - 1)
        while same_points[0][split_index] == 0 and same_points[1][split_index] == 0 and len(same_points[0]) > 1:
            split_index = random.randint(0, len(same_points[0]) - 1)
        return same_points[0][split_index], same_points[1][split_index]

    def cross_over(self, paths):
        new_gen = []
        for i in range(0, len(paths), 2):
            split1, split2 = self.get_split_indexes(paths[i], paths[i + 1])
            new_gen.append(np.concatenate([paths[i][:split1], paths[i + 1][split2:]]))
            new_gen.append(np.concatenate([paths[i + 1][:split2], paths[i][split1:]]))
        for i in range(len(paths) // 2):
            split1, split2 = self.get_split_indexes(paths[i], paths[len(paths) - i - 1])
            new_gen.append(np.concatenate([paths[i][:split1], paths[len(paths) - i - 1][split2:]]))
            new_gen.append(np.concatenate([paths[len(paths) - i - 1][:split2], paths[i][split1:]]))
        return new_gen

    def compute_mutation(self, paths):
        p = 0.3
        for j, path in enumerate(paths):
            mutate_path = [path[0]]
            path_deg = []
            for i in range(len(path) - 1):
                angle = np.arctan2(path[i + 1][1] - path[i][1], path[i + 1][0] - path[i][0])
                dist = np.linalg.norm(path[i] - path[i + 1])
                path_deg.append([angle, dist])
            if random.uniform(0, 1) > p:
                rand_index = random.randint(0, len(path_deg) - 1)
                # rand_deg = np.deg2rad(random.randint(0, 360))
                rand_deg = np.deg2rad(random.choice([0, 90, 270, 360]))
                path_deg[rand_index][0] = rand_deg
            for i in range(len(path_deg)):
                new_x = mutate_path[-1][0] + int(path_deg[i][1] * np.cos(path_deg[i][0]))
                new_y = mutate_path[-1][1] + int(path_deg[i][1] * np.sin(path_deg[i][0]))
                if not self.verify_step(new_x, new_y):
                    break
                mutate_path.append(np.array([new_x, new_y]))
            paths[j] = np.array(mutate_path)
        return paths

    def fitness(self, path, car, gen_num):
        og_path = path
        path = self.flatten_path(path)
        self.goal_x, self.goal_y = 75, 32
        points_in_parking = [(self.goal_x, self.goal_y + i) for i in range(-15, 16)]
        num_of_appearances = 0
        for p in points_in_parking:
            num_of_appearances += np.count_nonzero(path == p)
        car_location = np.array(path[-1])
        norm = np.linalg.norm(car_location - np.array([self.goal_x, self.goal_y]))
        path_counter = np.unique(self.flatten_path(path), return_counts=True)[1]
        n_counter = np.count_nonzero(path_counter > 2)
        norm_w, n_counter_w, num_of_appearances_w = 8, 9, 8
        if 100 <= gen_num <=105:
            car_psi = self.degree_of_parking(og_path, car)
            # _, degree = controller.optimize(car, og_path[-5:])
            # degree = degree % 360
            delta_degree = min(abs(car_psi - 90), abs(car_psi - 270))
        else :
            delta_degree = 0

        return 10*norm + n_counter + 2*(20 - num_of_appearances) + delta_degree * 5

    def move_car_in_path(self, path, car):
        controller = control.MPC_Controller()
        for i,point in enumerate(path):
            acc, delta = controller.optimize(car, path[i:i+5])
            # acc, delta = accelerates[i], deltas[i]
            car.update_state(car.move(acc,  delta))

    def degree_of_parking(self, path, car):
        self.move_car_in_path(path,car)
        return car.psi % 360

    def run_genetics(self, gen_num):
        price_of_paths = []
        for path in self.population:
            my_car = control.Car_Dynamics(self.start_x, self.start_y, 0, np.deg2rad(270), length=4, dt=0.2)
            price_of_paths.append(self.fitness(path, my_car, gen_num))
        good_indexes = np.argsort(np.array(price_of_paths))
        children_pop = self.cross_over(self.population[good_indexes[:20]])
        self.population = np.array(self.compute_mutation(children_pop))
