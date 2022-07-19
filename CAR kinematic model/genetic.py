import random
import math
import numpy as np


class GeneticAlg:

    def __init__(self, start_x, start_y, ox, oy, resolution, rr):
        self.motion = [[1, 0],
                       [0, 1],
                       [-1, 0],
                       [0, -1],
                       [-1, -1],
                       [-1, 1],
                       [1, -1],
                       [1, 1]]
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
        self.create_init_population()

    def create_init_population(self):
        # # TODO - how many routes? how many steps?
        # for route in range(50):
        #     deltas = [random.randrange(-15, 15, 1) for i in range(900)]
        #     dist = [1 for i in range(900)]
        #     final_path = [np.array([self.start_x, self.start_y])]
        #     for i in range(len(deltas)):
        #         xx = final_path[-1][0] + (dist[i] * math.cos(np.deg2rad(deltas[i])))
        #         yy = final_path[-1][1] + (dist[i] * math.sin(np.deg2rad(deltas[i])))
        #         if not self.verify_step(xx, yy):
        #             break
        #         final_path.append(np.array([xx, yy]))
        #     self.population.append(np.array(final_path))

        for route in range(50):
            directions = [self.motion[random.randrange(0, 7, 1)] for _ in range(900)]
            final_path = [np.array([self.calc_xy_index(self.start_x, self.min_x),
                                    self.calc_xy_index(self.start_y, self.min_y)])]
            for direction in directions:
                xx = final_path[-1][0] + direction[0]
                yy = final_path[-1][1] + direction[1]
                if not self.verify_step(xx, yy):
                    break
                final_path.append(np.array([xx, yy]))
            self.population.append(np.array(final_path))

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
                    if d < self.rr:
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
