import pygame
from abc import ABC, abstractmethod
import math
import random
import numpy as np
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


class Game:
    def __init__(self,network,score, width=400, height=400):
        self.width, self.height = width, height
        self.isRunning = True
        self.screens = []

        pygame.init()
        self.score = 0
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.add_to_screens(GameScreen(self,network))
        self.main_loop()
        score.append(self.score)

    def main_loop(self):
        while self.isRunning:
            delta = pygame.time.Clock().tick(60) / 1000
            for event in pygame.event.get():
                self.get_top_screen().input(event)

            self.screen.fill(BLACK)
            self.get_top_screen().update(delta)
            self.get_top_screen().draw(delta, pygame.display.get_surface())
            pygame.display.flip()
        pygame.quit()

    def add_to_screens(self, screen):
        if not isinstance(screen, Screen):
            raise Exception("Only screens can be added to screen stack")
        self.screens.append(screen)

    def get_top_screen(self):
        if len(self.screens) < 1:
            raise Exception("Non screens currently present, game should probably close")
        return self.screens[-1]

    def pop_screens(self):
        if len(self.screens) < 1:
            return
        return self.screens.pop()


class Screen(ABC):
    game = None

    @abstractmethod
    def input(self, event):
        if event.type == pygame.QUIT:
            self.game.isRunning = False

    @abstractmethod
    def update(self, delta):
        pass

    @abstractmethod
    def draw(self, delta, surface):
        pass


class GameScreen(Screen):
    def __init__(self, game,network):
        self.game = game
        self.snake = Snake(20,20)
        self.network = network
        self.counter = 0

    def draw(self, delta, surface):
        if self.snake is None:
            return
        scaleX = self.game.width / self.snake.get_size_x()
        scaleY = self.game.height / self.snake.get_size_y()
        pygame.draw.circle(surface, WHITE,
                           (int((self.snake.grow_pos[0] + 0.5) * scaleX), int((self.snake.grow_pos[1] + 0.5) * scaleY)),
                           int(scaleX / 2))
        for position in self.snake.get_snake_tiles():
            pygame.draw.rect(surface, WHITE, pygame.Rect(position[0] * scaleX, position[1] * scaleY, scaleX, scaleY))

    def update(self, delta):
        self.counter += delta
        if self.counter > 10:
            self.snake.alive = False
        if self.snake.update(delta):
            input_data = np.reshape(self.snake.get_map(), (self.snake.get_size_x() * self.snake.get_size_y(),))
            output = self.network.out(input_data)
            i = np.argmax(output)
            i -= 1
            target_orientation = self.snake.get_orientation() + 90 * i
            if target_orientation != self.snake.get_orientation():
                if target_orientation == 0:
                    self.snake.down()
                if target_orientation == 90:
                    self.snake.right()
                if target_orientation == 180:
                    self.snake.up()
                if target_orientation == 270:
                    self.snake.left()

        if not self.snake.alive:
            self.game.isRunning = False
            self.game.score = self.snake.score

    def input(self, event):
        Screen.input(self, event) # Check for any MAJOR user input






class Snake:
    def __init__(self, sizeX, sizeY):
        self.alive = True
        self.grow_pos = []
        self.__timer = 0
        self.__length = 2
        self.__sizeX, self.__sizeY = sizeX, sizeY
        self.__map = []
        self.__speed = 0.01
        self.__orientation = 0
        self.score = 0
        for x in range(0, sizeX):
            self.__map.append([])
            for y in range(0, sizeY):
                self.__map[x].append(0)

        self.__headX, self.__headY = 10,10
        self.__map[self.__headX][self.__headY] = 2
        self.__map[self.__headX][self.__headY - 1] = 1
        self.__snakeTiles = [[self.__headX, self.__headY], [self.__headX, self.__headY - 1]]
        self.spawn_growth()

    def spawn_growth(self):
        x, y = 0, 0
        while True:
            x, y = random.randint(0, self.__sizeX - 1), random.randint(0, self.__sizeY - 1)
            if self.__map[x][y] == 0:
                break

        self.grow_pos = [x, y]
        self.__map[x][y] = - 10
    def get_map(self):
        return self.__map[:]

    def up(self):
        if self.__orientation == 0: return
        self.__orientation = 180

    def down(self):
        if self.__orientation == 180: return
        self.__orientation = 0

    def left(self):
        if self.__orientation == 90: return
        self.__orientation = 270

    def right(self):
        if self.__orientation == 270: return
        self.__orientation = 90

    def get_orientation(self):
        return self.__orientation

    def get_snake_tiles(self):
        return self.__snakeTiles

    def update(self, delta):
        if self.alive:
            self.__timer += delta
            if self.__timer > self.__speed:
                self.__headX += int(math.sin(math.radians(self.__orientation)))
                self.__headY += int(math.cos(math.radians(self.__orientation)))

                self.__timer = 0
                self.score += 0.01

                if (self.__headX % self.__sizeX) != self.__headX or (self.__headY % self.__sizeY) != self.__headY or \
                                self.__map[self.__headX][self.__headY] > 0:
                    self.alive = False
                    return

                grow = (self.grow_pos[0] == self.__headX and self.grow_pos[1] == self.__headY)

                self.__map[self.__headX][self.__headY] = self.__length + 1
                self.__snakeTiles = []
                for x in range(0, self.__sizeX):
                    for y in range(0, self.__sizeY):
                        if self.__map[x][y] > 0:
                            if self.__map[x][y] > (1 if not grow else 0):
                                self.__snakeTiles.append([x, y])
                            self.__map[x][y] -= 1 if not grow else 0

                if grow:
                    self.__length += 1
                    self.score += 1
                    self.spawn_growth()
            return True
        return False

    def get_size_x(self):
        return self.__sizeX

    def get_size_y(self):
        return self.__sizeY


