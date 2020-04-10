# Python imports
import random

# Library imports
import pygame
from pygame.key import *
from pygame.locals import *
from pygame.color import *

# pymunk imports
import pymunk
import pymunk.pygame_util

import cv2
import numpy as np


class BouncyBalls(object):
    def __init__(self):
        # Space
        self._space = pymunk.Space()
        self._space.gravity = (0.0, -900.0)

        # Physics
        # Time step
        self._dt = 1.0 / 60.0
        # Number of physics steps per screen frame
        self._physics_steps_per_frame = 5

        # pygame
        pygame.init()
        self._screen = self._display_surf = pygame.display.set_mode((600, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self._clock = pygame.time.Clock()

        self._draw_options = pymunk.pygame_util.DrawOptions(self._screen)

        # Static barrier walls (lines) that the balls bounce off of
        self._add_static_scenery()

        # Balls that exist in the world
        self._balls = []

        # Execution control and time until the next ball spawns
        self._ticks_to_next_ball = 2

    def get_frames(self, length):
        frames = []

        for i in range(length):
            # Progress time forward
            for x in range(self._physics_steps_per_frame):
                self._space.step(self._dt)
                self._update_balls()
            
            self._clear_screen()
            self._draw_objects()
            pygame.display.flip()
            # Delay fixed time between frames
            self._clock.tick(3000)
            pygame.display.set_caption("fps: " + str(self._clock.get_fps()))

            frame = pygame.surfarray.pixels3d(self._screen)
            frames.append(frame.copy())
        return frames



    def _add_static_scenery(self):
        static_body = self._space.static_body
        static_lines = [pymunk.Segment(static_body, (0, 200), (407.0, 246.0), 10)]
        for line in static_lines:
            line.elasticity = 0.95
            line.friction = 0.9
        self._space.add(static_lines)


    def _update_balls(self):
        self._ticks_to_next_ball -= 1
        if self._ticks_to_next_ball <= 0:
            self._create_ball()
            self._ticks_to_next_ball = 100
        # Remove balls that fall below 100 vertically
        balls_to_remove = [ball for ball in self._balls if ball.body.position.y < 100]
        for ball in balls_to_remove:
            self._space.remove(ball, ball.body)
            self._balls.remove(ball)

    def _create_ball(self):
        mass = 10
        radius = 25
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        body = pymunk.Body(mass, inertia)
        x = random.randint(115, 350)
        body.position = x, 400
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.elasticity = 0.95
        shape.friction = 0.9
        self._space.add(body, shape)
        self._balls.append(shape)

    def _clear_screen(self):
        self._screen.fill(THECOLORS["white"])

    def _draw_objects(self):
        self._space.debug_draw(self._draw_options)
