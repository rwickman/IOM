import pygame, sys
import random

pygame.init()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)


size = width, height = 512, 512
speed = [1, 1]
cirle_pos = 256, 256
cirle_pos_2 = 256, 128

screen = pygame.display.set_mode(size)
cur_i = 0
while 1:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()


    screen.fill(BLACK)
    pygame.draw.circle(screen, BLUE, cirle_pos, 20)
    cur_i += 1
    if cur_i % 10 == 0:
        rand_x = random.uniform(0, 512)
        rand_y = random.uniform(0, 512)
        demand_node_pos = (rand_x, rand_y) 
        pygame.draw.circle(screen, GREEN, demand_node_pos, 20)

        pygame.draw.line(screen, WHITE, cirle_pos, demand_node_pos, 5)
        cur_i = 0
    pygame.display.flip()
