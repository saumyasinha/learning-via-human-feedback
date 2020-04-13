import pygame
import os
os.environ['SDL_VIDEO_WINDOW_POS'] = '1000,200'

white = (255, 255, 255)
green = (0, 255, 0)
blue = (0, 0, 128)
pygame.init()
screen = pygame.display.set_mode((640, 480))
screen.fill(white)
pygame.display.set_caption('Pygame Keyboard Test')
font = pygame.font.Font('freesansbold.ttf', 32)
pygame.mouse.set_visible(0)


while True:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_0:
                text = font.render("Positive reward", True, green, blue)
                textRect = text.get_rect()
                textRect.center = (320, 240)
                screen.blit(text, textRect)
            if event.key == pygame.K_1:
                text = font.render("Negative reward", True, green, blue)
                textRect = text.get_rect()
                textRect.center = (320, 240)
                screen.blit(text, textRect)
            else:
                pygame.display.update()
                pass



            # print("key pressed")
            # time.sleep(0.1)
        pygame.display.update()



## Using cv2 for continous human input (wasn't asynchornous enough)
# import cv2
#
# k = ''
# last_input = ''
# flag = False
#
# while k != 113:
#
#     cv2.namedWindow('chomu', cv2.WINDOW_AUTOSIZE)
#     k = cv2.waitKey(1)
#     if k!=-1:
#         print("Last input :", )
#         print(k)
#     print("hi")
