import os
import pygame


class Interface:
    """ Pygame interface for training TAMER """
    def __init__(self, action_map):
        self.action_map = action_map
        pygame.init()
        self.font = pygame.font.Font("freesansbold.ttf", 32)

        # set position of pygame window (so it doesn't overlap with gym)
        os.environ["SDL_VIDEO_WINDOW_POS"] = "1000,100"

        self.screen = pygame.display.set_mode((200, 100))
        self.screen.fill((0, 0, 0))
        pygame.display.flip()

    def get_scalar_feedback(self):
        """
        Get human input. 'W' key for positive, 'A' key for negative.
        Returns: scalar reward (1 for positive, -1 for negative)
        """
        reward = 0
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    self.screen.fill((0, 255, 0))
                    reward = 1
                    break
                elif event.key == pygame.K_a:
                    self.screen.fill((255, 0, 0))
                    reward = -1
                    break
        pygame.display.flip()
        # print(reward)
        return reward

    def show_action(self, action):
        """
        Show agent's action on pygame screen
        Args:
            action: numerical action (for MountainCar environment only currently)
        """
        self.screen.fill((0, 0, 0))
        pygame.display.flip()
        text = self.font.render(self.action_map[action], True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.center = (100, 50)
        self.screen.blit(text, text_rect)
        pygame.display.flip()
