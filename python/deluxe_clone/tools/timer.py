# Jerrin Shirks

import pygame



class Timer:
    def __init__(self,
                 duration: float,
                 func=None,
                 args=None,
                 kwargs=None):

        self.duration = duration

        self.func = func
        self.args = args
        self.kwargs = kwargs

        self.start_time = 0
        self.active = False

    def getPercentage(self):
        current_time = pygame.time.get_ticks()
        return (current_time - self.start_time) / self.duration

    def activate(self):
        self.active = True
        self.start_time = pygame.time.get_ticks()

    def deactivate(self):
        self.active = False
        self.start_time = 0

    def update(self):
        current_time = pygame.time.get_ticks()
        if current_time - self.start_time > self.duration:
            self.deactivate()
            if self.func:
                if args is not None:
                    if kwargs is not None:
                        self.func(*args, **kwargs)
                    else:
                        self.func(*args)
                elif kwargs is not None:
                    self.func(*kwargs)
                else:
                    self.func()


