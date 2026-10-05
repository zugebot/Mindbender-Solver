import os
import asyncio

from chuzzleDeluxe import ChuzzleDeluxe
from engine.settings import Settings

async def main():
    if not os.path.isdir("cached"):
        os.mkdir("cached")

    settings = Settings()
    game = ChuzzleDeluxe(settings)

    while game.running:
        game.eventLoop()

        game.screen.fill((0, 0, 0))
        game.currentMenu().run(game.mousePos, game.mouseState)

        if len(game.menus) != game.lastMenuSize:
            game.currentMenu().update(game.screen, default=True)
            game.lastMenuImage = game.screen.copy()
            game.lastMenuSize = len(game.menus)
        else:
            game.currentMenu().update(game.screen)

        game._presentScreen()
        game.instance.display.flip()
        game.clock.tick(game.fps)

        await asyncio.sleep(0)

if __name__ == "__main__":
    asyncio.run(main())