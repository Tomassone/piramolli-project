import logging

import coloredlogs

from Coach import Coach
from tablut.TablutGame import TablutGame as Game
from tablut.NNet import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.


args = dotdict({
    'numIters': 100,            # Iterazioni totali di self-play
    'numEps': 50,               # Partite per iterazione (Tablut è lenta)
    'tempThreshold': 15,        # Dopo 10 mosse gioca deterministicamente
    'updateThreshold': 0.6,    # Accetta nuova rete se vince 55%+ nell'Arena
    'maxlenOfQueue': 200000,    # Esempi massimi in memoria
    'numMCTSSims': 100,          # Simulazioni MCTS per mossa durante self-play
    'arenaCompare': 30,         # Partite di confronto vecchia vs nuova rete
    'cpuct': 1.0,
    'checkpoint': './checkpoints/new/',
    'load_model': False,
    'load_folder_file': ('./checkpoints/new', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 10,
})

def main():
    log.info('Loading %s...', Game.__name__)
    g = Game()

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
