# vim: ts=4 sw=4 noet spell tw=80:

import sys
import pathlib
import logging
log = logging.getLogger(__name__)

import gymnasium as gym
import numpy as np
from tqdm import tqdm

import coco_rocket_lander

scenarios = [
    # bad initial
    {"initial_position": (0.50, 0.9, 0.0)},   #  0
    {"initial_position": (0.45, 0.9, 0.0)},   #  1
    {"initial_position": (0.30, 0.9, 0.0)},   #  2
    {"initial_position": (0.10, 0.9, 0.0)},   #  3
    # bad initial angle
    {"initial_position": (0.50, 0.9, 0.3)},   #  4
    {"initial_position": (0.50, 0.9, 0.45)},  #  5
    {"initial_position": (0.50, 0.9, 0.6)},   #  6
    # bad initial y
    {"initial_position": (0.3, 0.9, 0.0)},    #  7
    {"initial_position": (0.3, 0.6, 0.0)},    #  8
    {"initial_position": (0.3, 0.3, 0.0)},    #  9
    # super bad
    {"initial_position": (0.1, 0.9, 0.6)},    # 10
    {"initial_position": (0.1, 0.5, 0.6)},    # 11
    {"initial_position": (0.1, 0.4, 0.3)},    # 12
]

class Simulator:
    def __init__(self, algorithm, userparams={}, scenario=0, max_niters=5000,
              record_video=False, interactive=True, seed=0):
        self.alg = algorithm
        self.max_niters = max_niters
        self.interactive = interactive
        self.seed = seed

        self.trajectory = []
        self.actions = []
        self.solvetimes = []

        # Simulation environment
        # Custom scenario
        if "random_initial_position" in userparams.keys() or \
                "initial_position" in userparams.keys():
            self.userparams = userparams
            scenario = "custom"

        # A scenario from the array above
        else:
            self.userparams = scenarios[scenario]
            # Add any additional stuff to the scenario
            for k,v in userparams.items():
                self.userparams[k] = v

        self.env = gym.make("coco_rocket_lander/RocketLander-v0",
                      render_mode="rgb_array", args=self.userparams)

        # Video
        video_prefix = "-".join([self.alg.name, "scenario", str(scenario)])
        if record_video:
            self.has_video = record_video
            always = lambda x: True
            self.env = gym.wrappers.RecordVideo(self.env, "video",
                                       episode_trigger=always,
                                       name_prefix=video_prefix,
                                       disable_logger=True)

        self.videofile = pathlib.Path(f"./video/{video_prefix}-episode-0.mp4")

        log.info(f"Set up simulator for {self.alg.name} "
           f"with userparams={userparams}, ")

    def run(self):
        log.info(f"Runnning simulator for {self.alg.name}")

        done = False
        obs, info = self.env.reset(seed=self.seed)

        lpos = np.array(self.env.get_landing_position())
        self.alg.setup(self.env)

        for i in tqdm(range(self.max_niters), leave=False,
                disable=not self.interactive, file=sys.stdout):

            try:
                self.trajectory.append(obs)
                action, solvetime = self.alg.run(np.array(obs), self.env)
                self.actions.append(action)
                self.solvetimes.append(solvetime)
            except RuntimeError as e:
                log.error(f"Simulation failed: {str(e)}")
                break

            next_obs, rewards, done, _, info = self.env.step(action)
            obs = next_obs

            if done:
                break

        if not done:
            log.warning("Simulation terminated early! (needs more iterations)")

        self.trajectory = np.vstack(self.trajectory)
        self.actions = np.vstack(self.actions)

        self.env.close()
        log.info(f"Simulation completed in {i+1} iterations")

        return done


def mp_simulate(alg, params, scenario):
    sim = Simulator(alg, userparams=params, scenario=scenario,
                    record_video=False, interactive=False)
    done = sim.run()

    # delete environment to make object pickable
    sim.alg.make_picklable()
    sim.env = None
    return sim, done
