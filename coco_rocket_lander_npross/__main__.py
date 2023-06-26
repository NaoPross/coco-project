"""Rocket Lander Project for Computational Control

Usage:
    rocket [-v] [-r] [-o] [-s SCEN] [ALGORITHM...]
    rocket [-v] [-r] [-o] [-s SCEN] -e FILE ALGORITHM
    rocket [-v] [-p NUM] [-s SCEN] -e FILE [ALGORITHM...]
    rocket [-v] -l

Options:
    -h, --help                  Show this screen
    -v, --verbose               Be verbose (print debug statements)
    -l, --list                  List algorithms
    -r, --record                Record a video file of the simulation
    -o, --open-video            Show video when simulation is complete (implies -r)
    -R, --random                Run with random initial conditions
    -e FILE, --export FILE      Export trajectory to DAT file, or solvetimes when ran with -p
    -s SCEN, --scenario SCEN    Run only scenario number SCEN
    -p NUM, --parallel NUM      Run NUM parallel runs of the simulation, cannot be used with -r
"""
# vim: ts=4 sw=4 noet spell tw=80:
import sys
import os
import subprocess
import pathlib

from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
import colorlog as logging
import docopt
args = docopt.docopt(__doc__)

# Set up logging
log = logging.getLogger()

if args["--verbose"]:
    log.setLevel(logging.DEBUG)
else:
    log.setLevel(logging.INFO)

handler = logging.StreamHandler()

if "colorlog" in sys.modules and os.isatty(2):
    handler.setFormatter(logging.ColoredFormatter(
        "%(log_color)s%(levelname)s:%(reset)s %(message)s",
        log_colors = {
            "DEBUG": "green", "INFO": "blue",
            "WARNING": "yellow", "ERROR": "red",
            "CRITICAL": "bold_red"}))
else:
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

log.addHandler(handler)

try:
    from cpuinfo import get_cpu_info
    cpu_info = get_cpu_info()
    CPU_SPEED_HZ = cpu_info["hz_actual"][0]
except ImportError:
    CPU_SPEED_HZ = 3.22e9  # Apple M1 Pro max speed
    log.warning(f"cpuinfo not installed, manually set CPU_SPEED_HZ = {CPU_SPEED_HZ}")

import scipy
import numpy as np
np.set_printoptions(precision=2)

from coco_rocket_lander_npross.simulator import Simulator, scenarios, mp_simulate
from coco_rocket_lander_npross.algorithms import PID, ClassicMPC, RelaxedMPC, ParametricMPC


def open_file(filename):
    if sys.platform == "win32":
        os.startfile(filename)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, filename])

log.debug("Ran with args = " + str(args))

# ╻  ╻┏━┓╺┳╸   ┏━┓╻  ┏━╸┏━┓┏━┓╻╺┳╸╻ ╻┏┳┓┏━┓
# ┃  ┃┗━┓ ┃    ┣━┫┃  ┃╺┓┃ ┃┣┳┛┃ ┃ ┣━┫┃┃┃┗━┓
# ┗━╸╹┗━┛ ╹    ╹ ╹┗━╸┗━┛┗━┛╹┗╸╹ ╹ ╹ ╹╹ ╹┗━┛

ALGORITHMS = {
    "pid": PID,
    "cmpc": ClassicMPC,
    "rmpc": RelaxedMPC,
    "pmpc": ParametricMPC,
}

# List algorithms and exit
if args["--list"]:
    log.info("Listing available algorithms")
    for (name, obj) in ALGORITHMS.items():
        log.info(f"\t{name}: {obj.name}")
    # Quit here
    sys.exit(0)

# ┏━┓┏━╸╻  ┏━╸┏━╸╺┳╸   ┏━┓╻  ┏━╸┏━┓┏━┓╻╺┳╸╻ ╻┏┳┓┏━┓
# ┗━┓┣╸ ┃  ┣╸ ┃   ┃    ┣━┫┃  ┃╺┓┃ ┃┣┳┛┃ ┃ ┣━┫┃┃┃┗━┓
# ┗━┛┗━╸┗━╸┗━╸┗━╸ ╹    ╹ ╹┗━╸┗━┛┗━┛╹┗╸╹ ╹ ╹ ╹╹ ╹┗━┛

# Run only some algorithms
if args["ALGORITHM"]:
    to_run = {}
    for name in args["ALGORITHM"]:
        if not name in ALGORITHMS.keys():
            log.error(f"Algorithm {name} does not exits")
            continue
        to_run[name] = ALGORITHMS.get(name)

    ALGORITHMS = to_run
    log.info(f"Running algorithms: {', '.join(ALGORITHMS.keys())}")

# ┏━┓┏━╸╻  ┏━╸┏━╸╺┳╸   ┏━┓┏━╸┏━╸┏┓╻┏━┓┏━┓╻┏━┓┏━┓
# ┗━┓┣╸ ┃  ┣╸ ┃   ┃    ┗━┓┃  ┣╸ ┃┗┫┣━┫┣┳┛┃┃ ┃┗━┓
# ┗━┛┗━╸┗━╸┗━╸┗━╸ ╹    ┗━┛┗━╸┗━╸╹ ╹╹ ╹╹┗╸╹┗━┛┗━┛

if scen := args["--scenario"]:
    try:
        scen = int(scen)
    except ValueError:
        log.error(f"Invalid scenario {scen}, must be a number")
        sys.exit(1)

    if not 0 <= scen < len(scenarios):
        log.error(f"SCEN must be in [0, ..., {len(scenarios)-1}].")
        sys.exit(1)

    scenarios = [(scen, scenarios[scen])]

else:
    scenarios = list(enumerate(scenarios))

# ┏━┓┏━┓┏━┓┏━┓╻  ╻  ┏━╸╻     ┏┳┓┏━┓╺┳┓┏━╸
# ┣━┛┣━┫┣┳┛┣━┫┃  ┃  ┣╸ ┃     ┃┃┃┃ ┃ ┃┃┣╸
# ╹  ╹ ╹╹┗╸╹ ╹┗━╸┗━╸┗━╸┗━╸   ╹ ╹┗━┛╺┻┛┗━╸
# NOTE: this is mainly built to collect data of OnlineMPC runs, though it
# should work with the other algorithms too

# Run multiple simulations in parallel to get solvetime data
if nsim := args["--parallel"]:
    try:
        nsim = int(nsim)
        if nsim <= 0:
            raise ValueError
    except ValueError:
        log.error(f"'{nsim}' is not a positive non-zero number!")

    # Array to store rows of (sample_time, solve time mean, solve time std err)
    datapoints = []
    for (name, alg) in ALGORITHMS.items():
        def parallel_simulations(algs, scenarios, nsim):
            # Create pool executor
            futures = []
            with ProcessPoolExecutor() as ex:
                for alg, (i, params), _ in product(algs, scenarios, range(nsim)):
                    futures.append(ex.submit(mp_simulate, alg, params, i))

                with tqdm(total=len(futures), leave=False) as pbar:
                    for f in as_completed(futures):
                        yield f.result()
                        pbar.update(1)

        # compute mean solve time normalized wrt cpu frequency
        time_horizon = 10
        sample_times = 1e-3 * np.array([250, 100, 50, 20, 10])

        log.info(f"Testing time horizon of {time_horizon}s with sample times:")
        for ts in sample_times:
            h = int(time_horizon / ts)
            log.info(f"\t{ts:.3e}s = {ts * CPU_SPEED_HZ:.3e} cycles = horizon of {h}")

        # Create an algorithm object for each sample_time
        algs_with_ts = map(alg, sample_times)
        simulations = parallel_simulations(algs_with_ts, scenarios, nsim)

        # Collect all samples as they are ready
        solvetimes = {ts: [] for ts in sample_times}
        for (sim, done) in simulations:
            solvetimes[sim.alg.sample_time].extend(sim.solvetimes)
            if not done:
                log.warning(f"Simulation {sim} failed!")

        # Compute statistical values
        for sample_time, samples in solvetimes.items():
            samples = np.array(samples)
            sample_time_cycles = sample_time * CPU_SPEED_HZ

            mean, se = np.mean(samples), np.std(samples) / np.sqrt(samples.size)
            mean_cycles, se_cycles = mean * CPU_SPEED_HZ, se * CPU_SPEED_HZ


            data = (sample_time, sample_time_cycles, mean, se, mean_cycles, se_cycles)
            datapoints.append(data)

            log.info(f"For sample time = {sample_time:.3e} s = {sample_time_cycles:.3e} cycles")
            log.info(f"\tSolve time: mean = {mean:.3e} s, se = {se:.3e} s")
            log.info(f"\tMean updates per second = 1 / mean = {1 / mean:.1f} Hz")
            log.info(f"\tSolve cycles: mean = {mean_cycles:.0f}, se = {se_cycles:.0f}")

            if mean_cycles > sample_time_cycles:
                log.warning("Algorithm was slower than sampling time!")

    # save data
    file = pathlib.Path(args["--export"])
    log.info(f"Saving statistics time to {file}")
    if not file.resolve().parent.exists():
        log.error(f"Parent directory {file.resolve().parent} does not exists!")

    else:
        # Last two rows are only for MPC controllers
        log.warning("Adding extra columns for MPC controllers")
        header = ["sampletime", "sampletime-cycles", "time-mean", "time-se",
                  "cycles-mean", "cycles-se", "sampletime-ms", "horizon"]

        with file.resolve().open("w") as f:
            f.write("\t".join(header) + "\n")
            for row in datapoints:
                # Add extra MPC data easily import in PGFplots
                sample_time = row[0]
                sample_time_ms = "{:.0f}".format(sample_time * 1e3)
                horizon = "{:.0f}".format(int(time_horizon / sample_time))
                mpc_data = [sample_time_ms, horizon]

                # The others are formatted using scientific notation
                format_number = lambda n: "{:.4e}".format(n)
                row = list(map(format_number, row))

                f.write("\t".join(row + mpc_data) + "\n")

    # Quit here
    sys.exit(0)

# ┏┓╻┏━┓┏━┓┏┳┓┏━┓╻     ┏┳┓┏━┓╺┳┓┏━╸
# ┃┗┫┃ ┃┣┳┛┃┃┃┣━┫┃     ┃┃┃┃ ┃ ┃┃┣╸ 
# ╹ ╹┗━┛╹┗╸╹ ╹╹ ╹┗━╸   ╹ ╹┗━┛╺┻┛┗━╸

# Run only one simulation
for (name, alg) in ALGORITHMS.items():
    for i, params in scenarios:
        log.info(f"Simulating scenario {i}")
        record = args["--record"] or args["--open-video"]
        sim = Simulator(alg(), userparams=params, scenario=i, record_video=record)
        sim.run()

        # Save trajectory
        if args["--export"]:
            file = pathlib.Path(args["--export"])
            file = file.with_name(file.stem + "-" + str(i) + file.suffix)

            log.info(f"Saving trajectory to {file}")
            if file.resolve().parent.exists():
                nsamples = sim.trajectory.shape[0]
                q = nsamples // 300
                if q > 1:
                    log.info(f"Trajectory has {nsamples} samples, downsampling "
                      + f"by factor {q} to get {nsamples // q} samples")
                    trajectory = scipy.signal.decimate(sim.trajectory, q, axis=0)
                else:
                    trajectory = sim.trajectory

                header = ["x", "y", "xdot", "ydot", "theta", "thetadot", "cl", "cr"]
                np.savetxt(file, trajectory, delimiter="\t", 
                  header="\t".join(header), comments="")
            else:
                log.error(f"Parent directory {file.resolve().parent} does not exists!")

        # Show video
        if args["--open-video"] and sim.has_video:
            if sim.videofile.resolve().exists():
                log.info(f"Opening video file at {sim.videofile}")
                open_file(sim.videofile)
            else:
                log.error(f"Cannot find video file {sim.videofile}")

