from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio.types import (
    Route,
    Mission,
    Scenario,
)

missions = [
    Mission(Route(begin=("edge-south-SN", 1, 40), end=("edge-west-EW", 0, 60)))
]

gen_scenario(
    Scenario(ego_missions=missions), output_dir=Path(__file__).parent,
)
