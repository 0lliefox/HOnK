__author__ = "Giacomo Bergami"
__copyright__ = "Copyright 2025, Functional Match"
__credits__ = ["Giacomo Bergami"]
__license__ = "GPLv3"
__version__ = "2.0"
__maintainer__ = "Giacomo Bergami"
__email__ = "bergamigiacomo@gmail.com"
__status__ = "Production"

import dataclasses
import json
from typing import List, Tuple, Dict

import dacite


@dataclasses.dataclass(frozen=True, eq=True, order=True)
class LogicalConstructSpecs:
    attachTo: str
    argument: str
    property: str=None



@dataclasses.dataclass(frozen=True, eq=True, order=True)
class LogicalConstruct:
    name: str
    specs: Tuple[LogicalConstructSpecs]


@dataclasses.dataclass(frozen=True, eq=True, order=True)
class RewritingOutcome:
    type:str
    property:str=None

@dataclasses.dataclass()
class MatchingForRewriting:
    premise: dict
    classification: List[RewritingOutcome]

def load_logical_analysis(file)->Tuple[Dict[str, LogicalConstruct], List[MatchingForRewriting]]:
    data = None
    with open(file) as f:
        data = json.load(f)

    types = dict()
    for k,v in data["types"].items():
        types[k] = LogicalConstruct(k,tuple(dacite.from_dict(LogicalConstructSpecs, x) for x in v))

    rules = []
    for v in data["derivation_rules"]:
        rules.append(MatchingForRewriting(v["premise"], [dacite.from_dict(RewritingOutcome, x) for x in v["classification"]]))

    return types, rules


if __name__ == "__main__":
    data = list(load_logical_analysis("../../raw_data/logical_analysis.json"))
    print(data)
