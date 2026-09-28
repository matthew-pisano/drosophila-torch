"""Representations of neurotransmitter types."""

from enum import IntEnum


class NTType(IntEnum):
    """An enum of neurotransmitter types."""

    UNKNOWN = 0
    ACETYLCHOLINE = 1
    GABA = 2
    GLUTAMATE = 3
    DOPAMINE = 4
    SEROTONIN = 5
    OCTOPAMINE = 6

    def sign(self) -> float:
        return {
            NTType.UNKNOWN: 1.0,
            NTType.ACETYLCHOLINE: 1.0,
            NTType.GABA: -1.0,
            NTType.GLUTAMATE: -1.0,
            NTType.DOPAMINE: 1.0,
            NTType.SEROTONIN: 1.0,
            NTType.OCTOPAMINE: 1.0,
        }[self]

    def to_string(self) -> str:
        return {
            NTType.UNKNOWN: "unknown",
            NTType.ACETYLCHOLINE: "acetylcholine",
            NTType.GABA: "gaba",
            NTType.GLUTAMATE: "glutamate",
            NTType.DOPAMINE: "dopamine",
            NTType.SEROTONIN: "serotonin",
            NTType.OCTOPAMINE: "octopamine",
        }[self]

    @staticmethod
    def from_string(string: str) -> NTType:
        return {
            "acetylcholine": NTType.ACETYLCHOLINE,
            "gaba": NTType.GABA,
            "glutamate": NTType.GLUTAMATE,
            "dopamine": NTType.DOPAMINE,
            "serotonin": NTType.SEROTONIN,
            "octopamine": NTType.OCTOPAMINE,
        }.get(string, NTType.UNKNOWN)
