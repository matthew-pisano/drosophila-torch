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


NT_STRING_TO_ENUM = {
    "acetylcholine": NTType.ACETYLCHOLINE,
    "gaba": NTType.GABA,
    "glutamate": NTType.GLUTAMATE,
    "dopamine": NTType.DOPAMINE,
    "serotonin": NTType.SEROTONIN,
    "octopamine": NTType.OCTOPAMINE,
}

NT_SIGN = {
    NTType.UNKNOWN: 1.0,
    NTType.ACETYLCHOLINE: 1.0,
    NTType.GABA: -1.0,
    NTType.GLUTAMATE: -1.0,
    NTType.DOPAMINE: 1.0,
    NTType.SEROTONIN: 1.0,
    NTType.OCTOPAMINE: 1.0,
}
