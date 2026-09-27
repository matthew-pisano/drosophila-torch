"""Representations of sensory types."""

from enum import IntEnum


class SensoryType(IntEnum):
    UNKNOWN = 0
    """An unknown or unlabeled sensory type."""

    ENTERIC_NERVOUS_SYSTEM = 1
    """Gut-innervating neurons outside the central nervous system."""

    ASCENDING_NEURON = 2
    """Neurons which carry signals from the ventral nerve cord to the brain."""

    DESCENDING_NEURON = 3
    """Neurons which carry signals from the brain to the ventral nerve cord."""

    SENSORY_ASCENDING = 4
    """Neurons which cary sensory signals directly to the brain."""

    SENSORY_DESCENDING = 5
    """Neurons which cary sensory signals to the ventral nerve cord."""

    EFFERENT_ASCENDING = 6
    """Ventral nerve cord neurons which send signals to the peripheral nervous system and brain."""

    EFFERENT_DESCENDING = 7
    """Brain neurons which send signals to the peripheral nervous system."""

    CB_INTRINSIC = 8
    """Neurons internal to the central brain."""

    CB_SENSORY = 9
    """The primary sensory neurons in the central brain, carrying signals from the peripheral nervous system."""

    CB_MOTOR = 10
    """Motor neurons in the central brain, carrying signals to the neck."""

    CB_EFFERENT = 11
    """Modulatory neurons carrying signals from the central brain to the peripheral nervous system."""

    CB_ENDOCRINE = 12
    """Neurons which cary hormonal signals from the central brain to the corpora allata or corpora cardiaca."""

    OL_INTRINSIC = 13
    """Neurons internal to the optic lobes."""

    OL_SENSORY = 14
    """Neurons carrying signals directly from a compound eyes to its optic lobe."""

    VISUAL_PROJECTION = 15
    """Neurons which send signals from optic lobes to the central brain."""

    VISUAL_CENTRIFUGAL = 16
    """Neurons which send signals from the central brain to the optic lobe."""

    VNC_INTRINSIC = 17
    """Neurons internal to the ventral nerve cord."""

    VNC_SENSORY = 18
    """Neurons carrying signals from the peripheral nervous system to the ventral nerve cord."""

    VNC_MOTOR = 19
    """Ventral nerve cord neurons which cary motor signals."""

    VNC_EFFERENT = 20
    """Ventral nerve cord neurons which cary signals to the peripheral nervous system."""

    VNC_ENDOCRINE = 21
    """Neurons which cary hormonal signals from the ventral nerve cord to the corpora allata or corpora cardiaca."""

    def to_string(self):
        return {
            SensoryType.ENTERIC_NERVOUS_SYSTEM: "ENS",
            SensoryType.ASCENDING_NEURON: "ascending_neuron",
            SensoryType.DESCENDING_NEURON: "descending_neuron",
            SensoryType.SENSORY_ASCENDING: "sensory_ascending",
            SensoryType.SENSORY_DESCENDING: "sensory_descending",
            SensoryType.EFFERENT_ASCENDING: "efferent_ascending",
            SensoryType.EFFERENT_DESCENDING: "efferent_descending",
            SensoryType.CB_INTRINSIC: "cb_intrinsic",
            SensoryType.CB_SENSORY: "cb_sensory",
            SensoryType.CB_MOTOR: "cb_motor",
            SensoryType.CB_EFFERENT: "cb_efferent",
            SensoryType.CB_ENDOCRINE: "cb_endocrine",
            SensoryType.OL_INTRINSIC: "ol_intrinsic",
            SensoryType.OL_SENSORY: "ol_sensory",
            SensoryType.VISUAL_PROJECTION: "visual_projection",
            SensoryType.VISUAL_CENTRIFUGAL: "visual_centrifugal",
            SensoryType.VNC_INTRINSIC: "vnc_intrinsic",
            SensoryType.VNC_SENSORY: "vnc_sensory",
            SensoryType.VNC_MOTOR: "vnc_motor",
            SensoryType.VNC_EFFERENT: "vnc_efferent",
            SensoryType.VNC_ENDOCRINE: "vnc_endocrine",
        }[self]

    @staticmethod
    def from_string(string: str):
        return {
            "ENS": SensoryType.ENTERIC_NERVOUS_SYSTEM,
            "ascending_neuron": SensoryType.ASCENDING_NEURON,
            "descending_neuron": SensoryType.DESCENDING_NEURON,
            "descending_neuron_tbc": SensoryType.DESCENDING_NEURON,
            "sensory_ascending": SensoryType.SENSORY_ASCENDING,
            "sensory_ascending_tbc": SensoryType.SENSORY_ASCENDING,
            "sensory_descending": SensoryType.SENSORY_DESCENDING,
            "efferent_ascending": SensoryType.EFFERENT_ASCENDING,
            "efferent_descending": SensoryType.EFFERENT_DESCENDING,
            "cb_intrinsic": SensoryType.CB_INTRINSIC,
            "cb_sensory": SensoryType.CB_SENSORY,
            "cb_sensory_tbc": SensoryType.CB_SENSORY,
            "cb_motor": SensoryType.CB_MOTOR,
            "cb_efferent": SensoryType.CB_EFFERENT,
            "cb_endocrine": SensoryType.CB_ENDOCRINE,
            "ol_intrinsic": SensoryType.OL_INTRINSIC,
            "ol_sensory": SensoryType.OL_SENSORY,
            "visual_projection": SensoryType.VISUAL_PROJECTION,
            "visual_projection_tbc": SensoryType.VISUAL_PROJECTION,
            "visual_centrifugal": SensoryType.VISUAL_CENTRIFUGAL,
            "vnc_intrinsic": SensoryType.VNC_INTRINSIC,
            "vnc_sensory": SensoryType.VNC_SENSORY,
            "vnc_sensory_tbc": SensoryType.VNC_SENSORY,
            "vnc_motor": SensoryType.VNC_MOTOR,
            "vnc_efferent": SensoryType.VNC_EFFERENT,
            "vnc_endocrine": SensoryType.VNC_ENDOCRINE,
            "vnc_tbc": SensoryType.VNC_INTRINSIC,
        }.get(string, SensoryType.UNKNOWN)
