"""Representations of neuron superclass types."""

from enum import IntEnum


class NeuronSuperclass(IntEnum):
    UNKNOWN = 17
    """An unknown or unlabeled sensory type."""

    ENTERIC_NERVOUS_SYSTEM = 0
    """Gut-innervating neurons outside the central nervous system."""

    ASCENDING_NEURON = 1
    """Neurons which carry signals from the ventral nerve cord to the brain."""

    DESCENDING_NEURON = 8
    """Neurons which carry signals from the brain to the ventral nerve cord."""

    SENSORY_ASCENDING = 14
    """Neurons which cary sensory signals directly to the brain."""

    SENSORY_DESCENDING = 16
    """Neurons which cary sensory signals to the ventral nerve cord."""

    EFFERENT_ASCENDING = 10
    """Ventral nerve cord neurons which send signals to the peripheral nervous system and brain."""

    EFFERENT_DESCENDING = 11
    """Brain neurons which send signals to the peripheral nervous system."""

    CB_INTRINSIC = 4
    """Neurons internal to the central brain."""

    CB_SENSORY = 6
    """The primary sensory neurons in the central brain, carrying signals from the peripheral nervous system."""

    CB_MOTOR = 5
    """Motor neurons in the central brain, carrying signals to the neck."""

    CB_EFFERENT = 2
    """Modulatory neurons carrying signals from the central brain to the peripheral nervous system."""

    CB_ENDOCRINE = 3
    """Neurons which cary hormonal signals from the central brain to the corpora allata or corpora cardiaca."""

    OL_INTRINSIC = 12
    """Neurons internal to the optic lobes."""

    OL_SENSORY = 13
    """Neurons carrying signals directly from a compound eyes to its optic lobe."""

    VISUAL_PROJECTION = 19
    """Neurons which send signals from optic lobes to the central brain."""

    VISUAL_CENTRIFUGAL = 18
    """Neurons which send signals from the central brain to the optic lobe."""

    VNC_INTRINSIC = 23
    """Neurons internal to the ventral nerve cord."""

    VNC_SENSORY = 25
    """Neurons carrying signals from the peripheral nervous system to the ventral nerve cord."""

    VNC_MOTOR = 24
    """Ventral nerve cord neurons which cary motor signals."""

    VNC_EFFERENT = 21
    """Ventral nerve cord neurons which cary signals to the peripheral nervous system."""

    VNC_ENDOCRINE = 22
    """Neurons which cary hormonal signals from the ventral nerve cord to the corpora allata or corpora cardiaca."""

    def to_string(self):
        return {
            NeuronSuperclass.ENTERIC_NERVOUS_SYSTEM: "ENS",
            NeuronSuperclass.ASCENDING_NEURON: "ascending_neuron",
            NeuronSuperclass.DESCENDING_NEURON: "descending_neuron",
            NeuronSuperclass.SENSORY_ASCENDING: "sensory_ascending",
            NeuronSuperclass.SENSORY_DESCENDING: "sensory_descending",
            NeuronSuperclass.EFFERENT_ASCENDING: "efferent_ascending",
            NeuronSuperclass.EFFERENT_DESCENDING: "efferent_descending",
            NeuronSuperclass.CB_INTRINSIC: "cb_intrinsic",
            NeuronSuperclass.CB_SENSORY: "cb_sensory",
            NeuronSuperclass.CB_MOTOR: "cb_motor",
            NeuronSuperclass.CB_EFFERENT: "cb_efferent",
            NeuronSuperclass.CB_ENDOCRINE: "cb_endocrine",
            NeuronSuperclass.OL_INTRINSIC: "ol_intrinsic",
            NeuronSuperclass.OL_SENSORY: "ol_sensory",
            NeuronSuperclass.VISUAL_PROJECTION: "visual_projection",
            NeuronSuperclass.VISUAL_CENTRIFUGAL: "visual_centrifugal",
            NeuronSuperclass.VNC_INTRINSIC: "vnc_intrinsic",
            NeuronSuperclass.VNC_SENSORY: "vnc_sensory",
            NeuronSuperclass.VNC_MOTOR: "vnc_motor",
            NeuronSuperclass.VNC_EFFERENT: "vnc_efferent",
            NeuronSuperclass.VNC_ENDOCRINE: "vnc_endocrine",
        }[self]

    @staticmethod
    def from_string(string: str):
        return {
            "ENS": NeuronSuperclass.ENTERIC_NERVOUS_SYSTEM,
            "ascending_neuron": NeuronSuperclass.ASCENDING_NEURON,
            "descending_neuron": NeuronSuperclass.DESCENDING_NEURON,
            "descending_neuron_tbc": NeuronSuperclass.DESCENDING_NEURON,
            "sensory_ascending": NeuronSuperclass.SENSORY_ASCENDING,
            "sensory_ascending_tbc": NeuronSuperclass.SENSORY_ASCENDING,
            "sensory_descending": NeuronSuperclass.SENSORY_DESCENDING,
            "efferent_ascending": NeuronSuperclass.EFFERENT_ASCENDING,
            "efferent_descending": NeuronSuperclass.EFFERENT_DESCENDING,
            "cb_intrinsic": NeuronSuperclass.CB_INTRINSIC,
            "cb_sensory": NeuronSuperclass.CB_SENSORY,
            "cb_sensory_tbc": NeuronSuperclass.CB_SENSORY,
            "cb_motor": NeuronSuperclass.CB_MOTOR,
            "cb_efferent": NeuronSuperclass.CB_EFFERENT,
            "cb_endocrine": NeuronSuperclass.CB_ENDOCRINE,
            "ol_intrinsic": NeuronSuperclass.OL_INTRINSIC,
            "ol_sensory": NeuronSuperclass.OL_SENSORY,
            "visual_projection": NeuronSuperclass.VISUAL_PROJECTION,
            "visual_projection_tbc": NeuronSuperclass.VISUAL_PROJECTION,
            "visual_centrifugal": NeuronSuperclass.VISUAL_CENTRIFUGAL,
            "vnc_intrinsic": NeuronSuperclass.VNC_INTRINSIC,
            "vnc_sensory": NeuronSuperclass.VNC_SENSORY,
            "vnc_sensory_tbc": NeuronSuperclass.VNC_SENSORY,
            "vnc_motor": NeuronSuperclass.VNC_MOTOR,
            "vnc_efferent": NeuronSuperclass.VNC_EFFERENT,
            "vnc_endocrine": NeuronSuperclass.VNC_ENDOCRINE,
            "vnc_tbc": NeuronSuperclass.VNC_INTRINSIC,
        }.get(string, NeuronSuperclass.UNKNOWN)
