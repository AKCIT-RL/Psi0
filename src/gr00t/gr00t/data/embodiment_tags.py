from enum import Enum


"""
Embodiment tags are used to identify the robot embodiment in the data.

Naming convention:
<dataset>_<robot_name>

If using multiple datasets, e.g. sim GR1 and real GR1, we can drop the dataset name and use only the robot name.
"""


class EmbodimentTag(Enum):
    ##### N1.6 pretrain embodiment tags #####
    ROBOCASA_PANDA_OMRON = "robocasa_panda_omron"
    """
    The RoboCasa Panda robot with omron mobile base.
    """

    GR1 = "gr1"
    """
    The Fourier GR1 robot.
    """

    ##### N1.6 pre-registered posttrain embodiment tags #####
    UNITREE_G1 = "unitree_g1"
    """
    The Unitree G1 robot (N1.6).
    """

    LIBERO_PANDA = "libero_panda"
    """
    The Libero panda robot (N1.6).
    """

    OXE_GOOGLE = "oxe_google"
    """
    The Open-X-Embodiment Google robot.
    """

    OXE_WIDOWX = "oxe_widowx"
    """
    The Open-X-Embodiment WidowX robot.
    """

    BEHAVIOR_R1_PRO = "behavior_r1_pro"
    """
    The Behavior R1 Pro robot.
    """

    G1_EE_A16 = "g1_ee_a16"
    """
    G1 end-effector pretraining embodiment with action horizon 16.
    """

    H1_EE_A16 = "h1_ee_a16"
    """
    H1 end-effector pretraining embodiment with action horizon 16.
    """

    G1_LOCO_DOWNSTREAM = "g1_loco_downstream"
    """
    G1 locomotion downstream fine-tuning embodiment.
    """

    G1_UPPER_A16 = "g1_upper_a16"
    """
    G1 upper-body manipulation pretraining embodiment with action horizon 16.
    """

    ##### N1.7 pretrain embodiment tags (baked into nvidia/GR00T-N1.7-3B) #####

    OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT = "oxe_droid_relative_eef_relative_joint"
    """
    The Open-X-Embodiment DROID robot with relative EEF and relative joint position actions.
    """

    XDOF = "xdof_relative_eef_relative_joint"
    """
    The generic X-DOF robot with relative EEF and relative joint position actions.
    """

    XDOF_SUBTASK = "xdof_relative_eef_relative_joint_subtask"
    """
    The generic X-DOF robot (subtask variant).
    """

    REAL_G1 = "real_g1_relative_eef_relative_joints"
    """
    Real-world Unitree G1 with relative EEF and relative joint actions.
    """

    REAL_R1_PRO_SHARPA = "real_r1_pro_sharpa_relative_eef"
    """
    Real-world R1 Pro Sharpa with relative EEF actions.
    """

    REAL_R1_PRO_SHARPA_HUMAN = "real_r1_pro_sharpa_relative_eef_human"
    """
    Real-world R1 Pro Sharpa with relative EEF actions (human teleop data).
    """

    REAL_R1_PRO_SHARPA_MAXINSIGHTS = "real_r1_pro_sharpa_relative_eef_maxinsights"
    """
    Real-world R1 Pro Sharpa with relative EEF actions (MaxInsights data, single-cam).
    """

    REAL_R1_PRO_SHARPA_MECKA = "real_r1_pro_sharpa_relative_eef_mecka"
    """
    Real-world R1 Pro Sharpa with relative EEF actions (Mecka data, single-cam).
    """

    ##### N1.7 pre-registered posttrain embodiment tags #####

    UNITREE_G1_N1D7 = "unitree_g1_full_body_with_waist_height_nav_cmd"
    """
    The Unitree G1 robot (N1.7, full-body with waist height and nav commands).
    """

    UNITREE_G1_SONIC = "unitree_g1_sonic"
    """
    The Unitree G1 robot with SONIC whole-body controller.
    """

    SIMPLER_ENV_GOOGLE = "simpler_env_google"
    """
    The SimplerEnv Google robot.
    """

    SIMPLER_ENV_WIDOWX = "simpler_env_widowx"
    """
    The SimplerEnv WidowX robot.
    """

    LIBERO_PANDA_N1D7 = "libero_sim"
    """
    The LIBERO Panda robot for N1.7 (LIBERO-Goal, LIBERO-Object, LIBERO-Spatial, LIBERO-10).
    """

    # New embodiment during post-training
    NEW_EMBODIMENT = "new_embodiment"
    """
    Any new embodiment.
    """

    @classmethod
    def resolve(cls, tag: "str | EmbodimentTag") -> "EmbodimentTag":
        """Resolve a string to an EmbodimentTag, case-insensitively.

        Matches by enum **name** first (e.g. ``"xdof"`` -> ``XDOF``), then by
        enum **value** (e.g. ``"xdof_relative_eef_relative_joint"`` -> ``XDOF``).

        Raises:
            ValueError: If *tag* does not match any known embodiment.
        """
        if isinstance(tag, cls):
            return tag
        key = tag.strip()
        key_lower = key.lower()
        # Match by enum name (case-insensitive)
        for member in cls:
            if member.name.lower() == key_lower:
                return member
        # Match by enum value (case-insensitive)
        for member in cls:
            if member.value.lower() == key_lower:
                return member
        raise ValueError(
            f"Unknown embodiment tag: {tag!r}\n"
            f"Known tags: {[m.name for m in cls]}"
        )

    @classmethod
    def reverse_lookup(cls, value: str) -> "str":
        """Map a tag value string back to its enum name, or return the value as-is."""
        for member in cls:
            if member.value == value:
                return member.name
        return value


# Module-level tag category sets for N1.7 (cannot be Enum class attributes).
PRETRAIN_TAGS: frozenset[EmbodimentTag] = frozenset(
    {
        EmbodimentTag.OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT,
        EmbodimentTag.XDOF,
        EmbodimentTag.XDOF_SUBTASK,
        EmbodimentTag.REAL_G1,
        EmbodimentTag.REAL_R1_PRO_SHARPA,
        EmbodimentTag.REAL_R1_PRO_SHARPA_HUMAN,
        EmbodimentTag.REAL_R1_PRO_SHARPA_MAXINSIGHTS,
        EmbodimentTag.REAL_R1_PRO_SHARPA_MECKA,
    }
)
"""Tags baked into the base model (nvidia/GR00T-N1.7-3B) — usable without finetuning."""

POSTTRAIN_TAGS: frozenset[EmbodimentTag] = frozenset(
    {
        EmbodimentTag.UNITREE_G1_N1D7,
        EmbodimentTag.UNITREE_G1_SONIC,
        EmbodimentTag.SIMPLER_ENV_GOOGLE,
        EmbodimentTag.SIMPLER_ENV_WIDOWX,
        EmbodimentTag.LIBERO_PANDA_N1D7,
    }
)
"""Tags that require a finetuned N1.7 checkpoint."""

FINETUNE_ONLY_TAGS: frozenset[EmbodimentTag] = frozenset(
    {
        EmbodimentTag.NEW_EMBODIMENT,
    }
)
"""Tags for custom robots — always require a finetuned checkpoint."""
