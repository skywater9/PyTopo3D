from pytopo3d.utils import config_loader
from pytopo3d.utils.assembly import build_force_field, build_support_mask


def test_force_field_preset_lookup_is_case_insensitive(monkeypatch):
    preset = {
        "x1": 0,
        "x2": 1,
        "y1": 2,
        "y2": 3,
        "z1": 4,
        "z2": 5,
        "F_x": 6,
        "F_y": 7,
        "F_z": 8,
    }
    monkeypatch.setattr(config_loader, "force_fields", {"force_field_TO_A": preset})

    assert config_loader.get_force_field_params("force_field_TO_A") == tuple(
        range(9)
    )
    assert config_loader.get_force_field_params("FORCE_FIELD_TO_A") == tuple(
        range(9)
    )


def test_support_and_protected_zone_lookups_accept_mixed_case_keys(monkeypatch):
    preset = {"x1": 0, "x2": 1, "y1": 2, "y2": 3, "z1": 4, "z2": 5}
    expected = tuple(range(6))
    monkeypatch.setattr(config_loader, "support_masks", {"support_mask_TO_A": preset})
    monkeypatch.setattr(
        config_loader, "protected_zones", {"protected_zone_TO_A": preset}
    )

    assert config_loader.get_support_mask_params("support_mask_TO_A") == expected
    assert config_loader.get_protected_zone_ranges(
        ("protected_zone_TO_A",)
    ) == (expected,)


def test_obstacle_range_lookup_accepts_multiple_case_insensitive_names(monkeypatch):
    first = {"x1": 0, "x2": 1, "y1": 2, "y2": 3, "z1": 4, "z2": 5}
    second = {"x1": 6, "x2": 7, "y1": 8, "y2": 9, "z1": 10, "z2": 11}
    monkeypatch.setattr(
        config_loader,
        "obstacle_ranges",
        {"Cooling_Channel": first, "FIXTURE_CLEARANCE": second},
    )

    assert config_loader.get_obstacle_ranges(
        ("cooling_channel", "fixture_clearance")
    ) == (tuple(range(6)), tuple(range(6, 12)))


def test_to_a_boundary_presets_fit_180_by_83_by_1_domain():
    force_field = build_force_field(
        180,
        83,
        1,
        *config_loader.get_force_field_params("force_field_TO_A"),
    )
    support_mask = build_support_mask(
        180,
        83,
        1,
        *config_loader.get_support_mask_params("support_mask_TO_A"),
    )
    protected_ranges = config_loader.get_protected_zone_ranges(
        ("protected_zone_TO_A_support", "protected_zone_TO_A_force")
    )

    assert force_field.shape == (83, 180, 1, 3)
    assert support_mask.shape == (83, 180, 1)
    assert all(
        0 <= x1 < x2 <= 180
        and 0 <= y1 < y2 <= 83
        and 0 <= z1 < z2 <= 1
        for x1, x2, y1, y2, z1, z2 in protected_ranges
    )
