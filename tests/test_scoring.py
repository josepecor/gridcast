"""Tests de invariantes de dominio: puntos, componentes PU, penalizaciones."""

import pytest
from gridcast.config import get_scoring, get_components, get_circuits


class TestRaceScoring:
    def test_points_top10(self):
        s = get_scoring().race.positions
        expected = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
        assert s == expected

    def test_no_fastest_lap_point(self):
        assert get_scoring().race.fastest_lap_point is False

    def test_only_top10_score(self):
        s = get_scoring().race.positions
        assert set(s.keys()) == set(range(1, 11))

    def test_points_strictly_decreasing(self):
        pts = [v for _, v in sorted(get_scoring().race.positions.items())]
        assert pts == sorted(pts, reverse=True)


class TestSprintScoring:
    def test_points_top8(self):
        s = get_scoring().sprint.positions
        expected = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}
        assert s == expected

    def test_only_top8_score(self):
        s = get_scoring().sprint.positions
        assert set(s.keys()) == set(range(1, 9))


class TestComponentAllocations:
    def test_mgu_h_eliminated(self):
        alloc = get_components().allocations_2026
        assert alloc.MGU_H == 0, "MGU-H fue eliminado en 2026"

    def test_2026_bonus_allocations(self):
        alloc = get_components().allocations_2026
        assert alloc.ICE == 4
        assert alloc.TC == 4
        assert alloc.EX == 4
        assert alloc.MGU_K == 3
        assert alloc.ES == 3
        assert alloc.CE == 3

    def test_penalty_first_element(self):
        p = get_components().penalties
        assert p.first_extra_element == 10

    def test_penalty_subsequent_elements(self):
        p = get_components().penalties
        assert p.subsequent_elements == 5

    def test_pit_lane_threshold(self):
        p = get_components().penalties
        assert p.pit_lane_threshold == 15


class TestCircuits:
    def test_22_rounds(self):
        # Bahrain y Arabia Saudí cancelados; calendario 2026 = 22 rondas
        circuits = get_circuits()
        rounds = [c.round_2026 for c in circuits.values()]
        assert sorted(rounds) == list(range(1, 23))

    def test_no_duplicate_rounds(self):
        circuits = get_circuits()
        rounds = [c.round_2026 for c in circuits.values()]
        assert len(rounds) == len(set(rounds))

    def test_exactly_6_sprint_weekends(self):
        # China (R2), Miami (R4), Canadá (R5), Gran Bretaña (R9), Países Bajos (R12), Singapur (R16)
        circuits = get_circuits()
        sprint = [c for c in circuits.values() if c.weekend_format == "sprint"]
        assert len(sprint) == 6

    def test_sprint_circuit_names(self):
        circuits = get_circuits()
        sprint_rounds = sorted(
            c.round_2026 for c in circuits.values() if c.weekend_format == "sprint"
        )
        assert sprint_rounds == [2, 4, 5, 9, 12, 16]

    def test_valid_weekend_formats(self):
        circuits = get_circuits()
        formats = {c.weekend_format for c in circuits.values()}
        assert formats <= {"normal", "sprint"}

    def test_jolpica_ids_unique(self):
        circuits = get_circuits()
        ids = [c.jolpica_id for c in circuits.values()]
        assert len(ids) == len(set(ids)), "Jolpica circuit IDs duplicados"

    def test_jolpica_ids_match_api(self):
        # IDs verificados contra api.jolpi.ca/ergast/f1/2026.json
        expected = {
            "albert_park", "shanghai", "suzuka", "miami", "villeneuve",
            "monaco", "catalunya", "red_bull_ring", "silverstone", "spa",
            "hungaroring", "zandvoort", "monza", "madring", "baku",
            "marina_bay", "americas", "rodriguez", "interlagos", "vegas",
            "losail", "yas_marina",
        }
        circuits = get_circuits()
        actual = {c.jolpica_id for c in circuits.values()}
        assert actual == expected
