"""
Unit tests for Candidate Comparison Module.
Tests multi-candidate comparison, empty skills, single candidate handling,
and set union/intersection edge cases.
"""

import pytest
from modules.candidate_comparison import CandidateComparator


def test_compare_insufficient_candidates():
    # Empty list
    res_empty = CandidateComparator.compare_candidates([])
    assert res_empty["status"] == "insufficient_data"

    # Single candidate
    cand1 = {"candidate_name": "Alice", "skills": ["Python", "AWS"]}
    res_single = CandidateComparator.compare_candidates([cand1])
    assert res_single["status"] == "insufficient_data"


def test_compare_candidates_with_no_skills():
    cand1 = {"candidate_name": "Alice", "skills": []}
    cand2 = {"candidate_name": "Bob", "skills": {}}
    res = CandidateComparator.compare_candidates([cand1, cand2])

    assert res["status"] == "success"
    assert res["common_skills"] == []
    assert res["unique_skills"]["Alice"] == []
    assert res["unique_skills"]["Bob"] == []
    assert res["total_skills_universe_count"] == 0
    assert not res["skill_matrix_df"].empty or "Skill" in res["skill_matrix_df"].columns


def test_compare_candidates_overlapping_skills():
    cand1 = {
        "candidate_name": "Alice",
        "skills": {"Languages": ["Python", "SQL"], "Cloud": ["AWS"]}
    }
    cand2 = {
        "candidate_name": "Bob",
        "skills": {"Languages": ["Python", "Java"], "Cloud": ["AWS", "Docker"]}
    }
    res = CandidateComparator.compare_candidates([cand1, cand2])

    assert res["status"] == "success"
    assert sorted(res["common_skills"]) == ["AWS", "Python"]
    assert res["unique_skills"]["Alice"] == ["SQL"]
    assert sorted(res["unique_skills"]["Bob"]) == ["Docker", "Java"]
    assert res["total_skills_universe_count"] == 5


def test_compare_candidates_completely_distinct_skills():
    cand1 = {"candidate_name": "Alice", "skills": ["Python", "Django"]}
    cand2 = {"candidate_name": "Bob", "skills": ["C++", "Unreal"]}
    res = CandidateComparator.compare_candidates([cand1, cand2])

    assert res["status"] == "success"
    assert res["common_skills"] == []
    assert sorted(res["unique_skills"]["Alice"]) == ["Django", "Python"]
    assert sorted(res["unique_skills"]["Bob"]) == ["C++", "Unreal"]
    assert res["total_skills_universe_count"] == 4


def test_compare_three_candidates():
    cand1 = {"candidate_name": "Alice", "skills": ["Python", "Git"]}
    cand2 = {"candidate_name": "Bob", "skills": ["Python", "Docker"]}
    cand3 = {"candidate_name": "Charlie", "skills": ["Python", "AWS"]}
    res = CandidateComparator.compare_candidates([cand1, cand2, cand3])

    assert res["status"] == "success"
    assert res["common_skills"] == ["Python"]
    assert res["unique_skills"]["Alice"] == ["Git"]
    assert res["unique_skills"]["Bob"] == ["Docker"]
    assert res["unique_skills"]["Charlie"] == ["AWS"]

