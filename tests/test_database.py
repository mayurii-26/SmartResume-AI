"""
Unit tests for database management operations, focusing on resume deletion,
relational data cascading, storage file cleanups, and cross-candidate isolation.
"""

import pytest
import sqlite3
from pathlib import Path
from database.db_manager import DatabaseManager
from config.settings import STORAGE_DIR, SKILL_KB_PATH


@pytest.fixture
def temp_db(tmp_path):
    """Creates an isolated temporary database for testing."""
    test_db_path = tmp_path / "test_smartresume.db"
    manager = DatabaseManager(db_path=test_db_path)
    return manager


def test_delete_resume_isolation_and_file_removal(temp_db, tmp_path, monkeypatch):
    """Verifies that deleting a resume removes its database records and storage files,
    while leaving other candidates and shared knowledge base completely intact."""
    # Isolate storage dir
    test_storage = tmp_path / "storage"
    test_storage.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("database.db_manager.STORAGE_DIR", test_storage)

    # 1. Insert Candidate A (ID will be 1)
    res_a_id = temp_db.insert_resume(
        filename="alice_resume.pdf",
        file_type="PDF",
        original_text="Alice Developer Python AWS",
        file_size=1024,
        is_scanned=0,
        page_count=1,
        cleaned_text="alice developer python aws",
        tokens_count=4
    )
    temp_db.save_candidate_profile(res_a_id, {
        "candidate_name": "Alice Smith",
        "email": "alice@example.com",
        "phone": "+1234567890",
        "skills": {"Technical": ["Python", "AWS"]}
    })
    temp_db.save_entities(res_a_id, [
        {"text": "Python", "label": "SKILL", "start": 16, "end": 22, "confidence": 0.95}
    ])
    temp_db.save_entity_links(res_a_id, [
        {"extracted_term": "Python", "canonical_name": "Python", "category": "Languages", "match_method": "exact", "similarity_score": 1.0}
    ])

    # 2. Insert Candidate B (ID will be 2)
    res_b_id = temp_db.insert_resume(
        filename="bob_resume.pdf",
        file_type="PDF",
        original_text="Bob Engineer Java Docker",
        file_size=2048,
        is_scanned=0,
        page_count=2,
        cleaned_text="bob engineer java docker",
        tokens_count=4
    )
    temp_db.save_candidate_profile(res_b_id, {
        "candidate_name": "Bob Jones",
        "email": "bob@example.com",
        "phone": "+0987654321",
        "skills": {"Technical": ["Java", "Docker"]}
    })
    temp_db.save_entities(res_b_id, [
        {"text": "Java", "label": "SKILL", "start": 13, "end": 17, "confidence": 0.92}
    ])
    temp_db.save_entity_links(res_b_id, [
        {"extracted_term": "Java", "canonical_name": "Java", "category": "Languages", "match_method": "exact", "similarity_score": 1.0}
    ])

    # Create dummy storage files for both
    file_a = test_storage / f"{res_a_id}_alice_resume.pdf"
    file_a.write_bytes(b"%PDF-1.4 Alice dummy content")
    file_b = test_storage / f"{res_b_id}_bob_resume.pdf"
    file_b.write_bytes(b"%PDF-1.4 Bob dummy content")

    assert file_a.exists()
    assert file_b.exists()

    # 3. Delete Candidate A
    success = temp_db.delete_resume(res_a_id)
    assert success is True

    # 4. Verify Candidate A is completely deleted from storage
    assert not file_a.exists()

    # 5. Verify Candidate A is completely removed from all database tables
    assert temp_db.get_resume(res_a_id) is None
    assert temp_db.get_candidate_profile(res_a_id) in (None, {})
    assert len(temp_db.get_entities(res_a_id)) == 0
    assert len(temp_db.get_entity_links(res_a_id)) == 0

    # 6. Verify Candidate B is COMPLETELY INTACT
    assert file_b.exists()
    resume_b = temp_db.get_resume(res_b_id)
    assert resume_b is not None
    assert resume_b["filename"] == "bob_resume.pdf"
    profile_b = temp_db.get_candidate_profile(res_b_id)
    assert profile_b["candidate_name"] == "Bob Jones"
    entities_b = temp_db.get_entities(res_b_id)
    assert len(entities_b) == 1
    assert entities_b[0]["entity_text"] == "Java"
    links_b = temp_db.get_entity_links(res_b_id)
    assert len(links_b) == 1
    assert links_b[0]["canonical_name"] == "Java"


def test_delete_nonexistent_resume(temp_db):
    """Verifies that attempting to delete an invalid resume ID returns False safely."""
    result = temp_db.delete_resume(999999)
    assert result is False


def test_delete_resume_without_storage_file(temp_db):
    """Verifies that delete_resume succeeds when no physical storage file exists."""
    res_id = temp_db.insert_resume(
        filename="nofile_resume.pdf",
        file_type="PDF",
        original_text="No file on disk test",
        file_size=512,
        is_scanned=0,
        page_count=1,
        cleaned_text="no file on disk test",
        tokens_count=5
    )
    temp_db.save_candidate_profile(res_id, {"candidate_name": "Ghost Candidate"})

    success = temp_db.delete_resume(res_id)
    assert success is True
    assert temp_db.get_resume(res_id) is None
