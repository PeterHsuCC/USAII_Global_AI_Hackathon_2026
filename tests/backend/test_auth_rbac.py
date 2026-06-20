import uuid

import pytest

from backend.auth.roles import (
    ACCESS_REFERRAL_DATA,
    SUBMIT_CASE,
    VIEW_ALL_ORG_CASES,
    can_access_case,
    has_permission,
)
from backend.auth.security import (
    TokenError,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)


def test_password_hash_roundtrip():
    hashed = hash_password("ChangeMe123!")
    assert hashed != "ChangeMe123!"
    assert verify_password("ChangeMe123!", hashed)
    assert not verify_password("wrong-password", hashed)


def test_access_token_roundtrip():
    user_id, org_id = uuid.uuid4(), uuid.uuid4()
    token = create_access_token(user_id, org_id, "analyst")
    decoded = decode_token(token, expected_type="access")
    assert decoded.user_id == user_id
    assert decoded.organization_id == org_id
    assert decoded.role == "analyst"


def test_refresh_token_rejected_as_access_token():
    user_id, org_id = uuid.uuid4(), uuid.uuid4()
    refresh = create_refresh_token(user_id, org_id, "analyst")
    with pytest.raises(TokenError):
        decode_token(refresh, expected_type="access")


def test_garbage_token_rejected():
    with pytest.raises(TokenError):
        decode_token("not-a-real-token", expected_type="access")


def test_role_permissions_match_section_4_2():
    assert has_permission("analyst", SUBMIT_CASE)
    assert not has_permission("analyst", VIEW_ALL_ORG_CASES)

    assert has_permission("senior_reviewer", VIEW_ALL_ORG_CASES)

    assert has_permission("safeguarding_specialist", ACCESS_REFERRAL_DATA)
    assert not has_permission("safeguarding_specialist", VIEW_ALL_ORG_CASES)

    assert not has_permission("organization_admin", VIEW_ALL_ORG_CASES)
    assert not has_permission("system_admin", VIEW_ALL_ORG_CASES)
    assert not has_permission("auditor", SUBMIT_CASE)


class _FakeCase:
    def __init__(self, assigned_analyst_id=None, submitted_by=None) -> None:
        self.assigned_analyst_id = assigned_analyst_id
        self.submitted_by = submitted_by


def test_analyst_can_access_only_own_assigned_or_submitted_case():
    analyst_id = uuid.uuid4()
    other_id = uuid.uuid4()

    assigned_case = _FakeCase(assigned_analyst_id=analyst_id, submitted_by=other_id)
    submitted_case = _FakeCase(assigned_analyst_id=other_id, submitted_by=analyst_id)
    unrelated_case = _FakeCase(assigned_analyst_id=other_id, submitted_by=other_id)

    assert can_access_case("analyst", analyst_id, assigned_case)
    assert can_access_case("analyst", analyst_id, submitted_case)
    assert not can_access_case("analyst", analyst_id, unrelated_case)


def test_senior_reviewer_can_access_any_case_in_scope():
    reviewer_id = uuid.uuid4()
    unrelated_case = _FakeCase(assigned_analyst_id=uuid.uuid4(), submitted_by=uuid.uuid4())
    assert can_access_case("senior_reviewer", reviewer_id, unrelated_case)


def test_safeguarding_specialist_needs_referral_flag():
    specialist_id = uuid.uuid4()
    unrelated_case = _FakeCase(assigned_analyst_id=uuid.uuid4(), submitted_by=uuid.uuid4())
    assert not can_access_case("safeguarding_specialist", specialist_id, unrelated_case, is_referred=False)
    assert can_access_case("safeguarding_specialist", specialist_id, unrelated_case, is_referred=True)