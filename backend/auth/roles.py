"""Role -> permission mapping (v6 Section 4.2 Recommended Roles table)."""

import uuid

SUBMIT_CASE = "submit_case"
VIEW_ASSIGNED_CASES = "view_assigned_cases"
VIEW_ALL_ORG_CASES = "view_all_org_cases"
VIEW_HIGH_RISK_CASES = "view_high_risk_cases"
SUBMIT_DECISION = "submit_decision"
OVERRIDE_DECISION = "override_decision"
ACCESS_REFERRAL_DATA = "access_referral_data"
RECORD_SAFEGUARDING_OUTCOME = "record_safeguarding_outcome"
MANAGE_ORG_MEMBERS = "manage_org_members"
READ_AUDIT_LOG = "read_audit_log"
MAINTAIN_PLATFORM = "maintain_platform"

# "Analyst | View assigned cases, submit review decisions."
# "Senior Reviewer | View high-risk cases, override analyst decisions."
# "Safeguarding Specialist | Access referral data, record safeguarding outcomes."
# "Organization Admin | Manage members and roles; cannot modify analysis results."
# "System Admin | Maintain platform infrastructure; no default access to case content."
# "Auditor | Read-only access to audit metadata; cannot modify cases."
ROLE_PERMISSIONS: dict[str, frozenset[str]] = {
    "analyst": frozenset({VIEW_ASSIGNED_CASES, SUBMIT_CASE, SUBMIT_DECISION}),
    "senior_reviewer": frozenset(
        {VIEW_ASSIGNED_CASES, VIEW_ALL_ORG_CASES, VIEW_HIGH_RISK_CASES, SUBMIT_CASE, SUBMIT_DECISION, OVERRIDE_DECISION}
    ),
    "safeguarding_specialist": frozenset(
        {VIEW_ASSIGNED_CASES, ACCESS_REFERRAL_DATA, RECORD_SAFEGUARDING_OUTCOME, SUBMIT_DECISION}
    ),
    "organization_admin": frozenset({MANAGE_ORG_MEMBERS}),
    "system_admin": frozenset({MAINTAIN_PLATFORM}),
    "auditor": frozenset({READ_AUDIT_LOG}),
}


def has_permission(role: str, permission: str) -> bool:
    return permission in ROLE_PERMISSIONS.get(role, frozenset())


def can_access_case(role: str, user_id: uuid.UUID, case, *, is_referred: bool = False) -> bool:
    """Section 4.1 steps 3-4: org ownership already checked by caller scoping
    the query to the user's organization_id; this covers case-assignment /
    authorized-scope.
    """
    if has_permission(role, VIEW_ALL_ORG_CASES):
        return True
    if has_permission(role, VIEW_ASSIGNED_CASES) and (
        case.assigned_analyst_id == user_id or case.submitted_by == user_id
    ):
        return True
    if has_permission(role, ACCESS_REFERRAL_DATA) and is_referred:
        return True
    return False