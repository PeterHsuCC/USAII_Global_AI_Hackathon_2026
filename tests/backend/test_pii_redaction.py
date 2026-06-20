from backend.preprocessing.pii_redaction import redact_text


def test_redacts_email():
    result = redact_text("reach me at jane.doe@example.com please")
    assert "[REDACTED_EMAIL]" in result.redacted_text
    assert "jane.doe@example.com" not in result.redacted_text
    assert "EMAIL" in result.categories_found


def test_redacts_url():
    result = redact_text("check this out https://example.com/path?x=1")
    assert "[REDACTED_URL]" in result.redacted_text
    assert "URL" in result.categories_found


def test_redacts_phone_number():
    result = redact_text("call me at 555-123-4567 tonight")
    assert "[REDACTED_PHONE]" in result.redacted_text
    assert "PHONE" in result.categories_found


def test_redacts_handle():
    result = redact_text("add me on @cool_user99 later")
    assert "[REDACTED_HANDLE]" in result.redacted_text
    assert "HANDLE" in result.categories_found


def test_email_not_double_matched_as_handle():
    result = redact_text("jane.doe@example.com")
    assert result.redacted_text == "[REDACTED_EMAIL]"
    assert result.categories_found == ("EMAIL",)


def test_clean_text_untouched():
    result = redact_text("hello, how are you today?")
    assert result.redacted_text == "hello, how are you today?"
    assert result.categories_found == ()