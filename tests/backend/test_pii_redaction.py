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


def test_redacts_bare_phone_number_with_no_separators():
    # Exactly the format someone fires off mid-chat with no punctuation --
    # previously matched none of the four patterns and leaked unredacted.
    result = redact_text("text me 5551234567 ok")
    assert "[REDACTED_PHONE]" in result.redacted_text
    assert "5551234567" not in result.redacted_text
    assert "PHONE" in result.categories_found


def test_redacts_bare_phone_number_with_country_code():
    result = redact_text("its 15551234567 call anytime")
    assert "[REDACTED_PHONE]" in result.redacted_text
    assert "15551234567" not in result.redacted_text
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


def test_redacts_street_address():
    result = redact_text("I know you live at 1036 S Jackson St, Seattle, WA 98104.")
    assert "[REDACTED_ADDRESS]" in result.redacted_text
    assert "1036" not in result.redacted_text
    assert "Jackson" not in result.redacted_text
    assert "ADDRESS" in result.categories_found


def test_redacts_person_name_via_ner():
    # The exact live case that surfaced this gap: a plain name with no
    # surrounding URL/email/phone/address structure for regex to latch onto.
    result = redact_text("No. What do you want, John?")
    assert "[REDACTED_PERSON]" in result.redacted_text
    assert "John" not in result.redacted_text
    assert "PERSON" in result.categories_found


def test_ner_does_not_corrupt_an_already_redacted_email_marker():
    # Regression test: NER previously ran on the regex-substituted text,
    # which let it tag a fragment of "[REDACTED_EMAIL]" itself as a
    # false-positive entity, producing a nested, corrupted marker. NER must
    # only ever see the original text.
    result = redact_text("email me at a@b.com")
    assert result.redacted_text == "email me at [REDACTED_EMAIL]"
    assert result.categories_found == ("EMAIL",)