from docqa.text import count_tokens, stem, tokenize


def test_tokenize_lowercases_drops_stopwords_and_apostrophes():
    assert tokenize("What's the RATE limit?") == ["what", "rate", "limit"]
    assert tokenize("Doesn't work") == ["work"]


def test_stemmer_conflates_common_variants():
    assert stem("rotated") == stem("rotate") == "rotat"
    assert stem("logging") == stem("logs") == "log"
    assert stem("retries") == "retry"
    assert stem("policies") == "policy"


def test_stemmer_leaves_short_digit_and_special_tokens_alone():
    assert stem("sev1") == "sev1"
    assert stem("99") == "99"
    assert stem("api") == "api"
    assert stem("status") == "status"
    assert stem("access") == "access"


def test_tokenize_can_keep_stopwords_and_skip_stemming():
    assert tokenize("The logs", drop_stopwords=False, stem_tokens=False) == ["the", "logs"]


def test_count_tokens_is_whitespace_based():
    assert count_tokens("one  two\nthree\tfour") == 4
    assert count_tokens("") == 0
