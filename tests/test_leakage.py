from ragcpt import leakage


def test_exact_normalized_match():
    gold = ["What is the minimum UE transmit power?"]
    train = ["what is the MINIMUM ue transmit power?  ", "unrelated question"]
    leaks = leakage.find_leaks(gold, train)
    assert len(leaks) == 1
    assert leaks[0]["kind"] == "exact"


def test_token_jaccard_catches_reorder():
    gold = ["minimum UE transmit power for QPSK above 15 RB"]
    train = ["for QPSK above 15 RB what is the minimum UE transmit power"]
    leaks = leakage.find_leaks(gold, train)
    assert len(leaks) == 1
    assert leaks[0]["kind"].startswith("jaccard")


def test_no_false_positive_on_distinct_questions():
    gold = ["What UICC form factors are supported?"]
    train = ["What is the periodic TAU timer value?", "How does paging work?"]
    assert leakage.find_leaks(gold, train) == []
