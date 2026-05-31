"""Tests for the vanilla LiMIT word tokenizer."""

from med.unlimit.tokenizers.vanilla import STOP_WORDS, VanillaWordTokenizer


def test_vanilla_tokenizer_builds_compact_observed_word_vocab():
    tokenizer = VanillaWordTokenizer()
    corpus = tokenizer.tokenize_corpus_records(
        [
            {
                "_id": "d1",
                "title": "",
                "text": "Alice likes The Ming Dynasty, Soy Milk.",
            }
        ]
    )

    words = tokenizer._word_to_id

    assert corpus[0]["unknown_phrases"] == []
    assert corpus[0]["token_ids"] == [
        words["alice"],
        words["likes"],
        words["ming"],
        words["dynasty"],
        words["soy"],
        words["milk"],
    ]
    assert "the" not in words
    assert tokenizer.num_token_types() == len(words)


def test_vanilla_query_reuses_corpus_word_ids_and_removes_stop_words():
    tokenizer = VanillaWordTokenizer()
    tokenizer.tokenize_corpus_records(
        [
            {
                "_id": "d1",
                "title": "",
                "text": "Alice likes The Ming Dynasty, Soy Milk.",
            }
        ]
    )

    query = tokenizer.tokenize_query_records(
        [{"_id": "q1", "text": "Who likes Soy Milk?"}]
    )

    words = tokenizer._word_to_id
    assert "who" in STOP_WORDS
    assert query[0]["unknown_phrases"] == []
    assert query[0]["token_ids"] == [
        words["likes"],
        words["soy"],
        words["milk"],
    ]


def test_vanilla_tokenizer_keeps_words_not_phrases_or_subwords():
    tokenizer = VanillaWordTokenizer()
    encoded = tokenizer.tokenize_query_records(
        [{"_id": "q1", "text": "Who likes Lo-fi music, Rubik's Cubes?"}]
    )[0]["token_ids"]

    words = tokenizer._word_to_id
    assert encoded == [
        words["likes"],
        words["lo-fi"],
        words["music"],
        words["rubik's"],
        words["cubes"],
    ]
    assert "Lo-fi music" not in words
    assert "rubik" not in words
    assert "s" not in words
