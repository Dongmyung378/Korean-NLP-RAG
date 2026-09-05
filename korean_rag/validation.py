"""질문을 자르지 않고, 처리 가능한 길이인지 먼저 확인한다."""

MAX_QUESTION_CHARS = 1000
MAX_QUESTION_TOKENS = 128


class QuestionValidationError(ValueError):
    pass


def validate_question_text(question: str) -> str:
    if not isinstance(question, str) or not question.strip():
        raise QuestionValidationError("질문을 입력해 주세요.")
    question = question.strip()
    if len(question) > MAX_QUESTION_CHARS:
        raise QuestionValidationError(
            f"질문이 너무 깁니다. {MAX_QUESTION_CHARS}자 이내로 줄여 주세요."
        )
    return question


def validate_question_tokens(question: str, tokenizer) -> str:
    question = validate_question_text(question)
    token_ids = tokenizer(question, add_special_tokens=False, truncation=False)["input_ids"]
    if len(token_ids) > MAX_QUESTION_TOKENS:
        raise QuestionValidationError(
            f"질문이 너무 깁니다. 모델 기준 {MAX_QUESTION_TOKENS}토큰 이내로 줄여 주세요 "
            f"(현재 {len(token_ids)}토큰)."
        )
    return question
