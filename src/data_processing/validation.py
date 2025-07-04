import json

def validate_squad_format(dataset):
    """
    Validates if a dataset follows the SQuAD format.

    Args:
        dataset (dict): The dataset to validate.

    Returns:
        bool: True if the dataset is valid, False otherwise.
        list: A list of errors found in the dataset.
    """
    errors = []

    # Check if "data" is present and a list
    if "data" not in dataset or not isinstance(dataset["data"], list):
        errors.append("Missing or invalid 'data' key (should be a list).")
        return False, errors  # Stop early if top-level structure is wrong

    for order in dataset["data"]:
        # Check if each order has "title" and "paragraphs"
        if "title" not in order or not isinstance(order["title"], str):
            errors.append(f"Order missing 'title' or it's not a string: {order}")
        if "paragraphs" not in order or not isinstance(order["paragraphs"], list):
            errors.append(f"Order missing 'paragraphs' or it's not a list: {order}")

        for paragraph in order.get("paragraphs", []):
            # Check if paragraph has "context" (string) and "qas" (list)
            if "context" not in paragraph or not isinstance(paragraph["context"], str):
                errors.append(f"Paragraph missing 'context' or it's not a string: {paragraph}")
            if "qas" not in paragraph or not isinstance(paragraph["qas"], list):
                errors.append(f"Paragraph missing 'qas' or it's not a list: {paragraph}")

            for qa in paragraph.get("qas", []):
                # Check required fields in Q&A
                if "id" not in qa or not isinstance(qa["id"], int):
                    errors.append(f"QA missing 'id' or it's not an integer: {qa}")
                if "question" not in qa or not isinstance(qa["question"], str):
                    errors.append(f"QA missing 'question' or it's not a string: {qa}")
                if "lang" not in qa or not isinstance(qa["lang"], str):
                    errors.append(f"QA missing 'lang' or it's not a string: {qa}")
                if "answers" not in qa or not isinstance(qa["answers"], list):
                    errors.append(f"QA missing 'answers' or it's not a list: {qa}")

                for answer in qa.get("answers", []):
                    # Check if "text" and "answer_type" are present
                    if "text" not in answer or not isinstance(answer["text"], str):
                        errors.append(f"Answer missing 'text' or it's not a string: {answer}")
                    if "answer_type" not in answer or answer["answer_type"] not in {"string", "boolean"}:
                        errors.append(f"Invalid 'answer_type' (should be 'string' or 'boolean'): {answer}")

                    # Ensure "answer_start" exists only for "string" answers
                    if answer["answer_type"] == "string":
                        if "answer_start" not in answer or not isinstance(answer["answer_start"], int):
                            errors.append(f"String answer missing 'answer_start' or it's not an int: {answer}")
                    else:
                        if "answer_start" in answer:
                            errors.append(f"Boolean answer should NOT have 'answer_start': {answer}")

    return len(errors) == 0, errors  # Return True if no errors, otherwise False with errors
