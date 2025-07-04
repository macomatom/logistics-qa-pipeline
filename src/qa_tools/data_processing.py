def count_answer_types_SQuAD(data):
    """Counts the occurrences of each answer type in the dataset."""
    type_counts = {"string": 0, "number": 0, "boolean": 0}

    for order in data:
        for paragraph in order.get("paragraphs", []):
            for question in paragraph.get("qas", []):
                for answer in question.get("answers", []):
                    answer_type = answer.get("answer_type")
                    if answer_type in type_counts:
                        type_counts[answer_type] += 1

    return type_counts


def count_answer_types_qas(data, langs):
    """Counts the occurrences of each answer type in the dataset."""
    type_counts = {"string": 0, "number": 0, "boolean": 0}

    for order in data:
        for lang in langs:
            if lang not in order.get("qas", []):
                continue
                
            for qa in order["qas"][lang]:
                answer_type = qa.get("answer_type")
                if answer_type in type_counts:
                    type_counts[answer_type] += 1

    return type_counts


def is_yes_no_question(question):
    """Determines if a question expects a yes/no answer based on common auxiliary verbs."""
    yes_no_keywords = {
        "is", "are", "do", "does", "did", "can", "could",
        "will", "would", "should", "has", "have", "had"
    }
    return question.lower().split()[0] in yes_no_keywords if question else False


def filter_question_types(data, exclude_types):
    """Removes answers of excluded types from the dataset."""
    for order in data.get("data", []):
        for paragraph in order.get("paragraphs", []):
            for question in paragraph.get("qas", []):
                question["answers"] = [
                    answer for answer in question.get("answers", [])
                    if answer.get("answer_type") not in exclude_types
                ]
    return data