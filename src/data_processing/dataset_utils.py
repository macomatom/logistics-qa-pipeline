import numpy as np

def reshuffle_questions(data):
    """
    Reshuffles questions in each language within the dataset.

    Args:
        data (dict): Dataset containing orders with paragraphs and Q&A pairs.

    Returns:
        dict: Dataset with shuffled Q&A pairs.
    """
    new_data = {"data": []}

    for item in data["data"]:
        order_dict = {
            "title": item["title"],
            "paragraphs": []
        }

        for paragraph in item["paragraphs"]:
            shuffled_qas = []
            indices = np.random.permutation(len(paragraph["qas"]))

            for idx in indices:
                shuffled_qas.append(paragraph["qas"][idx])  # Preserve structure

            order_dict["paragraphs"].append({
                "context": paragraph["context"],
                "qas": shuffled_qas
            })

        new_data["data"].append(order_dict)

    return new_data