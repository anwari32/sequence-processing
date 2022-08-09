def accuracy_and_error_rate(input_ids, prediction, target):
    if not (input_ids.shape == prediction.shape == target.shape):
        raise ValueError(f"All inputs must have 1 dimension. Found {input_ids.shape}, {prediction.shape}, {target.shape}")

    # Remove CLS token
    input_ids = input_ids[1:]
    prediction = prediction[1:]
    target = target[1:]

    # Remove special tokens.
    input_ids = [a for a in input_ids if a >= 0]
    input_ids_len = len(input_ids)
    prediction = prediction[0:input_ids_len]
    target = target[0:input_ids_len]

    accuracy = 0
    for i, j in zip(prediction, target):
        accuracy = accuracy + (1 if i == j else 0)
    accuracy = accuracy / input_ids_len
    return accuracy, (1 - accuracy)



