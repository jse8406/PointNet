def count_classes(data):
    class_counts = {}
    for item in data:
        class_name = item[0]
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1
    return class_counts

