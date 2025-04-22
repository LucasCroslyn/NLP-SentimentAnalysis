from collections import Counter

def score(predictions, golds):
    '''
    Calculates a confusion matrix between predicted values and the true, actual values.
    Also calculates the per class and averaged macro Precision, Recall, F1 Scores and overall Accuracy.

    :param: predictions: A list of each prediction for all samples.
    :param: golds: A list of all true classes for the predicted samples.
    :return: Does not return the calculations but prints them out instead. 
    '''
    assert len(predictions) == len(golds)

    confusion_matrix = Counter()
    for p,g in zip(predictions, golds):
        confusion_matrix[(p,g)] += 1

    assert confusion_matrix.total() == len(predictions)

    # Simple way to make sure all unique types of classes are stored
    all_classes = set(predictions).union(golds)
    
    correct = 0
    macro_precision = 0
    macro_recall = 0
    macro_f1score = 0
    for clas in all_classes:
        num = confusion_matrix[(clas,clas)]
        correct += num

        precision_denom = sum([confusion_matrix[(clas,x)] for x in all_classes])
        if precision_denom == 0:
            print("WARNING: P undefined: Setting P to 0")
            precision = 0
        else:
            precision = num / float(precision_denom)

        recall_denom = sum([confusion_matrix[(x,clas)] for x in all_classes])
        if recall_denom == 0:
            print("WARNING: R undefined: Setting R to 0")
            recall = 0
        else:
            recall = num / float(recall_denom)
            
        f1score_denom = precision + recall
        if f1score_denom == 0:
            print("WARNING: F undefined: Setting F to 0")
            f1score = 0
        else:
            f1score = 2 * precision * recall / float(f1score_denom)

        macro_precision += precision
        macro_recall += recall
        macro_f1score += f1score

        print(f"Current class: {clas}")
        print(f"Precision: {round(precision, 3)}")
        print(f"Recall: {round(precision, 3)}")
        print(f"F1-Score: {round(precision, 3)}")
        print()

    accuracy = correct / float(confusion_matrix.total())
    avg_macro_precision = macro_precision / float(len(all_classes))
    avg_macro_recall = macro_recall / float(len(all_classes))
    avg_macro_f1score = macro_f1score / float(len(all_classes))
    print(f"Accuracy: {round(accuracy, 3)}")
    print(f"Macro averaged P: {round(avg_macro_precision, 3)}")
    print(f"Macro averaged R: {round(avg_macro_recall, 3)}")
    print(f"Macro averaged F: {round(avg_macro_f1score, 3)}")
