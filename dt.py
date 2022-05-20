def entropy(bucket):
    """
    Calculates the entropy.
    :param bucket: A list of size num_classes. bucket[i] is the number of
    examples that belong to class i.
    :return: A float. Calculated entropy.
    """
    
    total_examples = 0
    for examplesByClass in bucket:
        total_examples += examplesByClass
    
    entropy = 0
    for examplesByClass in bucket:
        if(examplesByClass != 0):
            entropy += (-1)*(examplesByClass/total_examples)*(math.log((examplesByClass/total_examples),2))
        
    return entropy


def info_gain(parent_bucket, left_bucket, right_bucket):
    """
    Calculates the information gain. A bucket is a list of size num_classes.
    bucket[i] is the number of examples that belong to class i.
    :param parent_bucket: Bucket belonging to the parent node. It contains the
    number of examples that belong to each class before the split.
    :param left_bucket: Bucket belonging to the left child after the split.
    :param right_bucket: Bucket belonging to the right child after the split.
    :return: A float. Calculated information gain.
    """
    
    entropy_of_parent = entropy(parent_bucket)
    entropy_of_left = entropy(left_bucket)
    entropy_of_right = entropy(right_bucket)
    
    total_examples_left = 0
    total_examples_right = 0
    total_examples_parent = 0
    
    for examplesByClass in parent_bucket:
        total_examples_parent += examplesByClass
        
    for examplesByClass in left_bucket:
        total_examples_left += examplesByClass
        
    for examplesByClass in right_bucket:
        total_examples_right += examplesByClass
        
    information = ((total_examples_left/total_examples_parent)*entropy_of_left) + ((total_examples_right/total_examples_parent)*entropy_of_right)
        
    information_gain = entropy_of_parent - information
    
    return information_gain


def gini(bucket):
    """
    Calculates the gini index.
    :param bucket: A list of size num_classes. bucket[i] is the number of
    examples that belong to class i.
    :return: A float. Calculated gini index.
    """
    
    total_examples = 0
    for examplesByClass in bucket:
        total_examples += examplesByClass
        
    sum_of_probs_squared = 0
    for examplesByClass in bucket:
        sum_of_probs_squared += pow((examplesByClass/total_examples),2)
        
    gini = 1 - sum_of_probs_squared
    
    return gini


def avg_gini_index(left_bucket, right_bucket):
    """
    Calculates the average gini index. A bucket is a list of size num_classes.
    bucket[i] is the number of examples that belong to class i.
    :param left_bucket: Bucket belonging to the left child after the split.
    :param right_bucket: Bucket belonging to the right child after the split.
    :return: A float. Calculated average gini index.
    """
    
    total_examples_left = 0
    total_examples_right = 0
    total_examples = 0
        
    for examplesByClass in left_bucket:
        total_examples_left += examplesByClass
        
    for examplesByClass in right_bucket:
        total_examples_right += examplesByClass
        
    total_examples = total_examples_left + total_examples_right
    
    average_gini = ((total_examples_left/total_examples)*gini(left_bucket)) + ((total_examples_right/total_examples)*gini(right_bucket))
    
    return average_gini


def calculate_split_values(data, labels, num_classes, attr_index, heuristic_name):
    """
    For every possible values to split the data for the attribute indexed by
    attribute_index, it divides the data into buckets and calculates the values
    returned by the heuristic function named heuristic_name. The split values
    should be the average of the closest 2 values. For example, if the data has
    2.1 and 2.2 in it consecutively for the values of attribute index by attr_index,
    then one of the split values should be 2.15.
    :param data: An (N, M) shaped numpy array. N is the number of examples in the
    current node. M is the dimensionality of the data. It contains the values for
    every attribute for every example.
    :param labels: An (N, ) shaped numpy array. It contains the class values in
    it. For every value, 0 <= value < num_classes.
    :param num_classes: An integer. The number of classes in the dataset.
    :param attr_index: An integer. The index of the attribute that is going to
    be used for the splitting operation. This integer indexs the second dimension
    of the data numpy array.
    :param heuristic_name: The name of the heuristic function. It should either be
    'info_gain' of 'avg_gini_index' for this homework.
    :return: An (L, 2) shaped numpy array. L is the number of split values. The
    first column is the split values and the second column contains the calculated
    heuristic values for their splits.
    """
    
    to_be_used_data = np.empty(0)
    for i in range(np.shape(data)[0]):
        to_be_used_data = np.append(to_be_used_data, data[i][attr_index])
    
    indexes = np.argsort(to_be_used_data)
    to_be_used_data = to_be_used_data[indexes]
    labels = labels[indexes]
    split_vals = np.empty([0,2])
    
    if(heuristic_name == "info_gain"):
        parent_bucket = np.zeros(num_classes)
        
        for x in range(np.size(labels)):
            parent_bucket[labels[x]] += 1
        
        for i in range(np.size(to_be_used_data)-1):
            
            less_then_equals_to = np.zeros(num_classes)
            greater_then = np.zeros(num_classes)
            
            average = ((to_be_used_data[i] + to_be_used_data[i+1])/2)
            j=0
            for j in range(0,i+1):
                less_then_equals_to[labels[j]] += 1
            
            for j in range(i+1,np.size(to_be_used_data)):
                greater_then[labels[j]] += 1
                
            information_gain = info_gain(parent_bucket,less_then_equals_to,greater_then)
            split_vals = np.append(split_vals, [[average,information_gain]], axis=0)
            

    
    elif(heuristic_name == "avg_gini_index"):
        for i in range(np.size(to_be_used_data)-1):
            
            less_then_equals_to = np.zeros(num_classes)
            greater_then = np.zeros(num_classes)
            
            average = ((to_be_used_data[i] + to_be_used_data[i+1])/2)
            j=0
            for j in range(0,i+1):
                less_then_equals_to[labels[j]] += 1
            
            for j in range(i+1,np.size(to_be_used_data)):
                greater_then[labels[j]] += 1
                
            avg_gini = avg_gini_index(less_then_equals_to,greater_then)
            split_vals = np.append(split_vals, [[average,avg_gini]], axis=0)
            

    return split_vals


def chi_squared_test(left_bucket, right_bucket):
    """
    Calculates chi squared value and degree of freedom between the selected attribute
    and the class attribute. A bucket is a list of size num_classes. bucket[i] is the
    number of examples that belong to class i.
    :param left_bucket: Bucket belonging to the left child after the split.
    :param right_bucket: Bucket belonging to the right child after the split.
    :return: A float and and integer. Chi squared value and degree of freedom.
    """
    
    classes_to_decrease = 0
    for i in range (np.size(left_bucket)):
        if(left_bucket[i] == 0 and right_bucket[i] == 0):
             classes_to_decrease += 1
    
    df = (np.size(left_bucket) - 1 - classes_to_decrease)
    
    total_in_left = 0
    total_in_right = 0
    total = 0
    
    for val in left_bucket:
        total_in_left += val
        
    for val in right_bucket:
        total_in_right += val
        
    total = total_in_left + total_in_right
        
    class_totals = np.empty(0)
    for i in range (np.size(left_bucket)):
        class_totals = np.append(class_totals, left_bucket[i]+right_bucket[i])
        
    chi_squared = 0
    l=-1
    r=-1
    size = np.size(left_bucket) + np.size(right_bucket)
    for i in range(size):
        if(i < size//2):
            l += 1
            if(left_bucket[l] != 0 or right_bucket[l]!=0):
                expected = (total_in_left*class_totals[l])/total
                actual = left_bucket[l]
        else:
            r += 1
            if(left_bucket[r] != 0 or right_bucket[r]!=0):
                expected = (total_in_right*class_totals[r])/total
                actual = right_bucket[r]
                
        if(i < size//2 and class_totals[l] != 0):
            chi_squared += (pow((actual-expected),2)/expected)
        elif(i >= size//2 and class_totals[r] != 0):
            chi_squared += (pow((actual-expected),2)/expected)
        
    return chi_squared, df
