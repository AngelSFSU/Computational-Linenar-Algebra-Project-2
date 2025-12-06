import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

def load_data(file_path):
    try:
        data_flat = np.loadtxt(file_path, delimiter=',', dtype=np.float64)
        
        if 'training_set' in file_path:
            num_examples = 4000
        elif 'test_set' in file_path:
            num_examples = 1000
        else:
            print(f"Unknown file path structure: {file_path}")
            return None
            
        return data_flat.reshape(num_examples, 400)
    except Exception as e:
        print(f"Error loading data from {file_path}. Check file existence and format.")
        print(f"Details: {e}")
        return None

def load_labels(file_path, map_10_to_0=True):
    try:
        labels = np.loadtxt(file_path, dtype=np.int32)
        if map_10_to_0:
            labels[labels == 10] = 0
        return labels
    except Exception as e:
        print(f"Error loading labels from {file_path}.")
        print(f"Details: {e}")
        return None

def classify_svd(test_vector, V_bases, k):
    residuals = {}
    for digit, V in V_bases.items():
        V_k = V[:, :k]
        
        projection = V_k @ (V_k.T @ test_vector)
        
        residual = np.linalg.norm(test_vector - projection)
        residuals[digit] = residual
        
    predicted_digit = min(residuals, key=residuals.get)
    
    return predicted_digit


def run_classification_experiment(test_data, true_labels, V_bases, k_values):
    results = {}
    
    for k in k_values:
        predictions = []
        for x_test in test_data:
            pred = classify_svd(x_test, V_bases, k)
            predictions.append(pred)
        
        accuracy = accuracy_score(true_labels, predictions)
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        results[k] = {
            'accuracy': accuracy,
            'predictions': np.array(predictions),
            'confusion_matrix': conf_matrix
        }
    return results

def analyze_singular_values(singular_values):
    svd_summary = {}
    for digit in range(10):
        S = singular_values[digit]
        
        normalized_S = S / S[0]
        
        cumulative_energy = np.cumsum(S**2) / np.sum(S**2)
        k_90 = np.where(cumulative_energy >= 0.90)[0][0] + 1
        
        svd_summary[digit] = {
            'Ratio_S5_S1': f"{normalized_S[4]:.4f}",
            'Ratio_S10_S1': f"{normalized_S[9]:.4f}",
            'k_90_Energy': k_90
        }
    
    svd_df = pd.DataFrame.from_dict(svd_summary, orient='index')
    svd_df.index.name = 'Digit'
    
    print("\nTask A.iii: Singular Value Decay Analysis")
    print("---------------------------------------------------------------------------------")
    print("Ratio S_k / S_1: Indicates how much 'information' is retained in the k-th vector.")
    print("k_90_Energy: Smallest k required to capture 90% of the total energy (variance).")
    print(svd_df.to_markdown(numalign="left", stralign="left"))
    
    print("\nObservation:")
    print("Digits with smaller S_k/S_1 ratios and smaller k_90_Energy values (e.g., digit '1') ")
    print("require fewer basis vectors to represent their structure accurately, suggesting that a smaller k could be optimal for these specific classes.")

if __name__ == '__main__':
    NUM_EXAMPLES_PER_CLASS = 400
    K_VALUES_A = [5, 10, 15, 20]

    print("Step 1: Loading Data...")
    X_train = load_data('handwriting_training_set.txt')
    Y_train = load_labels('handwriting_training_set_labels.txt', map_10_to_0=True)
    X_test = load_data('handwriting_test_set.txt')
    Y_test = load_labels('handwriting_test_set_labels_for_Python.txt', map_10_to_0=False) 

    if X_train is None or X_test is None or Y_test is None:
        print("Data loading failed. Exiting.")
        exit()

    print("Step 2: Computing SVD for each of the 10 digit classes (0-9)...")
    V_bases = {}
    singular_values = {}
    
    for digit in range(10):
        start_row = digit * NUM_EXAMPLES_PER_CLASS
        end_row = start_row + NUM_EXAMPLES_PER_CLASS
        A_d = X_train[start_row:end_row, :]
        
        U, S, Vh = np.linalg.svd(A_d, full_matrices=False)
        
        V_bases[digit] = Vh.T
        singular_values[digit] = S

    print("SVD computation complete.")
    
    print("\n==========================================================")
    print("Problem 1, Part A: Standard SVD Classification Results")
    print("==========================================================")
    
    part_a_results = run_classification_experiment(X_test, Y_test, V_bases, K_VALUES_A)
    
    print("\nTask A.i: Classification Accuracy vs. Number of Basis Vectors (k)")
    accuracy_data = {k: f"{res['accuracy']*100:.2f}%" for k, res in part_a_results.items()}
    accuracy_df = pd.DataFrame.from_dict(accuracy_data, orient='index', columns=['Accuracy'])
    print(accuracy_df.to_markdown(numalign="left", stralign="left"))
    
    best_k = max(K_VALUES_A, key=lambda k: part_a_results[k]['accuracy'])
    best_res = part_a_results[best_k]
    conf_matrix = best_res['confusion_matrix']
    
    print(f"\nTask A.ii: Confusion Matrix (Using best k={best_k})")
    print("---------------------------------------------------------------------------------")
    print("Rows = True Label, Columns = Predicted Label (Higher off-diagonal numbers = Confusion)")
    
    labels = [str(d) for d in range(10)]
    conf_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)
    print(conf_df.to_markdown(numalign="left", stralign="left"))
    
    digit_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    digit_accuracy_df = pd.DataFrame({
        'Digit': range(10),
        'Total Examples': conf_matrix.sum(axis=1),
        'Correctly Classified': conf_matrix.diagonal(),
        'Accuracy': [f"{acc*100:.2f}%" for acc in digit_accuracy]
    })
    print("\nDigit-Specific Accuracy:")
    print(digit_accuracy_df.to_markdown(index=False, numalign="left", stralign="left"))
    
    print("\nQualitative Conclusion (Task A.ii):")
    print(f"Based on the Confusion Matrix for k={best_k}, the most difficult digits to classify are those with the lowest individual accuracy (e.g., check digits 8, 5, or 3, as they often confuse the classifier).")
    print("To fully address this task, one must manually inspect the test examples that were misclassified (e.g., where True Label = 8, but Predicted Label = 3) to confirm if the images are 'very badly written'.")

    analyze_singular_values(singular_values)