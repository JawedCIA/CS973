# CS973
## ASSIGMENT 1

# Method Descriptions

## my_fit Method Flowchart Description

1. Start
2. Initialize `xor_data_map` as an empty dictionary.
3. For each row in `Z_train`:
   - Calculate `x` and `y` from selection bits.
   - Compute `model_key = 16 * min(x, y) + max(x, y)`. [Each pair of XORRO selections (identified by x and y) needs a unique identifier for indexing into a collection of trained models. This allows the program to correctly access the model corresponding to any given selection of XORROs.
		Given that both x and y can take values from 0 to 15, the keys generated will range from 0 to 255 (i.e., 16×15+15), allowing for efficient mapping of configurations to model indices.
   - If `model_key` is not in `xor_data_map`:
     - Initialize `xor_data_map[model_key]` as an empty list.
   - Create a copy of the current row.
   - If `x > y`, flip the response (last element of the row).[a data preprocessing step aimed at maintaining consistent labeling of the output responses based on the selected XORROs. This helps prevent confusion during model training by ensuring that the model learns from consistently defined input-output mappings.]
   - Append the modified row to `xor_data_map[model_key]`.
4. Convert lists in `xor_data_map` to NumPy arrays.
5. Initialize `trained_models` as an empty dictionary.
6. For each combination of `i` and `j` (where `i < j` from 0 to 15):
   - Compute `model_key = 16 * i + j`.
   - If `model_key` is in `xor_data_map`:
     - Initialize a `LinearSVC` model.
     - Fit the model on the data associated with `model_key`.
     - Store the trained model in `trained_models` using `model_key`.
7. Return `trained_models`.
8. End

## my_predict Method Flowchart Description

1. Start
2. Initialize `X_pred` as a zero array of length equal to the number of rows in `X_tst`.
3. For each row in `X_tst`:
   - Calculate `x` and `y` from selection bits.
   - Compute `model_key = 16 * min(x, y) + max(x, y)`.
   - If `model_key` is not in `models`:
     - Continue to the next iteration.
   - Reshape the first 64 features of the current row for prediction.
   - Use the corresponding model to predict the response based on the reshaped features.
   - If `x > y`, flip the prediction.
   - Store the prediction in `X_pred`.
4. Return `X_pred`.
5. End



## Response Flipping Explanation

In the context of the XORRO PUF implementation, the indices `x` and `y` are derived from specific bits in the input data, representing the selected XORROs for processing. The response, located in the last element of the row, indicates the output of the PUF for the given challenge.
### Short explanation
The use of the formula `model_key = 16 * min(x, y) + max(x, y)` ensures that each combination of XORRO selections is uniquely and consistently identified, facilitating the efficient storage and retrieval of trained models in the implementation.
### Meaning of `x` and `y`
- **`x`**: Represents the index of the first selected XORRO.
- **`y`**: Represents the index of the second selected XORRO.

### Reason for Flipping
- The configuration and the corresponding response of the XORRO can be sensitive to the order of the XORRO indices.
- Flipping the response ensures consistent labeling. Specifically, when the index of the first selected XORRO (`x`) is greater than the second (`y`), it indicates an inversion of the intended configuration.

### Flipping Mechanism
- When the condition `x > y` is met, the last element of the row (the response) is modified as follows:
    ```python
    row_copy[-1] = 1 - row_copy[-1]
    ```
- This effectively changes a response of `0` to `1` and vice versa, ensuring that the model is trained on correctly labeled data.

## Explanation of `model_key = 16 * min(x, y) + max(x, y)`

The expression `model_key = 16 * min(x, y) + max(x, y)` is used to compute a unique identifier for each combination of selected XORROs in the XORRO PUF implementation. This calculation serves several important purposes:

### Unique Identification
- The `model_key` allows for the indexing of trained models based on the specific configuration of XORRO selections, ensuring that each configuration can be easily retrieved later.

### Consistent Ordering
- By using `min(x, y)` and `max(x, y)`, the order of the selected XORRO indices does not affect the resulting key. This ensures that the same pair of selections yields the same `model_key`, regardless of their order:
  - For instance, if `x = 3` and `y = 12`, both configurations `(3, 12)` and `(12, 3)` will result in the same key:
    ```plaintext
    model_key = 16 * min(3, 12) + max(3, 12) = 16 * 3 + 12 = 60
    ```

### Efficient Mapping
- The multiplication by `16` effectively groups keys based on the value of `x` (the first XORRO), while `y` (the second XORRO) extends the key within that grouping.
- Given that both `x` and `y` can take values from `0` to `15`, the keys generated will range from `0` to `255` (i.e., \(16 * 15 + 15\)), allowing for efficient mapping of configurations to model indices.



