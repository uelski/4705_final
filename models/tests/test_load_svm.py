def test_svm_shapes(fitted_svm_pipeline, dummy_text_test, dummy_labels_train):
    # LinearSVC has decision_function instead of predict_proba
    preds = fitted_svm_pipeline.predict(dummy_text_test)
    assert preds.shape == (len(dummy_text_test), dummy_labels_train.shape[1])