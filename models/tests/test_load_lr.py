def test_lr_predict_shapes(fitted_lr_pipeline, dummy_text_test, dummy_labels_train):
    preds = fitted_lr_pipeline.predict(dummy_text_test)
    assert preds.shape == (len(dummy_text_test), dummy_labels_train.shape[1])