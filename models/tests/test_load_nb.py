def test_nb_predict_shapes(fitted_nb_pipeline, dummy_text_train, dummy_labels_train):
    preds = fitted_nb_pipeline.predict(dummy_text_train)
    assert preds.shape == (len(dummy_text_train), dummy_labels_train.shape[1])

