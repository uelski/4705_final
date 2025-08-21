from model_training import train_model

def test_train_model_runs_with_nb(fitted_nb_pipeline, dummy_text_train, dummy_labels_train, dummy_text_test):
    y_pred = train_model(
        pipeline=fitted_nb_pipeline,
        X_train=dummy_text_train,
        y_train=dummy_labels_train,
        X_test=dummy_text_test,
        model_name="NB",
    )
    assert y_pred.shape == (len(dummy_text_test), dummy_labels_train.shape[1])

def test_train_model_runs_with_lr(fitted_lr_pipeline, dummy_text_train, dummy_labels_train, dummy_text_test):
    y_pred = train_model(
        pipeline=fitted_lr_pipeline,
        X_train=dummy_text_train,
        y_train=dummy_labels_train,
        X_test=dummy_text_test,
        model_name="LR",
    )
    assert y_pred.shape == (len(dummy_text_test), dummy_labels_train.shape[1])

def test_train_model_runs_with_svm(fitted_svm_pipeline, dummy_text_train, dummy_labels_train, dummy_text_test):
    y_pred = train_model(
        pipeline=fitted_svm_pipeline,
        X_train=dummy_text_train,
        y_train=dummy_labels_train,
        X_test=dummy_text_test,
        model_name="SVM",
    )
    assert y_pred.shape == (len(dummy_text_test), dummy_labels_train.shape[1])