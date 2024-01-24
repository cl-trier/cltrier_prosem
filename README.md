# CLTrier ProSem

## Usage

```python
from cltrier_prosem import Pipeline

# init pipeline object (load model, data, trainer)
pipeline = Pipeline({
    'encoder': {
        'model': 'deepset/gbert-base',  # huggingface model slug 
    },
    'dataset': {
        'path': './path/data',  # path to data directory (containing train/test.parquet)
        'text_column': 'text',  # column containing src text
        'label_column': 'label',  # column containing target label
        'label_classes': ['class_1', 'class_2'],  # list of target classes
    },
    'classifier': {
        'hid_size': 512,  # size of classifier perceptron
        'dropout': 0.2,  # dropout value
    },
    'pooler': {
        'form': 'cls',
        # type of pooling, possible values: 
        # 'cls', 'sent_mean', 'subword_{first|last|mean|min|max}'
        # if subword probing used
        'span_columns': ['span']
    },
    'trainer': {
        'num_epochs': 5,  # number of training epochs
        'batch_size': 32,  # batch size in both training and evaluation
        'learning_rate': 1e-3,  # trainer learning rate
        'export_path': './path/output',  # output path for logging and results
    },
})

# call pipeline object (training and evaluation)
pipeline()
```
