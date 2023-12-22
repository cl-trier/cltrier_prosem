from src.cltrier_prosem import Pipeline

def test_pipeline():
    pipeline = Pipeline({
        'encoder': {
            'model': 'prajjwal1/bert-tiny',
        },
        'dataset': {
            'path': './examples/data/dataset',
            'text_column': 'text',
            'label_column': 'label',
            'label_classes': ['corona', 'web1', 'web2'],
        },
        'classifier': {
            'hid_size': 64,
            'dropout': 0.2,
        },
        'pooler': {
            'form': 'subword_first',
            'span_column': 'span'
        },
        'trainer': {
            'num_epochs': 1,
            'batch_size': 2,
            'learning_rate': 1e-3,
            'export_path': './examples/results',
        }
    })

    pipeline()
