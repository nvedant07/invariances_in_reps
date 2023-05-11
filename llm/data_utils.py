from datasets import Dataset
import numpy as np

DATASET_VERSIONS = {
    'databricks--databricks-dolly-15k': 'b122d1b59226377d51cb23bc16e9769e8ede491f'
}


PARTIAL_CATEGORIES = {
    'databricks--databricks-dolly-15k': {
        'qa_only': ['open_qa', 'closed_qa', 'general_qa'],
        'context_only': ['closed_qa', 'information_extraction', 'summarization'],
        'creative_writing': ['creative_writing'],
        'classification': ['classification'],
        'information_extraction': ['information_extraction'],
        'brainstorming': ['brainstorming'],
        'summarization': ['summarization']
    }
}

def construct_filter(dataset: Dataset, partial: str):
    cats_to_include = PARTIAL_CATEGORIES[dataset.config_name]
    filter = np.array([False] * len(dataset))
    for cat in cats_to_include:
        filter = filter | (dataset.data['category'].to_numpy() == cat)
    return filter

def slice_dataset(dataset: Dataset, partial: str) -> Dataset:
    filter = construct_filter(dataset, partial)
    return Dataset(dataset.data.filter(filter), info=dataset.info)
