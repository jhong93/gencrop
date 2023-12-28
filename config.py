import os.path as path

# Where are the Unsplash images?
UNSPLASH_IMAGE_DIR = '/some/valid/path'

BENCHMARK_DATA_DIR = '/some/valid/path'

# Point these to the appropriate directories
class BenchmarkConfig:
    GAICD_DIR = path.join(BENCHMARK_DATA_DIR, 'GAICD')
    CPC_DIR = path.join(BENCHMARK_DATA_DIR, 'CPCDataset')
    FCDB_DIR = path.join(BENCHMARK_DATA_DIR, 'FCDB')
    FLMS_DIR = path.join(BENCHMARK_DATA_DIR, 'FLMS')
    SACD_DIR = path.join(BENCHMARK_DATA_DIR, 'SACD')