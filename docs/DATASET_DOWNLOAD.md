# Dataset Download System

This document describes the CSV-driven dataset download system for Guac training, which allows batch downloading of HuggingFace datasets using configurable CSV files.

## Overview

The dataset download system uses the `hf_partial_download.py` script with CSV configuration files to download multiple datasets in batch. Downloads are automatically organized under the configured `SCRATCH_DIR`.

## Quick Start

1. **Configure datasets** in `datasets_download.csv`:
   ```csv
   # Format: repo,size,split,config,format
   HuggingFaceFW/fineweb,100MB,train,,jsonl
   HuggingFaceFW/fineweb-edu,100MB,train,,jsonl
   wikimedia/wikipedia,100MB,train,20231101.en,jsonl
   ```

2. **Download datasets**:
   ```bash
   make download-datasets
   ```

3. **Downloaded files** will be in `scratch/datasets/downloaded/`

4. **Use them** by path: `pretrain.py --datasets scratch/datasets/downloaded/<name>`
   (a `.jsonl`/`.json`/`.parquet`/`.csv`/`.txt` file, or a directory of them,
   with a `text` column), and the same in tokenizer training and vocab specs.

## CSV Configuration Format

The CSV format is: `repo,size,split,config,format`

### Fields

- **`repo`** (required): HuggingFace dataset repository ID (e.g., `username/dataset-name`)
- **`size`** (required): Amount to download (see size formats below)
- **`split`** (optional): Dataset split, defaults to `train`
- **`config`** (optional): Dataset configuration/subset name
- **`format`** (optional): Output format, defaults to `jsonl`

### Size Formats

- **Sample counts**: `1000`, `50K`, `1M`, `1B`
- **Byte sizes**: `100MB`, `1GB`, `500KB`
- **Percentages**: `10%`, `25%`, `50%`

### CSV Examples

```csv
# Download 100MB of FineWeb
HuggingFaceFW/fineweb,100MB,train,,jsonl

# Download 50K samples from specific config
mozilla-foundation/common_voice_16_0,50K,train,en,parquet

# Download 10% of validation split
username/dataset-name,10%,validation,,jsonl

# Download with all parameters specified
HuggingFaceFW/fineweb-edu,1M,train,default,parquet
```

## Commands

### Main Commands

```bash
# Download from default CSV file
make download-datasets

# Download from custom CSV file
make download-datasets-from CSV=custom.csv

# Show help and examples
make download-help
```

### Configuration Variables

```bash
# CSV configuration file (default: datasets_download.csv)
make download-datasets DOWNLOAD_CSV=my_datasets.csv

# Output directory (default: scratch/datasets/downloaded)
make download-datasets DOWNLOAD_OUTPUT=/custom/path
```

## Output Organization

Downloaded datasets are saved to `scratch/datasets/downloaded/` with filenames like:
- `{repo_name}_{split}_partial.{format}`
- `HuggingFaceFW_fineweb_train_partial.jsonl`
- `mozilla-foundation_common_voice_16_0_train_partial.parquet`

Repository names with slashes (`/`) are converted to underscores (`_`) for filenames.

## Examples

### Basic Usage

```bash
# Download datasets defined in datasets_download.csv
make download-datasets
```

### Custom Configuration

```bash
# Download from a different CSV file
make download-datasets-from CSV=production.csv

# Download to a custom location
make download-datasets DOWNLOAD_OUTPUT=/data/training
```

### Sample CSV Files

**`datasets_download.csv`** (default, shipped with the repository):
```csv
# Lines starting with # are skipped
HuggingFaceFW/fineweb-edu,100MB,train,sample-10BT,jsonl
Salesforce/wikitext,100%,validation,wikitext-103-raw-v1,jsonl
tatsu-lab/alpaca,10K,train,,jsonl
```

**`quick_test.csv`** (for testing):
```csv
# Small samples for testing
imdb,100,train,,jsonl
rotten_tomatoes,50,test,,jsonl
```

**`production.csv`** (for production training):
```csv
# Production datasets - larger samples
HuggingFaceFW/fineweb,1GB,train,,parquet
HuggingFaceFW/fineweb-edu,1GB,train,,parquet
c4,500MB,train,en,parquet
openwebtext,1GB,train,,parquet
```

## Error Handling

- **Failed downloads** are reported with exit codes
- **Successful downloads** continue even if some fail
- **Comments** (lines starting with `#`) are ignored
- **Empty lines** are skipped
- **Missing CSV** files show helpful error messages

## Integration with Path Manager

Downloads automatically use the centralized path management system:
- Output directory: `{SCRATCH_DIR}/datasets/downloaded`
- Configurable via `DOWNLOAD_OUTPUT` variable
- Follows the established directory structure

## Supported Datasets

Any HuggingFace dataset that supports streaming can be downloaded:
- **Text datasets**: FineWeb, C4, OpenWebText, WikiText
- **Multilingual**: mC4, multilingual Wikipedia
- **Code**: The Stack, CodeParrot
- **Conversational**: OpenOrca, UltraChat
- **Domain-specific**: PubMed, ArXiv, Books

## Tips

1. **Start small**: Use sample counts (e.g., `1K`) for testing
2. **Use byte sizes**: For consistent download sizes across datasets
3. **Batch similar datasets**: Group by format and size for efficiency
4. **Check disk space**: Large datasets can use significant storage
5. **Monitor progress**: Downloads show progress bars and statistics

## Troubleshooting

### Common Issues

**CSV not found**:
```
❌ CSV file 'datasets_download.csv' not found!
```
Solution: Create the CSV file or specify correct path with `CSV=filename.csv`

**Download failures**:
```
❌ Failed to download repo-name (exit code: 1)
```
Solution: Check repository name, network connection, and HuggingFace availability

**Permission errors**:
```
Permission denied: /path/to/output
```
Solution: Ensure write permissions to output directory or change `DOWNLOAD_OUTPUT`

### Debug Tips

1. **Test with small samples**: Use `10` or `100` sample counts first
2. **Check repository exists**: Verify on HuggingFace Hub
3. **Try different splits**: Some datasets only have `train` or `validation`
4. **Check config names**: Use HuggingFace dataset viewer to find configs