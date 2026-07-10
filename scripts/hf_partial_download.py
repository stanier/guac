#!/usr/bin/env python3
"""
HuggingFace Partial Dataset Downloader
Downloads a portion of a HuggingFace dataset using streaming.
"""

import argparse
import sys
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import json
import yaml


def parse_size(size_str):
    """Parse size string like '1GB', '500MB', '100K' into number of bytes or samples."""
    size_str = size_str.strip().upper()
    
    # Check if it's a percentage
    if size_str.endswith('%'):
        try:
            percentage = float(size_str[:-1])
            if 0 < percentage <= 100:
                return ('percentage', percentage)
        except ValueError:
            pass
        raise ValueError(f"Invalid percentage: {size_str}")
    
    # Check if it's a count (just a number or number with K/M suffix)
    multipliers = {'K': 1000, 'M': 1_000_000, 'B': 1_000_000_000}
    
    for suffix, multiplier in multipliers.items():
        if size_str.endswith(suffix):
            try:
                num = float(size_str[:-1])
                return ('count', int(num * multiplier))
            except ValueError:
                pass
    
    # Try as plain number (sample count)
    try:
        return ('count', int(size_str))
    except ValueError:
        pass
    
    # Check if it's a byte size (GB, MB, KB)
    byte_multipliers = {'GB': 1024**3, 'MB': 1024**2, 'KB': 1024}
    for suffix, multiplier in byte_multipliers.items():
        if size_str.endswith(suffix):
            try:
                num = float(size_str[:-2])
                return ('bytes', int(num * multiplier))
            except ValueError:
                pass
    
    raise ValueError(f"Invalid size format: {size_str}. Use formats like: 10000, 100K, 1M, 10%, 500MB, 1GB")


def estimate_sample_size(sample):
    """Rough estimate of sample size in bytes."""
    return len(json.dumps(sample).encode('utf-8'))


def load_site_config(config_path='site_config.yml'):
    """Load site configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Site config file '{config_path}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing site config: {e}")
        sys.exit(1)


def get_dataset_config(site_config, dataset_name, dataset_type='pretraining_datasets'):
    """Get dataset configuration from site config."""
    try:
        datasets = site_config['dataset_configs'][dataset_type]
        if dataset_name not in datasets:
            available = list(datasets.keys())
            print(f"Error: Dataset '{dataset_name}' not found in {dataset_type}.")
            print(f"Available datasets: {', '.join(available)}")
            sys.exit(1)
        return datasets[dataset_name]
    except KeyError as e:
        print(f"Error: Invalid site config structure. Missing key: {e}")
        sys.exit(1)


def download_from_site_config(dataset_name, output_dir, site_config_path='site_config.yml',
                             dataset_type='pretraining_datasets', split='train', format='jsonl'):
    """Download dataset using configuration from site_config.yml."""
    print(f"Loading dataset configuration from {site_config_path}...")
    site_config = load_site_config(site_config_path)
    dataset_config = get_dataset_config(site_config, dataset_name, dataset_type)

    # Extract configuration
    repo_id = dataset_config['dataset_name']
    config_name = dataset_config.get('config_name')
    if config_name == 'default':
        config_name = None  # HuggingFace uses None for default config

    max_articles = dataset_config.get('max_articles')
    if max_articles is None:
        print(f"Warning: No max_articles specified for {dataset_name}, defaulting to 10000")
        max_articles = 10000

    print(f"Dataset: {dataset_name}")
    print(f"Repository: {repo_id}")
    print(f"Config: {config_name or 'default'}")
    print(f"Max articles: {max_articles}")

    # Convert max_articles to size specification
    size_spec = ('count', max_articles)

    # Download the dataset
    download_partial_dataset(
        repo_id=repo_id,
        split=split,
        size_spec=size_spec,
        output_dir=output_dir,
        config=config_name,
        format=format
    )


def download_partial_dataset(repo_id, split, size_spec, output_dir, config=None, format='jsonl'):
    """
    Download a partial dataset from HuggingFace.
    
    Args:
        repo_id: HuggingFace dataset repository ID (e.g., 'username/dataset-name')
        split: Dataset split to download (e.g., 'train', 'test')
        size_spec: Tuple of (type, value) where type is 'count', 'bytes', or 'percentage'
        output_dir: Directory to save the downloaded data
        config: Optional dataset configuration/subset name
        format: Output format ('jsonl', 'parquet', 'csv')
    """
    print(f"Loading dataset: {repo_id}")
    print(f"Split: {split}")
    if config:
        print(f"Config: {config}")
    
    # Load dataset in streaming mode
    try:
        dataset = load_dataset(
            repo_id,
            config,
            split=split,
            streaming=True,
            trust_remote_code=False
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    size_type, size_value = size_spec
    
    # Determine output file
    safe_repo_name = repo_id.replace('/', '_')
    config_suffix = f"_{config}" if config else ""
    output_file = output_path / f"{safe_repo_name}{config_suffix}_{split}_partial.{format}"
    
    print(f"Output file: {output_file}")
    print(f"Size limit: {size_type} = {size_value}")
    print("\nDownloading...")
    
    samples_downloaded = 0
    bytes_downloaded = 0
    samples_written = []
    
    try:
        # For percentage, we need to estimate total size (not always available in streaming)
        if size_type == 'percentage':
            print("Note: Percentage-based downloads estimate dataset size, which may be inaccurate for streaming datasets.")
            print("Consider using sample count instead for more predictable results.")
            # We'll convert percentage to a rough sample count estimate
            # This is approximate and may not work well for all datasets
            size_type = 'count'
            size_value = int(size_value * 10000)  # Rough estimate: 10000 samples per 1%
            print(f"Estimating ~{size_value} samples for {size_spec[1]}%")
        
        # Progress bar
        if size_type == 'count':
            pbar = tqdm(total=size_value, unit='samples')
        else:
            pbar = tqdm(total=size_value, unit='B', unit_scale=True)
        
        # Iterate and collect samples
        for sample in dataset:
            sample_size = estimate_sample_size(sample)
            
            samples_written.append(sample)
            samples_downloaded += 1
            bytes_downloaded += sample_size
            
            if size_type == 'count':
                pbar.update(1)
                if samples_downloaded >= size_value:
                    break
            elif size_type == 'bytes':
                pbar.update(sample_size)
                if bytes_downloaded >= size_value:
                    break
        
        pbar.close()
        
        # Write to file based on format
        print(f"\nWriting {len(samples_written)} samples to {output_file}...")
        
        if format == 'jsonl':
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in samples_written:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        elif format == 'parquet':
            from datasets import Dataset
            ds = Dataset.from_list(samples_written)
            ds.to_parquet(str(output_file))
        
        elif format == 'csv':
            from datasets import Dataset
            ds = Dataset.from_list(samples_written)
            ds.to_csv(str(output_file))
        
        print(f"\n✓ Successfully downloaded {samples_downloaded} samples ({bytes_downloaded / 1024 / 1024:.2f} MB)")
        print(f"✓ Saved to: {output_file}")
        
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        if samples_written:
            print(f"Partial data ({len(samples_written)} samples) can still be saved.")
            save = input("Save partial data? (y/n): ")
            if save.lower() == 'y':
                print(f"Writing {len(samples_written)} samples to {output_file}...")
                if format == 'jsonl':
                    with open(output_file, 'w', encoding='utf-8') as f:
                        for sample in samples_written:
                            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                print(f"✓ Saved partial data to: {output_file}")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n✗ Error during download: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Download a portion of a HuggingFace dataset using streaming',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download first 10,000 samples
  %(prog)s username/dataset-name --size 10000 --output ./data

  # Download first 100K samples from 'train' split
  %(prog)s username/dataset-name --size 100K --split train --output ./data

  # Download approximately 10%% of dataset
  %(prog)s username/dataset-name --size 10%% --output ./data

  # Download 500MB worth of data
  %(prog)s username/dataset-name --size 500MB --output ./data

  # Download with specific config/subset
  %(prog)s mozilla-foundation/common_voice_16_0 --config en --size 1000 --output ./data

  # Download and save as Parquet
  %(prog)s username/dataset-name --size 50K --output ./data --format parquet

  # Download from site config (replaces repo and size arguments)
  %(prog)s --from-site-config fineweb --output ./data

  # Download instruction dataset from site config
  %(prog)s --from-site-config openorca --dataset-type instruction_datasets --output ./data
        """
    )

    parser.add_argument(
        'repo',
        nargs='?',  # Make repo optional when using --from-site-config
        help='HuggingFace dataset repository ID (e.g., username/dataset-name)'
    )
    
    parser.add_argument(
        '--size',
        required=False,  # Not required when using --from-site-config
        help='Amount to download: number (e.g., 10000), K/M/B suffix (e.g., 100K, 1M), percentage (e.g., 10%%), or bytes (e.g., 500MB, 1GB)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='./downloaded_data',
        help='Output directory (default: ./downloaded_data)'
    )
    
    parser.add_argument(
        '--split',
        default='train',
        help='Dataset split to download (default: train)'
    )
    
    parser.add_argument(
        '--config',
        default=None,
        help='Dataset configuration/subset name (if applicable)'
    )
    
    parser.add_argument(
        '--format',
        choices=['jsonl', 'parquet', 'csv'],
        default='jsonl',
        help='Output format (default: jsonl)'
    )

    # Site config integration arguments
    parser.add_argument(
        '--from-site-config',
        metavar='DATASET_NAME',
        help='Download dataset using configuration from site_config.yml (e.g., fineweb, openorca)'
    )

    parser.add_argument(
        '--site-config-path',
        default='site_config.yml',
        help='Path to site configuration file (default: site_config.yml)'
    )

    parser.add_argument(
        '--dataset-type',
        choices=['pretraining_datasets', 'instruction_datasets'],
        default='pretraining_datasets',
        help='Type of dataset in site config (default: pretraining_datasets)'
    )
    
    args = parser.parse_args()

    # Check if using site config mode or traditional mode
    if args.from_site_config:
        # Site config mode
        download_from_site_config(
            dataset_name=args.from_site_config,
            output_dir=args.output,
            site_config_path=args.site_config_path,
            dataset_type=args.dataset_type,
            split=args.split,
            format=args.format
        )
    else:
        # Traditional mode - require repo and size
        if not args.repo:
            print("Error: Repository ID is required when not using --from-site-config")
            parser.print_help()
            sys.exit(1)

        if not args.size:
            print("Error: --size is required when not using --from-site-config")
            parser.print_help()
            sys.exit(1)

        # Parse size specification
        try:
            size_spec = parse_size(args.size)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

        # Download the dataset
        download_partial_dataset(
            repo_id=args.repo,
            split=args.split,
            size_spec=size_spec,
            output_dir=args.output,
            config=args.config,
            format=args.format
        )


if __name__ == '__main__':
    main()
