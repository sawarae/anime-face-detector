#!/usr/bin/env python3
"""
Image crawler for collecting anime character images for training.
Uses gallery-dl to download images from various sources.
"""

import argparse
import json
import subprocess
from pathlib import Path


def crawl_safebooru(tags: str, limit: int, output_dir: Path):
    """Crawl images from Safebooru."""
    print(f'Crawling Safebooru with tags: {tags}, limit: {limit}')

    url = f'https://safebooru.org/index.php?page=post&s=list&tags={tags}'

    cmd = [
        'gallery-dl',
        '--range', f'1-{limit}',
        '--dest', str(output_dir),
        '--download-archive', str(output_dir / 'archive.txt'),
        url
    ]

    subprocess.run(cmd, check=True)


def crawl_danbooru(tags: str, limit: int, output_dir: Path, api_key: str = None):
    """Crawl images from Danbooru."""
    print(f'Crawling Danbooru with tags: {tags}, limit: {limit}')

    url = f'https://danbooru.donmai.us/posts?tags={tags}'

    cmd = [
        'gallery-dl',
        '--range', f'1-{limit}',
        '--dest', str(output_dir),
        '--download-archive', str(output_dir / 'archive.txt'),
    ]

    if api_key:
        cmd.extend(['--option', f'api-key={api_key}'])

    cmd.append(url)

    subprocess.run(cmd, check=True)


def crawl_gelbooru(tags: str, limit: int, output_dir: Path):
    """Crawl images from Gelbooru."""
    print(f'Crawling Gelbooru with tags: {tags}, limit: {limit}')

    url = f'https://gelbooru.com/index.php?page=post&s=list&tags={tags}'

    cmd = [
        'gallery-dl',
        '--range', f'1-{limit}',
        '--dest', str(output_dir),
        '--download-archive', str(output_dir / 'archive.txt'),
        url
    ]

    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description='Crawl anime character images from various sources'
    )
    parser.add_argument(
        '--source',
        choices=['safebooru', 'danbooru', 'gelbooru'],
        default='safebooru',
        help='Image source to crawl from'
    )
    parser.add_argument(
        '--tags',
        type=str,
        required=True,
        help='Tags to search for (space-separated, will be URL-encoded)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=100,
        help='Maximum number of images to download'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/raw'),
        help='Output directory for downloaded images'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='API key for sources that require authentication (e.g., Danbooru)'
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # URL-encode tags
    tags_encoded = args.tags.replace(' ', '+')

    # Crawl based on source
    if args.source == 'safebooru':
        crawl_safebooru(tags_encoded, args.limit, args.output_dir)
    elif args.source == 'danbooru':
        crawl_danbooru(tags_encoded, args.limit, args.output_dir, args.api_key)
    elif args.source == 'gelbooru':
        crawl_gelbooru(tags_encoded, args.limit, args.output_dir)

    print(f'Download complete! Images saved to {args.output_dir}')


if __name__ == '__main__':
    main()
