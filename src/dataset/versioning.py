"""
src/dataset/versioning.py

Dataset Versioning Utilities

Semantic versioning ve dataset snapshot yönetimi için utility fonksiyonlar.

Semantic Versioning Format: MAJOR.MINOR.PATCH
- MAJOR: Breaking changes (schema değişikliği)
- MINOR: Backward compatible features (yeni filtreler, yeni alanlar)
- PATCH: Bug fixes, re-compilation (aynı parametre, yeni data)
"""

import re
import logging
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class VersionBump(Enum):
    """Version bump tipi"""
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"


@dataclass
class SemanticVersion:
    """
    Semantic version representation.
    
    Attributes:
        major: Major version number
        minor: Minor version number
        patch: Patch version number
    """
    major: int
    minor: int
    patch: int
    
    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def __repr__(self) -> str:
        return f"SemanticVersion({self.major}.{self.minor}.{self.patch})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, SemanticVersion):
            return False
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)
    
    def __lt__(self, other) -> bool:
        if not isinstance(other, SemanticVersion):
            raise TypeError("Cannot compare SemanticVersion with non-SemanticVersion")
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
    
    def __gt__(self, other) -> bool:
        if not isinstance(other, SemanticVersion):
            raise TypeError("Cannot compare SemanticVersion with non-SemanticVersion")
        return (self.major, self.minor, self.patch) > (other.major, other.minor, other.patch)
    
    def bump(self, bump_type: VersionBump) -> "SemanticVersion":
        """
        Version'ı bump et.
        
        Args:
            bump_type: MAJOR, MINOR veya PATCH
            
        Returns:
            SemanticVersion: Yeni version
        """
        if bump_type == VersionBump.MAJOR:
            return SemanticVersion(self.major + 1, 0, 0)
        elif bump_type == VersionBump.MINOR:
            return SemanticVersion(self.major, self.minor + 1, 0)
        elif bump_type == VersionBump.PATCH:
            return SemanticVersion(self.major, self.minor, self.patch + 1)
        else:
            raise ValueError(f"Invalid bump type: {bump_type}")


def parse_version(version_str: str) -> SemanticVersion:
    """
    Version string'i parse et.
    
    Args:
        version_str: Version string (örn: "1.2.3")
        
    Returns:
        SemanticVersion: Parsed version
        
    Raises:
        ValueError: Geçersiz version format
        
    Examples:
        >>> parse_version("1.2.3")
        SemanticVersion(1.2.3)
        >>> parse_version("2.0.0")
        SemanticVersion(2.0.0)
    """
    pattern = r"^(\d+)\.(\d+)\.(\d+)$"
    match = re.match(pattern, version_str)
    
    if not match:
        raise ValueError(f"Invalid version format: {version_str}. Expected: MAJOR.MINOR.PATCH")
    
    major, minor, patch = match.groups()
    return SemanticVersion(int(major), int(minor), int(patch))


def increment_version(
    current_version: str,
    bump_type: VersionBump = VersionBump.PATCH
) -> str:
    """
    Version'ı increment et.
    
    Args:
        current_version: Mevcut version string
        bump_type: Bump tipi (MAJOR, MINOR, PATCH)
        
    Returns:
        str: Yeni version string
        
    Examples:
        >>> increment_version("1.2.3", VersionBump.PATCH)
        "1.2.4"
        >>> increment_version("1.2.3", VersionBump.MINOR)
        "1.3.0"
        >>> increment_version("1.2.3", VersionBump.MAJOR)
        "2.0.0"
    """
    version = parse_version(current_version)
    new_version = version.bump(bump_type)
    return str(new_version)


def get_latest_version(versions: list[str]) -> Optional[str]:
    """
    Version listesinden en yüksek version'ı bul.
    
    Args:
        versions: Version string listesi
        
    Returns:
        str: En yüksek version, boş liste ise None
        
    Examples:
        >>> get_latest_version(["1.0.0", "1.2.3", "1.1.0"])
        "1.2.3"
        >>> get_latest_version(["2.0.0", "1.9.9"])
        "2.0.0"
    """
    if not versions:
        return None
    
    parsed_versions = [parse_version(v) for v in versions]
    latest = max(parsed_versions)
    return str(latest)


def suggest_next_version(
    current_version: str,
    schema_changed: bool = False,
    params_changed: bool = False
) -> Tuple[str, VersionBump]:
    """
    Değişikliklere göre sonraki version'ı öner.
    
    Kurallar:
    - Schema değişti: MAJOR bump
    - Parametre değişti: MINOR bump
    - Sadece data değişti: PATCH bump
    
    Args:
        current_version: Mevcut version
        schema_changed: Schema değişti mi?
        params_changed: Compilation parametreleri değişti mi?
        
    Returns:
        Tuple[str, VersionBump]: (Yeni version, Bump tipi)
        
    Examples:
        >>> suggest_next_version("1.2.3", schema_changed=True)
        ("2.0.0", VersionBump.MAJOR)
        >>> suggest_next_version("1.2.3", params_changed=True)
        ("1.3.0", VersionBump.MINOR)
        >>> suggest_next_version("1.2.3")
        ("1.2.4", VersionBump.PATCH)
    """
    if schema_changed:
        bump_type = VersionBump.MAJOR
    elif params_changed:
        bump_type = VersionBump.MINOR
    else:
        bump_type = VersionBump.PATCH
    
    new_version = increment_version(current_version, bump_type)
    return new_version, bump_type


def is_version_compatible(
    dataset_version: str,
    required_version: str
) -> bool:
    """
    Dataset version'ı required version ile compatible mi kontrol et.
    
    Semantic versioning kuralları:
    - MAJOR version aynı olmalı (breaking changes)
    - MINOR version >= olmalı (backward compatible)
    
    Args:
        dataset_version: Dataset'in version'ı
        required_version: İstenen minimum version
        
    Returns:
        bool: Compatible ise True
        
    Examples:
        >>> is_version_compatible("1.3.0", "1.2.0")
        True
        >>> is_version_compatible("1.1.0", "1.2.0")
        False
        >>> is_version_compatible("2.0.0", "1.9.9")
        False
    """
    ds_version = parse_version(dataset_version)
    req_version = parse_version(required_version)
    
    # MAJOR version aynı olmalı
    if ds_version.major != req_version.major:
        return False
    
    # Dataset version >= required version
    return ds_version >= req_version


def generate_snapshot_name(base_name: str, version: str) -> str:
    """
    Snapshot için isim oluştur.
    
    Format: {base_name}_v{version}
    
    Args:
        base_name: Dataset base ismi
        version: Version string
        
    Returns:
        str: Snapshot ismi
        
    Examples:
        >>> generate_snapshot_name("training_data", "1.2.3")
        "training_data_v1.2.3"
    """
    # Dosya ismi için safe string
    safe_version = version.replace(".", "_")
    return f"{base_name}_v{safe_version}"
