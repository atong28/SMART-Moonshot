#!/usr/bin/env python3
"""
Check retrieval.pkl for invalid SMILES entries and indexing gaps.
"""
import os
import sys
import pickle
import json
import argparse
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict

def _load_as_dict(path: str) -> Any:
    """Load .pkl or .json file."""
    with open(path, "rb") as f:
        if path.endswith(".pkl") or path.endswith(".pickle"):
            return pickle.load(f)
    with open(path, "r") as f:
        return json.load(f)


def _extract_smiles(rec: Any) -> Optional[str]:
    """Extract SMILES from a record (same logic as fp_utils)."""
    if isinstance(rec, str):
        return rec
    if isinstance(rec, dict):
        for key in ("smiles", "canonical_2d_smiles"):
            v = rec.get(key)
            if isinstance(v, str) and v:
                return v
    return None


def check_retrieval_index(retrieval_path: str):
    """Check retrieval.pkl for invalid SMILES and indexing issues."""
    print(f"Loading {retrieval_path}...")
    data = _load_as_dict(retrieval_path)
    
    invalid_indices: List[int] = []
    valid_indices: List[int] = []
    all_indices: Set[int] = set()
    
    if isinstance(data, list):
        print(f"Detected list format with {len(data)} entries")
        for idx, rec in enumerate(data):
            s = _extract_smiles(rec)
            if s:
                valid_indices.append(idx)
            else:
                invalid_indices.append(idx)
            all_indices.add(idx)
    elif isinstance(data, dict):
        print(f"Detected dict format with {len(data)} entries")
        for k, v in data.items():
            try:
                idx = int(k)
            except (ValueError, TypeError):
                print(f"  Warning: Non-integer key '{k}' (type: {type(k).__name__})")
                continue
            
            all_indices.add(idx)
            if isinstance(v, str):
                s = v
            else:
                s = _extract_smiles(v)
            
            if s:
                valid_indices.append(idx)
            else:
                invalid_indices.append(idx)
    else:
        print(f"Error: Unsupported data type: {type(data)}")
        return
    
    # Find gaps in indexing
    if all_indices:
        min_idx = min(all_indices)
        max_idx = max(all_indices)
        expected_count = max_idx - min_idx + 1
        missing_indices = [i for i in range(min_idx, max_idx + 1) if i not in all_indices]
    else:
        min_idx = max_idx = 0
        expected_count = 0
        missing_indices = []
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total entries in file: {len(data)}")
    print(f"Valid SMILES entries: {len(valid_indices)}")
    print(f"Invalid/missing SMILES entries: {len(invalid_indices)}")
    print(f"\nIndex range: {min_idx} to {max_idx}")
    print(f"Expected consecutive indices: {expected_count}")
    print(f"Missing indices (gaps): {len(missing_indices)}")
    
    if invalid_indices:
        print(f"\nInvalid SMILES at indices ({len(invalid_indices)} total):")
        if len(invalid_indices) <= 50:
            print(f"  {invalid_indices}")
        else:
            print(f"  First 20: {invalid_indices[:20]}")
            print(f"  Last 20: {invalid_indices[-20:]}")
            print(f"  ... and {len(invalid_indices) - 40} more")
    
    if missing_indices:
        print(f"\nMissing indices (gaps in indexing) ({len(missing_indices)} total):")
        if len(missing_indices) <= 50:
            print(f"  {missing_indices}")
        else:
            print(f"  First 20: {missing_indices[:20]}")
            print(f"  Last 20: {missing_indices[-20:]}")
            print(f"  ... and {len(missing_indices) - 40} more")
    
    # Check for the specific index mentioned
    if 333536 in all_indices:
        status = "VALID" if 333536 in valid_indices else "INVALID"
        print(f"\nIndex 333536: {status}")
        if 333536 in all_indices:
            if isinstance(data, list):
                rec = data[333536]
            else:
                rec = data.get(333536) or data.get("333536")
            print(f"  Raw record type: {type(rec)}")
            print(f"  Raw record: {rec}")
    
    # Check if indices are consecutive starting from 0
    if all_indices:
        sorted_indices = sorted(all_indices)
        consecutive_from_zero = all(i == sorted_indices[i] for i in range(len(sorted_indices)))
        print(f"\nIndices consecutive from 0: {consecutive_from_zero}")
        if not consecutive_from_zero:
            print(f"  First gap at position: {next((i for i, idx in enumerate(sorted_indices) if i != idx), None)}")
    
    print("\n" + "="*60)
    return {
        "valid_count": len(valid_indices),
        "invalid_count": len(invalid_indices),
        "missing_count": len(missing_indices),
        "invalid_indices": invalid_indices,
        "missing_indices": missing_indices,
        "min_idx": min_idx,
        "max_idx": max_idx,
    }


def main():
    parser = argparse.ArgumentParser(description="Check retrieval.pkl for invalid SMILES")
    parser.add_argument("--retrieval", required=True, help="Path to retrieval.pkl")
    args = parser.parse_args()
    
    if not os.path.exists(args.retrieval):
        print(f"Error: File not found: {args.retrieval}")
        sys.exit(1)
    
    check_retrieval_index(args.retrieval)


if __name__ == "__main__":
    main()